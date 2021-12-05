import random
from pathlib import Path
from time import time
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from context import ContextualizedTerm
from examples.anbncn.anbncn_data import ABCDataset
from examples.models import MNISTNet, LSTMSequenceImage, ImageEncoder

root = Path(__file__).parent
data_root = root / ".." / ".." / "data" / "processed"


def get_term_datasets(
    allow_non_threefold: bool,
    data_seed: int,
    max_length: int,
    min_length: int,
    test_size: int,
    train_size: int,
    val_num_digits: int,
    val_size: int,
) -> Tuple[ABCDataset, ABCDataset, ABCDataset]:
    train_data = ABCDataset(
        split="train",
        size=train_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=data_seed,
        val_num_digits=val_num_digits,
    )
    val_data = ABCDataset(
        split="val",
        size=val_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=data_seed,
        val_num_digits=val_num_digits,
    )
    test_data = ABCDataset(
        split="test",
        size=test_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=data_seed,
    )
    return train_data, val_data, test_data


def convert_term_to_sequence(
    ct: ContextualizedTerm, device
) -> Tuple[float, torch.Tensor]:
    label = float(ct.term.arguments[0].functor)
    tensor_tokens = ct.term.arguments[1]
    tensors = [
        ct.context.get_tensor_representation(token).to(device)
        for token in tensor_tokens
    ]
    tensor_sequence = torch.stack(tensors).to(device)
    return label, tensor_sequence


def convert_terms_to_sequence(
    terms: ABCDataset,
    device,
    shuffle=False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if shuffle:
        terms = list(terms)
        random.shuffle(terms)
    tuples = [convert_term_to_sequence(term, device=device) for term in terms]
    labels, sequences = zip(*tuples)
    labels = torch.tensor(labels, device=device)
    return labels, sequences


def should_log(self, batch_idx, dataloader, epoch, epochs):
    return batch_idx % self.log_freq == 0 or (
        epoch == epochs - 1 and batch_idx % len(dataloader) == len(dataloader) - 1
    )


def run(
    min_length: int = 3,
    max_length: int = 21,
    allow_non_threefold: bool = False,
    #
    epochs=1,
    batch_size=32,
    lr=1e-3,
    loss_function=nn.BCELoss(),
    iterations=1,
    #
    image_encoder_size=200,
    rnn_hidden_size=200,
    #
    train_size=None,
    val_size=None,
    test_size=None,
    val_num_digits=300,
    #
    log_freq=5,
    data_seed=None,
):
    # Load the MNIST model, and Adam optimiser
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_terms, val_terms, test_terms = get_term_datasets(
        allow_non_threefold=allow_non_threefold,
        data_seed=data_seed,
        max_length=max_length,
        min_length=min_length,
        test_size=test_size,
        train_size=train_size,
        val_num_digits=val_num_digits,
        val_size=val_size,
    )
    train_labels, train_data = convert_terms_to_sequence(
        train_terms, shuffle=True, device=device
    )
    val_labels, val_data = convert_terms_to_sequence(
        val_terms, shuffle=True, device=device
    )
    test_labels, test_data = convert_terms_to_sequence(
        test_terms, shuffle=True, device=device
    )

    def calculate_accuracy(
        model, targets: torch.Tensor, data: List[torch.Tensor]
    ) -> float:
        predicted_targets: torch.Tensor = model.forward(data)
        predicted_labels = (predicted_targets >= 0.5).float()
        correct = sum(targets == predicted_labels)
        return float(correct) / len(data)

    best_test_accs = np.zeros(iterations)
    for i in range(iterations):
        image_encoder = ImageEncoder(output_features=image_encoder_size)
        rnn = LSTMSequenceImage(image_encoder, image_encoder_size, rnn_hidden_size)
        rnn.to(device=device)
        optimizer = Adam(rnn.parameters(), lr=lr)

        def get_val_accuracy():
            return calculate_accuracy(rnn, val_labels, val_data)

        def get_test_accuracy():
            return calculate_accuracy(rnn, test_labels, test_data)

        best_test_acc, best_val_acc = perform_training(
            batch_size,
            epochs,
            get_test_accuracy,
            get_val_accuracy,
            log_freq,
            loss_function,
            optimizer,
            rnn,
            train_data,
            train_labels,
        )
        print()
        print("Best validation acc", best_val_acc)
        print("Best test acc", best_test_acc)
        best_test_accs[i] = best_test_acc

    print("\n\nResults:")
    print("All:", best_test_accs)
    print("Mean:", np.mean(best_test_accs))
    print("Std:", np.std(best_test_accs))


def perform_training(
    batch_size,
    epochs,
    get_test_accuracy,
    get_val_accuracy,
    log_freq,
    loss_function,
    optimizer,
    rnn,
    train_data,
    train_labels,
):
    total_batches = 0
    total_loss = 0
    instances_since_last_log = 0
    print("Epoch\tBatch\tLoss\t\tVal\t\tTest\ttime")
    start_time = time()
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(epochs):
        number_of_batches = (len(train_labels) + batch_size - 1) // batch_size
        for batch_idx in range(number_of_batches):
            batch_end = min(len(train_labels), (batch_idx + 1) * batch_size)
            batch_data = train_data[batch_idx * batch_size : batch_end]
            batch_labels = train_labels[batch_idx * batch_size : batch_end]

            # Cross-Entropy (CE) loss
            predicted = rnn.forward(batch_data)
            loss = loss_function(predicted, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save loss for printing
            total_loss += float(loss)
            instances_since_last_log += len(batch_labels)

            # Print the loss every log_freq batches, as well as the final batch
            if batch_idx % log_freq == 0 or (
                epoch == epochs - 1
                and ((batch_idx + 1) * batch_size >= len(train_data))
            ):
                val_acc = get_val_accuracy()
                test_acc = get_test_accuracy()

                # Store best one so far
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                print(
                    "{}\t\t{}\t\t{:.5f}\t\t{:.3f}\t{:.3f}\t{:.2f}".format(
                        epoch,
                        batch_idx,
                        total_loss / instances_since_last_log,
                        val_acc,
                        test_acc,
                        time() - start_time,
                    )
                )
                total_loss = 0
                instances_since_last_log = 0

            total_batches += 1
    return best_test_acc, best_val_acc


if __name__ == "__main__":
    run(
        min_length=3,
        max_length=12,
        allow_non_threefold=False,
        val_num_digits=300,
        #
        # train_size=2,
        # val_size=2,
        # test_size=2,
        train_size=4000,
        val_size=100,
        test_size=200,
        #
        log_freq=10,
        epochs=1,
        iterations=5,
        data_seed=42,
    )
