from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from time import time

from examples.bracket.bracket_data import BracketDataset
from deepstochlog.network import Network, NetworkStore
from examples.models import MNISTNet
from deepstochlog.nn_models import TrainableProbability
from deepstochlog.utils import (
    set_fixed_seed,
    create_run_test_query,
    create_model_accuracy_calculator,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer, print_logger

root_path = Path(__file__).parent


def run(
    min_length: int = 2,
    max_length: int = 8,
    allow_uneven: float = False,
    only_bracket_language_examples=True,
    #
    epochs=2,
    batch_size=32,
    lr=1e-3,
    #
    train_size=None,
    val_size=None,
    test_size=None,
    # allow_negative_during_training=True,
    # allow_negative_during_testing=True,
    #
    log_freq=50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    val_batch_size=100,
    val_num_digits=1000,
    #
    seed=None,
    set_program_seed=True,
    verbose=True,
    verbose_building=False,
):
    start_time = time()
    # Setting seed for reproducibility
    if set_program_seed:
        set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_network = MNISTNet(output_features=2)

    # Create a network object, containing the MNIST network and the index list
    mnist_classifier = Network(
        "bracket_nn", mnist_network, index_list=[Term("("), Term(")")]
    )
    networks = NetworkStore(mnist_classifier)

    # Load the model "addition.pl" with this MNIST network
    possible_lengths = range(min_length, max_length + 1, 1 if allow_uneven else 2)
    if only_bracket_language_examples:
        queries = [
            Term("s", List(*[Term("t" + str(i)) for i in range(length)]))
            for length in possible_lengths
        ]
    else:
        queries = [
            Term("s", Term("_"), List(*[Term("t" + str(i)) for i in range(length)]))
            for length in possible_lengths
        ]
    grounding_start_time = time()
    model = DeepStochLogModel.from_file(
        file_location=str(
            (
                root_path
                / (
                    "bracket.pl"
                    if only_bracket_language_examples
                    else "bracket_all.pl"
                )
            ).absolute()
        ),
        query=queries,
        networks=networks,
        device=device,
        verbose=verbose_building,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()

    grounding_time = time() - grounding_start_time
    if verbose:
        print("Grounding the program took {:.3} seconds".format(grounding_time))

    train_data = BracketDataset(
        split="train",
        size=train_size,
        min_length=min_length,
        max_length=max_length,
        allow_uneven=allow_uneven,
        seed=seed,
        val_num_digits=val_num_digits,
        only_bracket_language_examples=only_bracket_language_examples,
    )
    val_data = BracketDataset(
        split="val",
        size=val_size,
        min_length=min_length,
        max_length=max_length,
        allow_uneven=allow_uneven,
        seed=seed,
        val_num_digits=val_num_digits,
        only_bracket_language_examples=only_bracket_language_examples,
    )
    test_data = BracketDataset(
        split="test",
        size=test_size,
        min_length=min_length,
        max_length=max_length,
        allow_uneven=allow_uneven,
        seed=seed,
        only_bracket_language_examples=only_bracket_language_examples,
    )

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)
    val_dataloader = DataLoader(val_data, val_batch_size)

    # Create test functions
    # run_test_query = create_run_test_query_probability(
    #     model, test_data, test_example_idx, verbose
    # )
    # calculate_model_accuracy = create_probability_accuracy_calculator(
    #     model,
    #     test_dataloader,
    #     start_time,
    #     threshold=threshold,
    #     validation_data=val_dataloader,
    #     most_probable_parse_accuracy=True,
    # )
    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
        generation_output_accuracy=False,
    )
    calculate_model_accuracy = create_model_accuracy_calculator(
        model,
        test_dataloader,
        start_time,
        generation_output_accuracy=False,
        most_probable_parse_accuracy=True,
        val_dataloader=val_dataloader if val_num_digits > 0 else None,
    )

    # Train the DeepStochLog model
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=calculate_model_accuracy,
        logger=logger,
        test_query=run_test_query,
        print_time=verbose,
    )
    train_time = trainer.train(
        model=model, optimizer=optimizer, dataloader=train_dataloader, epochs=epochs
    )

    return {
        "grounding_time": grounding_time,
        "training_time": train_time,
        "model": model,
    }


if __name__ == "__main__":
    run(
        min_length=2,
        max_length=10,
        train_size=1000,
        val_size=200,
        test_size=200,
        allow_uneven=False,
        epochs=1,
        test_example_idx=[0, 1, 2, 3],
        seed=42,
        log_freq=10,
        only_bracket_language_examples=True,
    )
