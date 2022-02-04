from pathlib import Path
from typing import Iterable
import numpy as np

from examples.addition.addition_data import AdditionDataset, argument_lists
import torch
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from examples.models import MNISTNet
from deepstochlog.utils import (
    set_fixed_seed,
    create_run_test_query,
    create_model_accuracy_calculator,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.logic import NNLeaf, LogicNode

root_path = Path(__file__).parent


class GreedyEvaluation:
    def __init__(self, digit_length, valid_data, test_data, store, device):

        self.digit_length = digit_length
        self.valid_data = valid_data
        self.test_data = test_data
        self.store = store
        self.header = "Val acc\tTest acc"
        self.max_val = 0.0
        self.test_acc = 0.0
        self.device = device

    def _addition(self, l, n):
        res = 0
        for i in range(len(l) // 2):
            n = n - 1
            li = [l[i], l[i + self.digit_length]]
            res = res + sum(li) * (10 ** (n))
        return int(res)

    def _acc(self, data, number):
        evaluations = []

        for term in data:
            res = []
            for i, (id, tensor) in enumerate(term.context._context.items()):
                n = torch.argmax(
                    number.neural_model(tensor.unsqueeze(dim=0).to(self.device)).cpu()
                ).numpy()
                res.append(n)

            res = self._addition(res, self.digit_length)
            ground = int(term.term.arguments[0].functor)
            evaluations.append(int(ground == res))
        s = np.mean(evaluations)
        return s

    def __call__(self):
        number = self.store.networks["number"]
        number.neural_model.eval()

        valid_acc = self._acc(self.valid_data, number)
        if valid_acc >= self.max_val:
            self.test_acc = self._acc(self.test_data, number)
            self.max_val = valid_acc

        number.neural_model.train()
        return "%s\t%s\t" % (str(valid_acc), str(self.test_acc))


def create_parse(term: Term, logic_node: Iterable[LogicNode], networks: NetworkStore):
    elements = dict()
    for nnleaf in logic_node:
        # elements[nnleaf.inputs[0]] = networks.get_network(nnleaf.network).idx2term(
        #     nnleaf.index
        # )
        if isinstance(nnleaf, NNLeaf):
            elements[nnleaf.inputs[0].functor] = str(nnleaf.index)

    digits = []
    for token in sorted(elements.keys()):
        digits.append(elements[token])
    result = (
        "".join(digits[: len(digits) // 2]) + "+" + "".join(digits[len(digits) // 2 :])
    )
    return result


def run(
    digit_length=1,
    epochs=2,
    batch_size=32,
    lr=1e-3,
    greedy=False,
    #
    train_size=None,
    val_size=500,
    test_size=None,
    #
    log_freq=100,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    val_batch_size=100,
    most_probable_parse_accuracy=False,
    #
    seed=None,
    verbose=False,
    verbose_building=False,
):
    start_time = time()
    # Setting seed for reproducibility
    set_fixed_seed(seed)

    # Create a network object, containing the MNIST network and the index list
    mnist_classifier = Network(
        "number", MNISTNet(), index_list=[Term(str(i)) for i in range(10)]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    networks = NetworkStore(mnist_classifier)

    # Load the model "addition.pl" with this MNIST network
    query = Term(
        "multi_addition",
        Term("_"),
        Term(str(digit_length)),
        argument_lists[digit_length],
    )

    grounding_start = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "addition.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
        verbose=verbose_building,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()

    grounding_time = time() - grounding_start
    if verbose:
        print("Grounding took", grounding_time, "seconds")

    train_and_val_data = AdditionDataset(
        True,
        digit_length=digit_length,
        size=train_size + val_size if train_size else None,
    )
    val_data = list(train_and_val_data[:val_size])
    train_data = list(train_and_val_data[val_size:])
    test_data = AdditionDataset(False, digit_length=digit_length, size=test_size)

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=val_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    if greedy:
        g = GreedyEvaluation(digit_length, val_data, test_data, networks, device=device)
        calculate_model_accuracy = g.header, g

        # Train the DeepStochLog model
        trainer = DeepStochLogTrainer(
            log_freq=log_freq,
            accuracy_tester=calculate_model_accuracy,
            logger=logger,
            print_time=verbose,
        )

    else:
        # Create test functions
        run_test_query = create_run_test_query(
            model=model,
            test_data=test_data,
            test_example_idx=test_example_idx,
            verbose=verbose,
            create_parse=create_parse,
        )
        calculate_model_accuracy = create_model_accuracy_calculator(
            model,
            test_dataloader,
            start_time,
            val_dataloader=val_dataloader,
            create_parse=create_parse,
            most_probable_parse_accuracy=most_probable_parse_accuracy,
        )

        # Train the DeepStochLog model
        trainer = DeepStochLogTrainer(
            log_freq=log_freq,
            accuracy_tester=calculate_model_accuracy,
            logger=logger,
            print_time=verbose,
            test_query=run_test_query,
        )

    end_time = trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
    )

    return {
        "proving_time": grounding_time,
        "neural_time": end_time,
    }


if __name__ == "__main__":
    run(
        test_example_idx=[0, 1, 2, 3, 4],
        seed=42,
        verbose=True,
        val_size=500,
        train_size=5000,
        test_size=100,
        digit_length=1,
        epochs=2,
        greedy=True,
    )
