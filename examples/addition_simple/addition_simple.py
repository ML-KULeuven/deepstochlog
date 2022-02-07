from pathlib import Path
from typing import Iterable, Sequence, Union, Tuple

from torch.tensor import Tensor

from deepstochlog.context import Context, ContextualizedTerm
import torch
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from examples.data_utils import get_mnist_data
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

root_path = Path(__file__).parent


t1 = Term("t1")
t2 = Term("t2")
argument_sequence = List(t1, t2)


class SimpleAdditionDataset(Sequence):
    def __init__(self, train: bool, digit_length=1, size: int = None):
        self.mnist_dataset = get_mnist_data(train)
        self.ct_term_dataset = []
        size = len(self.mnist_dataset) // 2
        for idx in range(0, 2 * size, 2):
            mnist_datapoint_1: Tuple[Tensor, int] = self.mnist_dataset[idx]
            mnist_datapoint_2: Tuple[Tensor, int] = self.mnist_dataset[idx + 1]
            digit_1 = mnist_datapoint_1[1]
            digit_2 = mnist_datapoint_2[1]
            total_sum = mnist_datapoint_1[1] + mnist_datapoint_2[1]

            addition_term = ContextualizedTerm(
                # Load context with the tensors
                context=Context({t1: mnist_datapoint_1[0], t2: mnist_datapoint_2[0]}),
                # Create the term containing the sum and a list of tokens representing the tensors
                term=Term(
                    "addition",
                    Term(str(total_sum)),
                    argument_sequence,
                ),
                meta=str(digit_1) + "+" + str(digit_2),
            )
            self.ct_term_dataset.append(addition_term)

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]


def run(
    epochs=2,
    batch_size=32,
    lr=1e-3,
    #
    train_size=None,
    val_size=500,
    test_size=None,
    #
    log_freq=100,
    logger=print_logger,
    test_example_idx=None,
    val_batch_size=100,
    test_batch_size=100,
    #
    verbose=True,
    seed=None,
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

    # Load the model "addition_simple.pl" with the specific query
    query = Term(
        # We want to use the "addition" non-terminal in the specific grammar
        "addition",
        # We want to calculate all possible sums by giving the wildcard "_" as argument
        Term("_"),
        # Denote that the input will be a list of two tensors, t1 and t2, representing the MNIST digit.
        List(Term("t1"), Term("t2")),
    )
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "addition_simple.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
        verbose=verbose,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()

    train_and_val_data = SimpleAdditionDataset(
        True,
        digit_length=1,
        size=train_size + val_size if train_size else None,
    )
    val_data = list(train_and_val_data[:val_size])
    train_data = list(train_and_val_data[val_size:])
    test_data = SimpleAdditionDataset(False, digit_length=1, size=test_size)

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=val_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    # Create test functions
    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
    )
    calculate_model_accuracy = create_model_accuracy_calculator(
        model,
        test_dataloader,
        start_time,
        val_dataloader=val_dataloader,
    )

    # Train the DeepStochLog model
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=calculate_model_accuracy,
        logger=logger,
        print_time=verbose,
        test_query=run_test_query,
    )

    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
    )

    print("Done running DeepStochLog simple addition example")


if __name__ == "__main__":
    run(
        test_example_idx=[0, 1, 2],
        seed=42,
        verbose=True,
        val_size=500,
        train_size=5000,
        test_size=100,
        epochs=2,
    )
