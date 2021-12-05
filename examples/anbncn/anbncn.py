from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from time import time

from examples.anbncn.anbncn_data import ABCDataset
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

root_path = Path(__file__).parent


def run(
    min_length: int = 3,
    max_length: int = 21,
    allow_non_threefold: float = False,
    #
    epochs=1,
    batch_size=32,
    lr=1e-3,
    #
    train_size=None,
    val_size=None,
    test_size=None,
    #
    log_freq=50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    val_batch_size=100,
    val_num_digits=300,
    most_probable_parse_accuracy=False,
    #
    seed=None,
    set_program_seed=True,
    verbose=True,
    verbose_building=False,
):
    if max_length < min_length:
        raise RuntimeError(
            "Max length can not be larger than minimum length:"
            + str(max_length)
            + "<"
            + str(min_length)
        )

    start_time = time()
    # Setting seed for reproducibility
    if set_program_seed:
        set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mnist_network = MNISTNet(output_features=3)

    # Create a network object, containing the MNIST network and the index list
    mnist_classifier = Network(
        "mnist", mnist_network, index_list=[Term("a"), Term("b"), Term("c")]
    )
    networks = NetworkStore(mnist_classifier)

    # Load the model "addition.pl" with this MNIST network
    queries = [
        Term("s", Term("_"), List(*[Term("t" + str(i)) for i in range(length)]))
        for length in range(min_length, max_length + 1)
        if allow_non_threefold or length % 3 == 0
    ]
    grounding_start_time = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "anbncn.pl").absolute()),
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

    train_data = ABCDataset(
        split="train",
        size=train_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=seed,
        val_num_digits=val_num_digits,
    )
    val_data = ABCDataset(
        split="val",
        size=val_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=seed,
        val_num_digits=val_num_digits,
    )
    # if val_size is None:
    #     val_size = train_size // 10
    # val_data = ABCDataset(
    #     split="train",
    #     size=val_size,
    #     min_length=min_length,
    #     max_length=max_length,
    #     allow_non_threefold=allow_non_threefold,
    #     seed=seed,
    # )
    test_data = ABCDataset(
        split="test",
        size=test_size,
        min_length=min_length,
        max_length=max_length,
        allow_non_threefold=allow_non_threefold,
        seed=seed,
    )

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(val_data, batch_size=val_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)
    # val_dataloader = DataLoader(
    #     val_data, val_batch_size if val_batch_size is not None else val_size
    # )

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
    #     most_probable_parse_accuracy=False,
    # )

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
        most_probable_parse_accuracy=most_probable_parse_accuracy,
        val_dataloader=val_dataloader,
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
        "proving_time": grounding_time,
        "neural_time": train_time,
        "model": model,
    }


if __name__ == "__main__":
    run(
        min_length=3,
        max_length=12,
        train_size=4000,
        val_size=100,
        test_size=200,
        val_num_digits=300,
        allow_non_threefold=False,
        log_freq=10,
        epochs=1,
        test_example_idx=[0, 1, 2, 3],
        seed=42,
        verbose_building=False,
    )
