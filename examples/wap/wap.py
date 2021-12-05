from pathlib import Path

import torch
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import (
    set_fixed_seed,
    create_run_test_query,
    create_model_accuracy_calculator,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from examples.wap.wap_data import WapDataset
from examples.wap.wap_network import RNN, vocab
from examples.models import MLP

root_path = Path(__file__).parent


def run(
    #
    epochs=2,
    batch_size=32,
    lr=1e-3,
    hidden_size=512,
    n=8,
    p_drop=0.5,
    #
    train_size=None,
    val_size=None,
    test_size=None,
    #
    log_freq=50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    val_batch_size=None,
    #
    seed=None,
    verbose=True,
    verbose_building=False,
):
    start_time = time()
    # Setting seed for reproducibility
    set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    operators = [Term(e) for e in ["plus", "minus", "times", "div"]]

    rnn_encoder = RNN(len(vocab), hidden_size, device=device, p_drop=p_drop)
    nn_permute = Network(
        "nn_permute",
        MLP(hidden_size * n, 6, encoder=rnn_encoder),
        index_list=[Term(str(e)) for e in range(6)],
        concat_tensor_input=False,
    )
    nn_op1 = Network(
        "nn_op1",
        MLP(hidden_size * n, 4, encoder=rnn_encoder),
        index_list=operators,
        concat_tensor_input=False,
    )
    nn_swap = Network(
        "nn_swap",
        MLP(hidden_size * n, 2, encoder=rnn_encoder),
        index_list=[Term(e) for e in ["no_swap", "swap"]],
        concat_tensor_input=False,
    )
    nn_op2 = Network(
        "nn_op2",
        MLP(hidden_size * n, 4, encoder=rnn_encoder),
        index_list=operators,
        concat_tensor_input=False,
    )

    networks = NetworkStore(nn_permute, nn_op1, nn_swap, nn_op2)

    train_data = WapDataset(
        split="train",
        size=train_size,
    )
    val_data = WapDataset(
        split="dev",
        size=val_size,
    )
    test_data = WapDataset(
        split="test",
        size=test_size,
    )

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        val_data, batch_size=val_batch_size if val_batch_size else test_batch_size
    )
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    queries = train_data.calculate_queries(
        masked_generation_output=False
    ) | test_data.calculate_queries(masked_generation_output=True)

    grounding_start_time = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "wap.pl").absolute()),
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

    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
        parse_is_nnleaf_outputs=True,
    )
    calculate_model_accuracy = create_model_accuracy_calculator(
        model=model,
        test_dataloader=test_dataloader,
        start_time=start_time,
        val_dataloader=val_dataloader,
        most_probable_parse_accuracy=False,
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
        epochs=40,
        test_example_idx=[0, 1, 2, 3],
        # train_size=4,
        # val_size=4,
        # test_size=4,
        seed=42,
        log_freq=10,
        # verbose_building=False,
    )
