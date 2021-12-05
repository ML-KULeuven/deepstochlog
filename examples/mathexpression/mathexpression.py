from pathlib import Path
from typing import Union
from shutil import copy2
import torch
import numpy as np
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from examples.mathexpression.mathexpression_data import (
    MathExprDataset,
    operator_word_list,
    mathexpression_dataset_max_seq_length,
    create_our_splits,
)
from examples.models import SymbolEncoder, SymbolClassifier
from deepstochlog.utils import (
    calculate_accuracy,
    test_single_instance,
    set_fixed_seed,
    create_model_accuracy_calculator,
    create_run_test_query,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List

root_path = Path(__file__).parent


def create_expression_sentence_query(
    number_img: int, total_sum: Union[str, float] = "_"
):
    """ Generates sentence query like s(_, [img1,img2,img3,img4,img5,img6,img7], [], _)"""
    total_sum = Term(str(total_sum))
    images_arg = List(*[Term(f"img{i + 1}") for i in range(number_img)])
    return Term("expression", total_sum, images_arg)


class GreedyDumbEvaluation:
    def __init__(self, valid_data, test_data, store):

        self.valid_data = valid_data
        self.test_data = test_data
        self.store = store
        self.header = "Valid acc\tTest acc\t"
        self.max_val = 0.0
        self.test_acc = 0.0

    def _acc(self, data, number, operator):
        operators = ["+", "-", "*", "/"]
        evaluations = []

        for term in data:
            res = ""
            for i, (id, tensor) in enumerate(term.context._context.items()):
                if i % 2 == 0:
                    n = torch.argmax(
                        number.neural_model(tensor.unsqueeze(dim=0))
                    ).numpy()
                    res = res + str(n)
                else:
                    o = torch.argmax(
                        operator.neural_model(tensor.unsqueeze(dim=0))
                    ).numpy()
                    res = res + operators[o]
            try:
                res = eval(res)
                ground = term.term.arguments[0]
                evaluations.append(int(ground == res))
            except:
                evaluations.append(False)
        s = np.mean(evaluations)
        return s

    def __call__(self):
        number, operator = (
            self.store.networks["number"],
            self.store.networks["operator"],
        )
        number.neural_model.eval()
        operator.neural_model.eval()

        valid_acc = self._acc(self.valid_data, number, operator)
        if valid_acc >= self.max_val:
            self.test_acc = self._acc(self.test_data, number, operator)
            self.max_val = valid_acc

        number.neural_model.train()
        operator.neural_model.train()
        return "%s\t%s\t" % (str(valid_acc), str(self.test_acc))


def load_expression_networks(lr=1e-3):
    encoder = SymbolEncoder()

    number_network = Network(
        "number",
        SymbolClassifier(encoder, N=10),
        index_list=[Term(str(i)) for i in range(10)],
    )
    operator_network = Network(
        "operator",
        SymbolClassifier(encoder, N=4),
        index_list=[Term(op) for op in operator_word_list],
    )

    return NetworkStore(number_network, operator_network)


def run(
    epochs=100,
    batch_size=32,
    lr=0.003,
    expression_length=None,
    expression_max_length=3,
    allow_division=True,
    device_str: str = None,
    #
    train_size=None,
    test_size=None,
    #
    log_freq=50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    #
    seed=None,
    verbose=False,
):
    # args = epochs,batch_size,lr,expression_length,expression_max_length,allow_division,\
    #        train_size,test_size,log_freq,logger,test_example_idx,test_batch_size,seed,verbose
    #
    # logger.print("\n".join(["{}={}".format(a,b) for a in args ]) TODO
    start_time = time()
    # Setting seed for reproducibility
    set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    networks = load_expression_networks()

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model "addition.pl" with this MNIST network
    max_length = (
        expression_max_length
        if expression_max_length
        else mathexpression_dataset_max_seq_length
    )

    query = [
        create_expression_sentence_query(number_img=length, total_sum="_")
        for length in range(1, max_length + 1, 2)
    ]

    proving_start = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "mathexpression.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()
    proving_time = time() - proving_start
    if verbose:
        logger.print("\nProving the program took {:.2f} seconds".format(proving_time))

    if expression_length is not None:
        train_data = MathExprDataset(
            split="train",
            num_samples=train_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
        valid_data = MathExprDataset(
            split="val",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
        test_data = MathExprDataset(
            split="test",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
    else:
        train_data = MathExprDataset(
            split="train",
            num_samples=train_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )
        valid_data = MathExprDataset(
            split="val",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )
        test_data = MathExprDataset(
            split="test",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    # Create test functions
    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
    )
    # calculate_model_accuracy = create_model_accuracy_calculator(model, test_dataloader, start_time)
    g = GreedyDumbEvaluation(valid_data, test_data, networks)
    calculate_model_accuracy = g.header, g

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

    logger.print("Best val accuracy:" + str(g.max_val))
    logger.print("Test accuracy:" + str(g.test_acc))

    return g.test_acc


if __name__ == "__main__":

    import os

    create_our_splits()
    logs = "logs_temp"
    if not os.path.exists(logs):
        os.mkdir(logs)
    copy2(__file__, logs)
    copy2("mathexpression_data.py", logs)
    copy2("mathexpression.pl", logs)

    for seed in [0, 1, 2, 3, 4]:
        folder = os.path.join(logs, "%d" % seed)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for l in [3]:
            logger = PrintFileLogger(os.path.join(folder, "exact_%d.txt" % l))
            res = run(
                test_example_idx=0,
                expression_max_length=l,
                expression_length=l,
                epochs=20,
                batch_size=2,
                seed=seed,
                logger=logger,
                log_freq=100,
                allow_division=True,
                verbose=True,
                device_str="cpu"
            )
