from pathlib import Path
from typing import Union
from shutil import copy2
import torch
import numpy as np
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from examples.cora.with_rule_weights.cora_data_withrules import train_dataset, valid_dataset, test_dataset, queries_for_model, citations
from examples.citeseer.citeseer_utils import create_model_accuracy_calculator, Classifier, RuleWeights, AccuracyCalculator
from deepstochlog.utils import set_fixed_seed
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List

root_path = Path(__file__).parent


def run(
    epochs=100,
    batch_size=32,
    lr=0.01,
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




    set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    input_size = len(train_dataset.documents[0])
    classifier = Classifier(input_size=input_size)
    rule_weights = RuleWeights(num_rules=2, num_classes=7)
    classifier_network = Network(
        "classifier",
        classifier,
        index_list=[Term("class"+str(i)) for i in range(7)],
    )
    rule_weight = Network(
        "rule_weight",
        rule_weights,
        index_list=[Term(str("neural")), Term(str("cite"))],
    )
    networks = NetworkStore(classifier_network, rule_weight)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    proving_start = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "cora_ruleweights.pl").absolute()),
        query=queries_for_model,
        networks=networks,
        device=device,
        prolog_facts= citations,
        normalization=DeepStochLogModel.FULL_NORM
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()
    proving_time = time() - proving_start

    if verbose:
        logger.print("\nProving the program took {:.2f} seconds".format(proving_time))



    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # Create test functions
    # run_test_query = create_run_test_query_probability(
    #     model, test_data, test_example_idx, verbose
    # )
    calculate_model_accuracy = AccuracyCalculator( model=model,
                                                    valid=valid_dataset,
                                                    test=test_dataset,
                                                    start_time=time())    # run_test_query = create_run_test_query(model, test_data, test_example_idx, verbose)
    # calculate_model_accuracy = create_model_accuracy_calculator(model, test_dataloader, start_time)
    # g = GreedyEvaluation(valid_data, test_data, networks)
    # calculate_model_accuracy = "Acc", GreedyEvaluation(documents, labels, networks)

    # Train the DeepStochLog model
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=(calculate_model_accuracy.header, calculate_model_accuracy),
        logger=logger,
        print_time=verbose,
    )
    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
    )

    return None


if __name__ == "__main__":


      run(
            test_example_idx=0,
            expression_max_length=1,
            expression_length=1,
            epochs=200,
            batch_size=len(train_dataset),
            seed=0,
            log_freq=1,
            allow_division=True,
            verbose=True,
        )
