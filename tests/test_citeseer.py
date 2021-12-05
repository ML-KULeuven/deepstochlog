import unittest

import examples.bracket.bracket
from examples.bracket.bracket_data import (
    create_bracket_language_of_length,
    create_bracket_language,
    BracketDataset,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel

# from utils import calculate_probability_predictions


from pathlib import Path
from typing import Union
from shutil import copy2
import torch
import numpy as np
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from examples.citeseer.with_rule_weights.citeseer_data_withrules import (
    train_dataset,
    valid_dataset,
    test_dataset,
    queries_for_model,
    citations,
    pretraining_data,
)
from examples.citeseer.citeseer_utils import (
    create_model_accuracy_calculator,
    Classifier,
    RuleWeights,
    AccuracyCalculator,
    pretraining,
)
from deepstochlog.utils import set_fixed_seed
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List

root_path = Path(__file__).parent

root = Path(__file__).parent
with open(
    root
    / ".."
    / "examples"
    / "citeseer"
    / "with_rule_weights"
    / "citeseer_ruleweights.pl",
    "r",
) as file:
    citeseer_program = file.read()


class CiteseerTest(unittest.TestCase):
    def test_citeseer_and_or_tree(self):

        # Load the MNIST model, and Adam optimiser
        input_size = len(train_dataset.documents[0])
        classifier = Classifier(input_size=input_size)
        rule_weights = RuleWeights(num_rules=2, num_classes=6)
        classifier_network = Network(
            "classifier",
            classifier,
            index_list=[Term("class" + str(i)) for i in range(6)],
        )
        rule_weight = Network(
            "rule_weight",
            rule_weights,
            index_list=[Term(str("neural")), Term(str("cite"))],
        )
        networks = NetworkStore(classifier_network, rule_weight)

        citations = """
            cite(0,1).
            cite(1,3)."""

        queries_for_model = [Term("s", Term("_"), List("0"))]
        # ,Term("s", Term("class4"), List("1"))]

        model = DeepStochLogModel.from_string(
            program_str=citeseer_program,
            query=queries_for_model,
            networks=networks,
            prolog_facts=citations,
            normalization=DeepStochLogModel.FULL_NORM,
        )

        tree = model.trees

        print(tree)


if __name__ == "__main__":
    unittest.main()
