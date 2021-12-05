from typing import Tuple, Callable, List, Dict
from time import time
from torch import nn

import torch
from deepstochlog.context import ContextualizedTerm
from deepstochlog.dataloader import DataLoader
from deepstochlog.utils import calculate_zipped_probabilities




class AccuracyCalculator():

    def __init__(self, model, valid, test, start_time, after_epoch = 0):
        self.model = model
        self.valid_set = valid
        self.test_set = test
        self.max_val = 0.
        self.current_test = 0.
        self.start_time = start_time
        self.header = "Valid acc\tTest acc\tP(correct)\ttime"
        self.after_epoch = after_epoch
        self.epoch = 0


    def __call__(self):

        self.epoch += 1
        if self.after_epoch< self.epoch:
            for network in self.model.neural_networks.networks.values():
                network.neural_model.eval()
            acc, average_right_prob= calculate_accuracy(self.model, self.valid_set)
            if acc > self.max_val:
                self.max_val = acc
                self.current_test, _ = calculate_accuracy(self.model, self.test_set)
            for network in self.model.neural_networks.networks.values():
                network.neural_model.train()
        else:
            acc = 0
            average_right_prob = 0

        return "{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(acc, self.current_test, average_right_prob,time() - self.start_time )






def create_model_accuracy_calculator(
    model, test_dataloader: DataLoader, start_time
) -> Tuple[str, Callable]:
    def calculate_model_accuracy() -> str:
        acc, average_right_prob = calculate_accuracy(model, test_dataloader)
        return "{:.3f}\t\t{:.3f}\t\t{:.3f}".format(
            acc, average_right_prob, time() - start_time
        )

    return "Test acc\tP(correct)\ttime", calculate_model_accuracy



def calculate_accuracy(
    model,
    test_data,
) -> Tuple[float, float]:
    if len(test_data) == 0:
        return 0, 0
    with torch.no_grad():
        test_acc = 0
        label_probability = 0
        for i, elem in enumerate(test_data):
            query_with_variable = test_data.queries_for_model[i]

            other_possibilities = [ContextualizedTerm(context=elem.context, term=t)
                                       for t in model.get_direct_proof_possibilities(query_with_variable)]

            # Map all results to queryable dict
            probabilities = dict()
            for possibility, prob in calculate_zipped_probabilities(model, other_possibilities):
                probabilities[possibility] = prob

            # Sum the probability of the labels
            own_prob = probabilities[elem]
            label_probability += own_prob

            # Check if it is the highest prediction out of all other possibilities
            is_highest = not any(
                [
                    pos
                    for pos in other_possibilities
                    if probabilities[pos] > own_prob
                ]
            )
            if is_highest:
                test_acc += 1

    return test_acc / len(test_data), label_probability / len(test_data)

class RuleWeights(nn.Module):
    def __init__(self, num_classes: 6, num_rules: int = 2):
        super(RuleWeights, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(num_classes, 2),
            nn.Softmax(-1)
        )
    def forward(self, x):
        x = x.view(-1)
        x = self.net(x)
        return x

class Influence(nn.Module):
    def __init__(self, num_documents: int,):
        super(Influence, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(num_documents, 5)
        )
    def forward(self, x,y,l):
        x = x.view(-1)
        embx = self.net(x)
        emby = self.net(l)
        mask = (l > 0).float() # compute the pad mask
        scores = torch.einsum("ij, ikj -> ik", embx, emby)
        masked_scores = torch.exp(scores) * mask
        scores = masked_scores / torch.sum(masked_scores, dim=-1, keepdim=True) # softmax only on the non padded elements
        res = scores[torch.arange(y.size(0)), y]
        return res


class Classifier(nn.Module):
    def __init__(
        self, input_size: int, num_outputs: int = 10, with_softmax: bool = True
    ):
        super(Classifier, self).__init__()
        self.with_softmax = with_softmax
        self.input_size = input_size
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.net = nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU(),
            nn.Linear(50, num_outputs)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.net(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


def pretraining(x,y, model,optimizer, epochs = 100):

    loss = nn.CrossEntropyLoss()
    for e in range(epochs):
        o = model(x)
        output = loss(o, y)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
