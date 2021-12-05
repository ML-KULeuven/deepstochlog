import typing
from typing import Iterable, List, Union, Collection

import torch

from deepstochlog.context import ContextualizedTerm
from deepstochlog.network import NetworkStore
from deepstochlog.logic import Or, And, TermLeaf, LogicNode
from deepstochlog.networkevaluation import RequiredEvaluation, NetworkEvaluations
from deepstochlog.tabled_and_or_trees import TabledAndOrTrees, LogicProbabilityEvaluator
from deepstochlog.tabled_tree_builder import TabledAndOrTreeBuilder
from deepstochlog.term import Term
from deepstochlog.parser import parse_rules
from deepstochlog.inferences import (
    SumProductVisitor,
    MaxProductVisitor,
    TermLeafDescendantsRetriever,
)


class DeepStochLogModel:

    NO_NORM = "NO_NORM"
    LEN_NORM = "LEN_NORM"
    FULL_NORM = "FULL_NORM"

    def __init__(
        self,
        trees: TabledAndOrTrees,
        neural_networks: NetworkStore,
        normalization=None,
        device=None,
    ):
        self.neural_networks = neural_networks
        self.device = device
        self.trees = trees
        self.normalization = self._set_normalization(normalization)
        if device is not None:
            self.to_device(device)

    def _set_normalization(self, normalization):
        if normalization is None:
            normalization = DeepStochLogModel.NO_NORM
        if normalization not in (
            DeepStochLogModel.NO_NORM,
            DeepStochLogModel.LEN_NORM,
            DeepStochLogModel.FULL_NORM,
        ):
            raise ValueError("Normalization %s unknown." % str(normalization))
        return normalization

    def to_device(self, *args, **kwargs):
        """ Allows to put the networks on the GPU """
        self.neural_networks.to_device(*args, **kwargs)
        self.device = args[0]

    def get_all_net_parameters(self):
        return self.neural_networks.get_all_net_parameters()

    def compute_normalization_constant(
        self, probability_evaluator, contextualized_term
    ):

        if self.normalization == DeepStochLogModel.NO_NORM:
            Z = 1.0
        elif self.normalization == DeepStochLogModel.LEN_NORM:
            raise NotImplementedError("Length based normalization not implemented")
        else:  # self.normalization == DeepStochLogModel.FULL_NORM
            Z = probability_evaluator.accept_term_visitor(
                term=contextualized_term.term.mask_generation_output(),
                visitor=SumProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
            )
        return Z

    def predict_sum_product(self, batch: Iterable[ContextualizedTerm]) -> torch.Tensor:

        probability_evaluator = self.create_probability_evaluator(batch)
        tensors = []
        for contextualized_term in batch:
            p = probability_evaluator.accept_term_visitor(
                term=contextualized_term.term,
                visitor=SumProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
            )
            # p = probability_evaluator.evaluate_term_sum_product_probability(
            #     term=contextualized_term.term, context=contextualized_term.context
            # )
            Z = self.compute_normalization_constant(
                probability_evaluator, contextualized_term
            )
            tensors.append(p / Z)

        return torch.stack(tensors)

    def predict_max_product_parse(
        self, batch: Iterable[ContextualizedTerm]
    ) -> typing.List[typing.Tuple[torch.Tensor, Iterable[LogicNode]]]:
        probability_evaluator = self.create_probability_evaluator(batch)

        predictions: typing.List[typing.Tuple[torch.Tensor, Iterable[LogicNode]]] = [
            TermLeaf(term=contextualized_term.term).accept_visitor(
                visitor=MaxProductVisitor(
                    probability_evaluator=probability_evaluator,
                    context=contextualized_term.context,
                ),
            )
            # probability_evaluator.evaluate_term_leaf_max_product_probability(
            #     term_leaf=TermLeaf(contextualized_term.term),
            #     context=contextualized_term.context,
            # )
            for contextualized_term in batch
        ]
        return predictions

    def create_probability_evaluator(self, batch: Iterable[ContextualizedTerm]):
        # Find all required evaluations: flatmap from req evaluations from each contextualized term
        required_evaluations: List[RequiredEvaluation] = list(
            {
                re
                for ct in batch
                for re in self.trees.calculate_required_evaluations(
                    contextualized_term=ct
                )
            }
        )
        # Evaluate all required evaluations on the neural networks
        network_evaluations: NetworkEvaluations = (
            NetworkEvaluations.from_required_evaluations(
                required_evaluations=required_evaluations,
                networks=self.neural_networks,
                device=self.device,
            )
        )
        # Calculate the probabilities using the evaluated networks results.
        probability_evaluator = LogicProbabilityEvaluator(
            trees=self.trees,
            network_evaluations=network_evaluations,
            device=self.device,
        )
        return probability_evaluator

    def calculate_probability(self, contextualized_term: ContextualizedTerm):
        """ For easy access from test cases, and demonstrating. Not really used in core model """
        probability_evaluator = self.create_probability_evaluator([contextualized_term])
        p = probability_evaluator.accept_term_visitor(
            term=contextualized_term.term,
            visitor=SumProductVisitor(
                probability_evaluator=probability_evaluator,
                context=contextualized_term.context,
            ),
        )
        Z = self.compute_normalization_constant(
            probability_evaluator, contextualized_term
        )
        return p / Z

    @staticmethod
    def from_string(
        program_str: str,
        networks: NetworkStore,
        query: Union[Term, Iterable[Term]] = None,
        device=None,
        verbose=False,
        add_default_zero_probability=False,
        prolog_facts: str = "",
        tabling=True,
        normalization=None,
    ):
        deepstochlog_rules = parse_rules(program_str)
        deepstochlog_rules, extra_networks = deepstochlog_rules.remove_syntactic_sugar()

        networks = networks + extra_networks

        builder = TabledAndOrTreeBuilder(
            deepstochlog_rules,
            verbose=verbose,
            prolog_facts=prolog_facts,
            tabling=tabling,
        )
        tabled_and_or_trees = builder.build_and_or_trees(
            networks=networks,
            queries=query,
            add_default_zero_probability=add_default_zero_probability,
        )
        return DeepStochLogModel(
            trees=tabled_and_or_trees,
            neural_networks=networks,
            device=device,
            normalization=normalization,
        )

    @staticmethod
    def from_file(
        file_location: str,
        networks: NetworkStore,
        query: Union[Term, Iterable[Term]] = None,
        device=None,
        add_default_zero_probability=False,
        verbose=False,
        prolog_facts: str = "",
        tabling=True,
        normalization=None,
    ) -> "DeepStochLogModel":
        with open(file_location) as f:
            lines = f.readlines()
        return DeepStochLogModel.from_string(
            "\n".join(lines),
            query=query,
            networks=networks,
            device=device,
            verbose=verbose,
            add_default_zero_probability=add_default_zero_probability,
            prolog_facts=prolog_facts,
            tabling=tabling,
            normalization=normalization,
        )

    def mask_and_get_direct_proof_possibilities(self, term: Term) -> List[Term]:
        if term.can_mask_generation_output():
            return self.get_direct_proof_possibilities(term.mask_generation_output())
        else:
            return [term]

    def get_direct_proof_possibilities(self, term: Term) -> List[Term]:
        term_tree: LogicNode = self.trees.get_and_or_tree(term)
        term_leafs: Iterable[TermLeaf] = term_tree.accept_visitor(
            visitor=TermLeafDescendantsRetriever()
        )
        terms = [tl.term for tl in term_leafs]
        return terms

    def get_direct_contextualized_proof_possibilities(
        self, ct: ContextualizedTerm
    ) -> List[ContextualizedTerm]:
        return [
            ContextualizedTerm(context=ct.context, term=t)
            for t in self.mask_and_get_direct_proof_possibilities(ct.term)
        ]
