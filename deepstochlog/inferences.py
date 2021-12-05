# These classes describe visitors that change the way the and/or trees are inferenced
import abc
import itertools
from functools import reduce
from typing import Tuple, Iterable, Callable

import torch

from deepstochlog.context import Context
from deepstochlog.logic import (
    And,
    Or,
    NNLeaf,
    TermLeaf,
    AbstractStaticProbability,
    LogicNode,
    ListBasedNode,
)
from deepstochlog.term import Term


def multiply(x, y):
    return x * y


def multiply_iterable(iterable):
    return reduce(multiply, iterable)


class LogicNodeVisitor(abc.ABC):
    def visit_and(self, node: And):
        raise NotImplementedError()

    def visit_or(self, node: Or):
        raise NotImplementedError()

    def visit_nnleaf(self, node: NNLeaf):
        raise NotImplementedError()

    def visit_termleaf(self, node: TermLeaf):
        raise NotImplementedError()

    def visit_static_probability(self, node: AbstractStaticProbability):
        raise NotImplementedError()


# == SEMIRING BASED VISITORS ==


class SemiringVisitor(LogicNodeVisitor, abc.ABC):
    def __init__(
        self, probability_evaluator: "LogicProbabilityEvaluator", context: Context
    ):
        self.probability_evaluator = probability_evaluator
        self.context = context

    def visit_nnleaf(self, node: NNLeaf):
        return self.probability_evaluator.evaluate_neural_network_probability(
            node.network, node.inputs, node.index, self.context
        )

    def visit_termleaf(self, node: TermLeaf):
        return self.probability_evaluator.accept_term_visitor(node.term, visitor=self)

    def visit_static_probability(self, node: AbstractStaticProbability):
        return torch.as_tensor(
            node.probability, device=self.probability_evaluator.device
        )


class SumProductVisitor(SemiringVisitor):
    def __init__(
        self, probability_evaluator: "LogicProbabilityEvaluator", context: Context
    ):
        super().__init__(probability_evaluator, context)

    def visit_and(self, node: And):
        return multiply_iterable((x.accept_visitor(self) for x in node.children))

    def visit_or(self, node: Or):
        return sum(x.accept_visitor(self) for x in node.children)


class MaxProductVisitor(SemiringVisitor):
    def __init__(
        self, probability_evaluator: "LogicProbabilityEvaluator", context: Context
    ):
        super().__init__(probability_evaluator, context)

    def get_children_max_product_probability_and_parse(
        self, node: ListBasedNode
    ) -> Tuple[Iterable[torch.Tensor], Iterable[Iterable[LogicNode]]]:
        values: Iterable[Tuple[torch.Tensor, Iterable[LogicNode]]] = (
            x.accept_visitor(self) for x in node.children
        )
        zipped: Tuple[Iterable[torch.Tensor], Iterable[Iterable[LogicNode]]] = zip(
            *values
        )
        return zipped

    def visit_and(self, node: And) -> Tuple[torch.Tensor, Iterable["LogicNode"]]:
        tensors, logic_nodes = self.get_children_max_product_probability_and_parse(node)
        return multiply_iterable(tensors), (
            x for sublist in logic_nodes for x in sublist
        )

    def visit_or(self, node: Or) -> Tuple[torch.Tensor, Iterable["LogicNode"]]:
        tensors, logic_nodes = self.get_children_max_product_probability_and_parse(node)
        tensors = list(tensors)
        logic_nodes = list(logic_nodes)
        max_idx = torch.argmax(torch.stack(tensors))

        return tensors[max_idx], logic_nodes[max_idx]

    def visit_nnleaf(self, node: NNLeaf) -> Tuple[torch.Tensor, Iterable["LogicNode"]]:
        return (
            super().visit_nnleaf(node),
            [node],
        )

    def visit_termleaf(
        self, node: TermLeaf
    ) -> Tuple[torch.Tensor, Iterable[LogicNode]]:
        prob, logic_nodes = super().visit_termleaf(node)
        return prob, itertools.chain([node], logic_nodes)

    def visit_static_probability(self, node: AbstractStaticProbability):
        return (
            super().visit_static_probability(node),
            [node],
        )


# ==  LEAF DESCENDANTS INSPECTORS ==


class NNLeafDescendantsRetriever(LogicNodeVisitor):
    def visit_list_based(self, node: ListBasedNode) -> Iterable["NNLeaf"]:
        neural_leaf_descendents = [c.accept_visitor(self) for c in node.children]
        # Flatmap
        return [item for descendents in neural_leaf_descendents for item in descendents]

    def visit_and(self, node: And) -> Iterable["NNLeaf"]:
        return self.visit_list_based(node)

    def visit_or(self, node: Or) -> Iterable["NNLeaf"]:
        return self.visit_list_based(node)

    def visit_nnleaf(self, node: NNLeaf) -> Iterable["NNLeaf"]:
        return (node,)

    def visit_termleaf(self, node: TermLeaf) -> Iterable["NNLeaf"]:
        return ()

    def visit_static_probability(
        self, node: AbstractStaticProbability
    ) -> Iterable["NNLeaf"]:
        return []


class TermLeafDescendantsRetriever(LogicNodeVisitor):
    def visit_list_based(self, node: ListBasedNode) -> Iterable["TermLeaf"]:
        term_leaf_descendants = [c.accept_visitor(self) for c in node.children]
        # Flatmap
        return [item for descendents in term_leaf_descendants for item in descendents]

    def visit_and(self, node: And) -> Iterable["TermLeaf"]:
        return self.visit_list_based(node)

    def visit_or(self, node: Or) -> Iterable["TermLeaf"]:
        return self.visit_list_based(node)

    def visit_nnleaf(self, node: NNLeaf) -> Iterable["TermLeaf"]:
        return ()

    def visit_termleaf(self, node: TermLeaf) -> Iterable["TermLeaf"]:
        return (node,)

    def visit_static_probability(
        self, node: AbstractStaticProbability
    ) -> Iterable["TermLeaf"]:
        return []


class DescendantTermMapper(LogicNodeVisitor):
    def __init__(self, mapper: Callable[[Term], Term]):
        self.mapper = mapper

    def visit_list_based(self, node: ListBasedNode, logic_node_class) -> "LogicNode":
        return logic_node_class(
            *[child.accept_visitor(self) for child in node.children]
        )

    def visit_and(self, node: And) -> "LogicNode":
        return self.visit_list_based(node, And)

    def visit_or(self, node: Or) -> "LogicNode":
        return self.visit_list_based(node, Or)

    def visit_nnleaf(self, node: NNLeaf) -> "LogicNode":
        return node

    def visit_termleaf(self, node: TermLeaf) -> "LogicNode":
        return TermLeaf(self.mapper(node.term))

    def visit_static_probability(self, node: AbstractStaticProbability) -> "LogicNode":
        return node

    def accept_raw_nn_leaf(self, node: "RawNNLeaf"):

        new_nn_term = self.mapper(node.term)
        network_name = new_nn_term.arguments[0].functor
        index = node.networks.get_network(network_name).term2idx(
            new_nn_term.arguments[1]
        )

        result = NNLeaf(
            network_name,
            index,
            new_nn_term.arguments[2],
        )
        return result


class SortedPrinter(LogicNodeVisitor):
    def visit_list_based(self, node: ListBasedNode, logic_node_class) -> str:
        children = [child.accept_visitor(self) for child in node.children]
        children.sort()
        return str(logic_node_class) + "(" + ", ".join(children) + ")"

    def visit_and(self, node: And) -> str:
        return self.visit_list_based(node, And)

    def visit_or(self, node: Or) -> str:
        return self.visit_list_based(node, Or)

    def visit_nnleaf(self, node: NNLeaf) -> str:
        return str(node)

    def visit_termleaf(self, node: TermLeaf) -> str:
        return str(node)

    def visit_static_probability(self, node: AbstractStaticProbability) -> str:
        return str(node)
