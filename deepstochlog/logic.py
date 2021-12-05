from abc import ABCMeta, ABC
from collections import Sized
from typing import Callable, Iterable
from deepstochlog.networkevaluation import RequiredEvaluation
from deepstochlog.term import Term


class LogicNode(Sized):
    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        raise NotImplementedError()

    def name(self):
        return type(self).__name__

    def __len__(self) -> int:
        return 1


class ListBasedNode(LogicNode, metaclass=ABCMeta):
    def __init__(self, *args):
        super().__init__()
        self.children = frozenset(args)

    def add_children(self, *arguments):
        self.children |= frozenset(self.children | set(arguments))

    def _string_children(self):
        return ",".join(str(c) for c in self.children)

    def __len__(self):
        return len(self.children)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, type(self)):
            return self.children == other.children
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash((type(self), self.children))


class And(ListBasedNode):
    def __init__(self, *args):
        super().__init__(*args)

    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        return visitor.visit_and(self)

    def __str__(self):
        return "And(" + self._string_children() + ")"

    def __repr__(self):
        return str(self)


class Or(ListBasedNode):
    def __init__(self, *args):
        super().__init__(*args)

    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        return visitor.visit_or(self)

    def __str__(self):
        return "Or(" + self._string_children() + ")"

    def __repr__(self):
        return str(self)


class NNLeaf(LogicNode):
    """
    Logic leaf node for calculating the probability using a neural network model
    """

    def __init__(self, network_model: str, index: int, inputs: Iterable):
        self.network = network_model
        self.index = index
        self.inputs = tuple(inputs)
        super().__init__()

    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        return visitor.visit_nnleaf(self)

    def __str__(self):
        return (
            "NNLeaf("
            + str(self.network)
            + ","
            + str(self.index)
            + ","
            + str(self.inputs)
            + ")"
        )

    def __repr__(self):
        return str(self)

    def name(self):
        return str((self.network, self.index, self.inputs))

    def __len__(self):
        return 1

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, NNLeaf):
            return (self.network, self.index, self.inputs) == (
                other.network,
                other.index,
                other.inputs,
            )
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash((self.network, self.index, self.inputs))

    def to_required_evaluation(self, context):
        return RequiredEvaluation(
            context=context, network_name=self.network, input_args=self.inputs
        )


class TermLeaf(LogicNode):
    """
    Term at the end of a logic node.
    Used for tabled logic trees
    """

    def __init__(self, term: Term):
        super().__init__()
        self.term = term

    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        return visitor.visit_termleaf(self)

    def __str__(self):
        return "TermLeaf(" + str(self.term) + ")"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, TermLeaf):
            return self.term == other.term
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.term)


class AbstractStaticProbability(LogicNode, ABC):
    def __init__(self, probability: float):
        self.probability = probability

    def accept_visitor(self, visitor: "LogicNodeVisitor"):
        return visitor.visit_static_probability(self)

    def __repr__(self):
        return str(self)


class StaticProbability(AbstractStaticProbability):
    def __init__(self, probability: float):
        super().__init__(probability)

    def __str__(self):
        return "StaticProbability(" + str(self.probability) + ")"

    def __eq__(self, other):
        """Overrides the default implementation"""
        return (
            isinstance(other, StaticProbability)
            and self.probability == other.probability
        )

    def __hash__(self):
        """Overrides the default implementation"""
        return 41 + hash(self.probability)


class AlwaysTrue(AbstractStaticProbability):
    def __init__(self):
        super().__init__(1.0)

    def __str__(self):
        return "AlwaysTrue()"

    def __eq__(self, other):
        """Overrides the default implementation"""
        return isinstance(other, AlwaysTrue)

    def __hash__(self):
        """Overrides the default implementation"""
        return 123


class AlwaysFalse(AbstractStaticProbability):
    def __init__(self):
        super().__init__(0.0)

    def __str__(self):
        return "AlwaysFalse()"

    def __eq__(self, other):
        """Overrides the default implementation"""
        return isinstance(other, AlwaysFalse)

    def __hash__(self):
        """Overrides the default implementation"""
        return 321
