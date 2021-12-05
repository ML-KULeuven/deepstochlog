from typing import Dict, List, Union

import torch
from torch import Tensor

from deepstochlog.term import Term


class Context:
    """ Represents the context of a query: maps logic terms to tensors """

    def __init__(self, context: Dict[Term, Tensor], map_default_to_term=False):
        self._context = context
        self._hash = hash(tuple(sorted(self._context.items())))
        self._map_default_to_term = map_default_to_term

    def has_tensor_representation(self, term: Term) -> bool:
        return term in self._context

    def get_tensor_representation(self, term: Term) -> Union[Tensor, str]:
        """
        Returns the tensor representation, unless it doesn't contain it, then it turns just the functor
        """
        if self._map_default_to_term and not self.has_tensor_representation(term):
            return term.functor
        if term.is_list():
            return torch.cat(
                [self.get_tensor_representation(a) for a in term.arguments]
            )

        return self._context[term]

    def get_all_tensor_representations(self, network_input_args) -> List[Tensor]:
        return [self.get_tensor_representation(term) for term in network_input_args]

    def __eq__(self, other):
        if isinstance(other, Context):
            return self._context == other._context
        return False

    def __hash__(self):
        return self._hash


class ContextualizedTerm:
    def __init__(
        self, context: Context, term: Term, probability: float = 1.0, meta=None
    ):
        self.context = context
        self.term = term
        self.probability = probability
        self.meta = meta

    def __str__(self):
        return "ContextualizedTerm(" + str(self.term) + ")"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, ContextualizedTerm):
            return self.context == other.context and self.term == other.term
        return False

    def __hash__(self):
        return hash((self.context, self.term))

    def mask_generation_output(self):
        return ContextualizedTerm(
            term=self.term.mask_generation_output(),
            context=self.context,
        )
