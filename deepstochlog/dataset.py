from abc import ABC
from collections.abc import Sequence
from typing import Set

from deepstochlog.context import ContextualizedTerm
from deepstochlog.term import Term, List


class ContextualizedTermDataset(Sequence, ABC):
    def calculate_queries(
        self, masked_generation_output=False, limit=None
    ) -> Set[Term]:
        """ Calculates which queries are necessary to ask Prolog based on which terms are in the dataset """
        queries = set()
        max_len = limit if limit is not None else len(self)
        for idx in range(max_len):
            elem: ContextualizedTerm = self[idx]
            if masked_generation_output:
                query = elem.term.mask_generation_output()
            else:
                query = elem.term
            if query not in queries:
                queries.add(query)
        return queries
