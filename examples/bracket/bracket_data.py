import json
import random
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Union

import typing

from torch import Tensor

from examples.data_utils import split_train_dataset, get_next
from examples.data_utils import get_mnist_digits
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.term import Term, List

max_sequence_length = 30
tokens = [Term("t" + str(i)) for i in range(max_sequence_length)]
arguments_lists = [List(*tokens[0:i]) for i in range(max_sequence_length)]


@lru_cache()
def create_bracket_language_of_length(exact_length: int) -> typing.List[str]:
    if exact_length < 2:
        return []
    if exact_length == 2:
        return ["()"]
    if exact_length % 2 == 1:
        raise RuntimeWarning(
            "Creating a bracket length of length "
            + str(exact_length)
            + " is impossible"
        )
        # return []

    result = []

    smaller_length = exact_length - 2
    # Add brackets around smaller one
    for smaller in create_bracket_language_of_length(smaller_length):
        result.append("(" + smaller + ")")

    # Put smaller ones next to each other
    for first_size in range(2, exact_length, 2):
        first_possibilities = create_bracket_language_of_length(first_size)
        second_size = exact_length - first_size
        second_possibilities = create_bracket_language_of_length(second_size)
        for first in first_possibilities:
            for second in second_possibilities:
                result.append(first + second)
    result = list(set(result))
    result.sort()
    return result


def create_bracket_language(min_length: int, max_length: int) -> typing.List[str]:
    if min_length < 2:
        min_length = 2
    if min_length % 2 != 0:
        print("Warning: min length should be even, but was given", min_length)
        min_length += 1
    result = []
    for i in range(min_length, max_length + 1, 2):
        result.extend(create_bracket_language_of_length(exact_length=i))
    return result


def create_all_possible_bracket_sequences(max_length: int) -> typing.List[str]:
    elements = [""]
    last_new_elements = elements
    while len(elements[len(elements) - 1]) <= max_length - 1:
        last_new_elements = [
            new_els
            for element in last_new_elements
            for new_els in (element + ")", element + "(")
        ]
        elements.extend(last_new_elements)
    return elements


def create_non_bracket_language(
    min_length: int,
    max_length: int,
    bracket_language: typing.List[str] = None,
    allow_uneven: bool = True,
) -> typing.List[str]:
    all_brackets = [
        el
        for el in create_all_possible_bracket_sequences(max_length=max_length)
        if len(el) >= min_length
    ]
    if bracket_language is None:
        bracket_language = create_bracket_language(
            min_length=min_length, max_length=max_length
        )
    bracket_language_set = set(bracket_language)
    return [
        el
        for el in all_brackets
        if el not in bracket_language_set and (allow_uneven or len(el) % 2 == 0)
    ]


class BracketDataset(Sequence):
    def __init__(
        self,
        split: str = "train",
        size: int = None,
        min_length=0,
        max_length=10,
        seed: int = None,
        open_bracket_digit=(0, 1, 2, 3, 4),
        closed_bracket_digit=(5, 6, 7, 8, 9),
        allow_uneven=True,
        output_names=False,
        val_num_digits=0,
        only_bracket_language_examples=True,
    ):
        self.valid_bracket_language = create_bracket_language(
            min_length=min_length, max_length=max_length
        )
        self.invalid_bracket_language = create_non_bracket_language(
            min_length=min_length,
            max_length=max_length,
            bracket_language=self.valid_bracket_language,
            allow_uneven=allow_uneven,
        )
        if min_length > 0:
            self.valid_bracket_language = [
                el for el in self.valid_bracket_language if len(el) >= min_length
            ]
            self.invalid_bracket_language = [
                el for el in self.invalid_bracket_language if len(el) >= min_length
            ]

        self.valid_bracket_idx = 0
        self.invalid_bracket_idx = 0
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.valid_bracket_language)
            random.shuffle(self.invalid_bracket_language)

        open_brackets_mnist = get_mnist_digits(
            train=False if split == "test" else True,
            digits=open_bracket_digit,
            output_names=output_names,
        )
        open_brackets_mnist = [i for sub in open_brackets_mnist for i in sub]
        closed_bracket_mnist = get_mnist_digits(
            train=False if split == "test" else True,
            digits=closed_bracket_digit,
            output_names=output_names,
        )
        closed_bracket_mnist = [i for sub in closed_bracket_mnist for i in sub]
        self.open_brackets_mnist = list(open_brackets_mnist)
        self.closed_bracket_mnist = list(closed_bracket_mnist)

        # Filter out validation
        if val_num_digits is not None and (split == "train" or split == "val"):
            train = split == "train"
            self.open_brackets_mnist = split_train_dataset(
                dataset=self.open_brackets_mnist,
                val_num_digits=val_num_digits,
                train=train,
            )
            self.closed_bracket_mnist = split_train_dataset(
                dataset=self.closed_bracket_mnist,
                val_num_digits=val_num_digits,
                train=train,
            )

        self.open_brackets_idx = 0
        self.closed_brackets_idx = 0

        if size is None:
            # Put an upper boundary on the length based on number of brackets where every mnist is used at most once
            max_size = min(
                len(self.open_brackets_mnist), len(self.closed_bracket_mnist)
            ) // max(1, max_length)
            size = max_size

        self.ct_term_dataset = []
        for idx in range(0, size // 2):
            pos_example = self.create_valid_bracket_example(
                output_names=output_names,
                only_bracket_language_examples=only_bracket_language_examples,
            )
            self.ct_term_dataset.append(pos_example)
            if not only_bracket_language_examples:
                neg_example = self.create_invalid_bracket_example(
                    output_names=output_names,
                )
                self.ct_term_dataset.append(neg_example)

    def create_valid_bracket_example(
        self, output_names: bool = False, only_bracket_language_examples=True
    ) -> ContextualizedTerm:
        example_str = self._get_valid_bracket_example()
        example_tensors = [self.get_bracket(ch) for ch in example_str]
        return create_term(
            True,
            example_tensors,
            example_str,
            output_names=output_names,
            only_bracket_language_examples=True,
        )

    def create_invalid_bracket_example(
        self, output_names: bool = False
    ) -> ContextualizedTerm:
        example_str = self._get_invalid_bracket_example()
        example_tensors = [self.get_bracket(ch) for ch in example_str]
        return create_term(
            False,
            example_tensors,
            example_str,
            output_names=output_names,
            only_bracket_language_examples=False,
        )

    def get_bracket(self, ch: str) -> Tensor:
        if ch == "(":
            return self._get_open_bracket()
        if ch == ")":
            return self._get_closed_bracket()
        raise RuntimeError("Not proper bracket:", ch)

    def _get_open_bracket(self) -> Tensor:
        result, self.open_brackets_idx = get_next(
            self.open_brackets_idx, self.open_brackets_mnist
        )
        return result

    def _get_closed_bracket(self) -> Tensor:
        result, self.closed_brackets_idx = get_next(
            self.closed_brackets_idx, self.closed_bracket_mnist
        )
        return result

    def _get_valid_bracket_example(self) -> str:
        result, self.valid_bracket_idx = get_next(
            self.valid_bracket_idx, self.valid_bracket_language
        )
        return result

    def _get_invalid_bracket_example(self) -> str:
        result, self.invalid_bracket_idx = get_next(
            self.invalid_bracket_idx, self.invalid_bracket_language
        )
        return result

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]


neg_label = Term("0")
pos_label = Term("1")


def create_term(
    valid: bool,
    tensors: typing.List[Tensor],
    original_brackets: str,
    output_names: bool = False,
    only_bracket_language_examples=True,
) -> ContextualizedTerm:
    if output_names:
        return {
            "valid": 1 if valid else 0,
            "original_sequence": original_brackets,
            "sequence": tensors,
        }

    number_of_tokens = len(tensors)
    context_dict = dict()
    for token, tensor in zip(tokens[0:number_of_tokens], tensors):
        context_dict[token] = tensor

    if only_bracket_language_examples:
        term = Term("s", arguments_lists[number_of_tokens])
    else:
        term = Term(
            "s", pos_label if valid else neg_label, arguments_lists[number_of_tokens]
        )

    return ContextualizedTerm(
        context=Context(context_dict),
        term=term,
        meta=original_brackets,
    )


def export_dataset(arguments: typing.Dict):
    """ Exports the bracket data """
    folder = (
        Path(__file__).parent
        / ".."
        / ".."
        / "data"
        / "processed"
        / (
            "brackets"
            + str(arguments["min_length"])
            + "-"
            + str(arguments["max_length"])
        )
    )
    folder.mkdir(exist_ok=True, parents=True)
    with open(folder / "config.json", "w") as json_file:
        json.dump(arguments, json_file, indent=4)

    train_size = arguments["train_size"]
    val_size = arguments["val_size"]
    test_size = arguments["test_size"]
    del arguments["train_size"]
    del arguments["val_size"]
    del arguments["test_size"]

    train_dataset = list(
        BracketDataset(split="train", size=train_size, output_names=True, **arguments)
    )
    val_dataset = list(
        BracketDataset(split="val", size=val_size, output_names=True, **arguments)
    )
    test_dataset = list(
        BracketDataset(split="test", size=test_size, output_names=True, **arguments)
    )
    with open(folder / "train.json", "w") as json_file:
        json.dump(train_dataset, json_file, indent=4)
    with open(folder / "val.json", "w") as json_file:
        json.dump(val_dataset, json_file, indent=4)
    with open(folder / "test.json", "w") as json_file:
        json.dump(test_dataset, json_file, indent=4)


if __name__ == "__main__":

    max_lengths = [10, 14, 18]

    for max_length in max_lengths:
        export_dataset(
            {
                "min_length": 2,
                "max_length": max_length,
                "train_size": 1000,
                "val_size": 200,
                "test_size": 200,
                "val_num_digits": 1000,
                "allow_uneven": False,
                "only_bracket_language_examples": True,
                "seed": 42,
            }
        )
