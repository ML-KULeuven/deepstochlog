import itertools
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Union, Tuple

import typing

from torch import Tensor

from examples.data_utils import get_mnist_digits, get_next, split_train_dataset
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.term import Term, List

max_sequence_length = 150
tokens = [Term("t" + str(i)) for i in range(max_sequence_length)]
arguments_lists = [List(*tokens[0:i]) for i in range(max_sequence_length)]


def create_anbncn(n, letters=("a", "b", "c")):
    return n * letters[0] + n * letters[1] + n * letters[2]


def create_anbncn_language(
    min_length: int, max_length: int, letters=("a", "b", "c")
) -> typing.List[str]:
    if min_length < 3:
        min_length = 3
    if min_length % 3 != 0:
        print("Warning: min length should be divisible by 3, but was given", min_length)
        min_length = (min_length // 3) * 3
    result = [
        create_anbncn(i, letters) for i in range(min_length // 3, max_length // 3 + 1)
    ]
    return result


def create_all_possible_abc_sequences(
    length: int, letters=("a", "b", "c")
) -> typing.List[str]:
    result = []
    for a_num in range(1, length - 1):
        for b_num in range(1, length - a_num):
            for c_num in range(1, length - a_num - b_num + 1):
                result.append(
                    a_num * letters[0] + b_num * letters[1] + c_num * letters[2]
                )
    return result


def create_non_anbncn_language(
    min_length: int,
    max_length: int,
    letters=("a", "b", "c"),
    anbncn_language: typing.List[str] = None,
    allow_non_threefold: bool = False,
) -> typing.List[str]:
    if anbncn_language is None:
        anbncn_language = create_anbncn_language(min_length=3, max_length=max_length)
    anbncn_language_set = set(anbncn_language)
    return [
        el
        for el in create_all_possible_abc_sequences(max_length, letters=letters)
        if min_length <= len(el) <= max_length
        and (allow_non_threefold or len(el) % 3 == 0)
        and el not in anbncn_language_set
    ]


class ABCDataset(Sequence):
    def __init__(
        self,
        split: str = "train",
        size: int = None,
        min_length=3,
        max_length=21,
        seed: int = None,
        allowed_digits=(0, 1, 2),
        letters=("a", "b", "c"),
        all_permutations=True,
        allow_non_threefold=False,
        output_names=False,
        val_num_digits=None,
    ):
        permutations = list(itertools.permutations(letters))

        self.valid_language = [
            el
            for letters in permutations
            for el in create_anbncn_language(
                min_length=min_length, max_length=max_length, letters=letters
            )
        ]
        self.invalid_language = [
            el
            for letters in permutations
            for el in create_non_anbncn_language(
                min_length=min_length,
                max_length=max_length,
                letters=letters,
                anbncn_language=self.valid_language,
                allow_non_threefold=allow_non_threefold,
            )
        ]

        self.valid_language_idx = 0
        self.invalid_language_idx = 0
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.valid_language)
            random.shuffle(self.invalid_language)

        a_mnist, b_mnist, c_mnist = get_mnist_digits(
            train=False if split == "test" else True,
            digits=allowed_digits,
            output_names=output_names,
        )
        self.a_mnist = list(a_mnist)
        self.b_mnist = list(b_mnist)
        self.c_mnist = list(c_mnist)

        if val_num_digits is not None and (split == "train" or split == "val"):
            train = split == "train"
            self.a_mnist = split_train_dataset(
                dataset=self.a_mnist,
                val_num_digits=val_num_digits,
                train=train,
            )
            self.b_mnist = split_train_dataset(
                dataset=self.b_mnist,
                val_num_digits=val_num_digits,
                train=train,
            )
            self.c_mnist = split_train_dataset(
                dataset=self.c_mnist,
                val_num_digits=val_num_digits,
                train=train,
            )

        self.a_idx = 0
        self.b_idx = 0
        self.c_idx = 0

        if size is None:
            # Put an upper boundary on the length based on number of letters where every mnist is used at most once
            max_size = min(
                len(self.a_mnist), len(self.b_mnist), len(self.c_mnist)
            ) // max(1, max_length)
            size = max_size

        self.ct_term_dataset = []
        for idx in range(0, size // 2):
            pos_example = self.create_valid_example(output_names=output_names)
            self.ct_term_dataset.append(pos_example)
            if len(self.invalid_language) > 0:
                neg_example = self.create_invalid_example(output_names=output_names)
                self.ct_term_dataset.append(neg_example)

    def create_valid_example(self, output_names: bool = False) -> ContextualizedTerm:
        example_str = self._get_valid_example()
        example_tensors = [self.get_letter(ch) for ch in example_str]
        return create_term(
            True, example_tensors, example_str, output_names=output_names
        )

    def create_invalid_example(self, output_names: bool = False) -> ContextualizedTerm:
        example_str = self._get_invalid_example()
        example_tensors = [self.get_letter(ch) for ch in example_str]
        return create_term(
            False, example_tensors, example_str, output_names=output_names
        )

    def get_letter(self, ch: str) -> Tensor:
        if ch == "a":
            return self._get_a()
        if ch == "b":
            return self._get_b()
        if ch == "c":
            return self._get_c()
        raise RuntimeError("Not proper letter:", ch)

    def _get_a(self) -> Tensor:
        result, self.a_idx = get_next(self.a_idx, self.a_mnist)
        return result

    def _get_b(self) -> Tensor:
        result, self.b_idx = get_next(self.b_idx, self.b_mnist)
        return result

    def _get_c(self) -> Tensor:
        result, self.c_idx = get_next(self.c_idx, self.c_mnist)
        return result

    def _get_valid_example(self) -> str:
        result, self.valid_language_idx = get_next(
            self.valid_language_idx, self.valid_language
        )
        return result

    def _get_invalid_example(self) -> str:
        result, self.invalid_language_idx = get_next(
            self.invalid_language_idx, self.invalid_language
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
    original_sequence: str,
    output_names: bool = False,
) -> ContextualizedTerm:
    if output_names:
        return {
            "valid": 1 if valid else 0,
            "original_sequence": original_sequence,
            "sequence": tensors,
        }

    number_of_tokens = len(tensors)
    context_dict = dict()
    for token, tensor in zip(tokens[0:number_of_tokens], tensors):
        context_dict[token] = tensor
    return ContextualizedTerm(
        context=Context(context_dict),
        term=Term(
            "s", pos_label if valid else neg_label, arguments_lists[number_of_tokens]
        ),
        meta=original_sequence,
    )


def export_dataset(arguments: typing.Dict):
    """ Exports the data """
    folder = (
        Path(__file__).parent
        / ".."
        / ".."
        / "data"
        / "processed"
        / ("anbncn" + str(arguments["min_length"]) + "-" + str(arguments["max_length"]))
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
        ABCDataset(split="train", size=train_size, output_names=True, **arguments)
    )
    val_dataset = list(
        ABCDataset(split="val", size=val_size, output_names=True, **arguments)
    )
    test_dataset = list(
        ABCDataset(split="test", size=test_size, output_names=True, **arguments)
    )
    with open(folder / "train.json", "w") as json_file:
        json.dump(train_dataset, json_file, indent=4)
    with open(folder / "val.json", "w") as json_file:
        json.dump(val_dataset, json_file, indent=4)
    with open(folder / "test.json", "w") as json_file:
        json.dump(test_dataset, json_file, indent=4)


if __name__ == "__main__":

    max_lengths = [12, 15, 18]

    for max_length in max_lengths:
        export_dataset(
            {
                "min_length": 3,
                "max_length": max_length,
                "train_size": 4000,
                "val_size": 100,
                "test_size": 200,
                "val_num_digits": 300,
                "allow_non_threefold": False,
                "seed": 42,
            }
        )
