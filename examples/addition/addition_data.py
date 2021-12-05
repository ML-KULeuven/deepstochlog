from collections.abc import Sequence
from typing import Union

import typing
from torch import Tensor

from examples.data_utils import get_mnist_data
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.term import Term, List


def calculate_number_from_digits(digit_tensors: typing.Tuple[int, ...]):
    digits = [d for d in digit_tensors]
    numbers_base_10 = sum(
        digit * pow(10, len(digit_tensors) - i - 1) for (i, digit) in enumerate(digits)
    )
    return numbers_base_10


class AdditionDataset(Sequence):
    def __init__(self, train: bool, digit_length=1, size: int = None):
        self.mnist_dataset = get_mnist_data(train)
        self.ct_term_dataset = []
        size = self.calculate_dataset_size(size=size, digit_length=digit_length)
        for idx in range(0, 2 * digit_length * size, 2 * digit_length):
            data_points = self.get_mnist_datapoints(digit_length, idx)
            tensors, digits = zip(*data_points)
            number_1 = calculate_number_from_digits(digits[:digit_length])
            number_2 = calculate_number_from_digits(digits[digit_length:])
            total_sum = number_1 + number_2
            addition_term = create_addition_term(
                total_sum=total_sum, digit_length=digit_length, tensors=tensors, number_1=number_1, number_2=number_2
            )
            self.ct_term_dataset.append(addition_term)

    def get_mnist_datapoints(self, digit_length, idx):
        return [self.mnist_dataset[i] for i in range(idx, idx + 2 * digit_length)]

    def calculate_dataset_size(self, size: int, digit_length: int):
        return (
            len(self.mnist_dataset) // (digit_length * 2)
            if size is None or size > (len(self.mnist_dataset) // 2)
            else size
        )

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]


max_digit_length = 4
terms = [Term("t" + str(i)) for i in range(1, 2 * max_digit_length + 1)]


def create_argument_list(digit_length: int):
    args = []
    for i in range(digit_length):
        args.append(terms[i])
        args.append(terms[digit_length + i])
    return List(*args)


argument_lists = [
    create_argument_list(digit_length=digit_length)
    for digit_length in range(max_digit_length + 1)
]
length_arguments = [Term(str(i)) for i in range(max_digit_length + 1)]


def create_addition_term(
    total_sum: int,
    digit_length,
    tensors: typing.Iterable[Tensor],
    number_1: int,
    number_2: int,
):
    context = {}
    argument_list = argument_lists[digit_length]
    for i, tensor in enumerate(tensors):
        context[terms[i]] = tensor

    return ContextualizedTerm(
        context=Context(context),
        term=Term(
            "multi_addition",
            Term(str(total_sum)),
            length_arguments[digit_length],
            argument_list,
        ),
        meta=str(number_1) + "+" + str(number_2),
    )
