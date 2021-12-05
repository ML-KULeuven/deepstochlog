import random
import typing
from pathlib import Path
from typing import List, Tuple, Generator

import torch
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"


def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root / "raw/"), train=train, download=True, transform=transform
    )


def get_mnist_digits(
    train: bool, digits: List[int], output_names: bool = False
) -> Tuple[Generator, ...]:
    dataset = get_mnist_data(train)
    if not output_names:
        return tuple(
            (
                dataset[i][0]
                for i in (dataset.targets == digit).nonzero(as_tuple=True)[0]
            )
            for digit in digits
        )
    prefix = "train_" if train else "test_"
    return tuple(
        (prefix + str(i.item()) for i in (dataset.targets == digit).nonzero(as_tuple=True)[0])
        for digit in digits
    )


def split_train_dataset(dataset: typing.List, val_num_digits: int, train: bool):
    if train:
        return dataset[val_num_digits:]
    else:
        return dataset[:val_num_digits]


def get_next(idx: int, elements: typing.MutableSequence) -> Tuple[any, int]:
    if idx >= len(elements):
        idx = 0
        random.shuffle(elements)
    result = elements[idx]
    idx += 1
    return result, idx