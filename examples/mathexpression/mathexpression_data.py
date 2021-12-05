import json
import os
import random
from abc import ABCMeta
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import typing
from PIL import Image
from torchvision import transforms

from deepstochlog import term
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.dataset import ContextualizedTermDataset
from deepstochlog.term import Term

operator_word_list = ["plus", "minus", "times", "div"]
operator_list = ["+", "-", "*", "/"]
digit_list = [str(i) for i in range(1, 10)]
unknown_representation = "UNK"
all_symbols_list = [unknown_representation] + digit_list + operator_list


def symbol2id(sym):
    return all_symbols_list.index(sym)


unknown_idx = symbol2id(unknown_representation)
digit_idx_list = [symbol2id(x) for x in digit_list]
operator_idx_list = [symbol2id(x) for x in operator_list]
root_dir = Path(__file__).parent / ".." / ".." / "data" / "raw" / "hwf"

mathexpression_dataset_max_seq_length = 13


def create_filter_function(
    expression_length=None, expression_max_length=None, allow_division=None
):
    validity_functions = []
    if expression_length is not None:
        validity_functions.append(lambda sample: sample["len"] == expression_length)
    if expression_max_length is not None:
        validity_functions.append(lambda sample: sample["len"] <= expression_max_length)
    if allow_division is not None and not allow_division:
        validity_functions.append(lambda sample: "/" not in sample["expr"])

    return lambda sample: all(is_valid(sample) for is_valid in validity_functions)


def calculate_rounded_result(sample):
    expression_result = eval(sample["expr"])

    # Round if necessary
    expression_result = (
        int(expression_result)
        if isinstance(expression_result, int) or expression_result.is_integer()
        # else (int(expression_result * 100) / 100)
        else expression_result
    )

    return expression_result


class AbstractMathExprDataset(Sequence, metaclass=ABCMeta):
    def __init__(
        self,
        split: str = 'train',
        num_samples=None,
        random_seed=None,
        expression_length=None,
        expression_max_length=None,
        allow_division=None,
    ):
        super(AbstractMathExprDataset, self).__init__()

        # Load in the whole dataset
        base_path = root_dir
        self.imgs_path = base_path / "Handwritten_Math_Symbols"
        self.split = split
        assert split in ['train', 'val', 'test']
        self.dataset = json.load(
            open(
                os.path.join(
                    base_path, "expr_{}.json".format(split)
                )
            )
        )
        resulting_dataset = []
        self.tensors = {}

        # Shuffle if required
        if random_seed:
            random.seed(random_seed)
            random.shuffle(self.dataset)

        # Transform to images
        self.image_transform = transforms.Compose(
            [
                # transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1,)),
            ]
        )

        is_valid_sample = create_filter_function(
            expression_length=expression_length,
            expression_max_length=expression_max_length,
            allow_division=allow_division,
        )
        for example in self.dataset:
            sample_id = example["id"]
            example["len"] = len(example["expr"])

            if not is_valid_sample(example):
                continue

            sample = deepcopy(example)

            sample["res"] = calculate_rounded_result(sample)

            img_seq = []
            for idx, img_path in enumerate(sample["img_paths"]):
                img = Image.open(self.imgs_path / img_path).convert("L")

                img = self.image_transform(img)
                img_id = str(sample_id) + "_" + str(idx)
                self.tensors[img_id] = img
                img_seq.append(img_id)
            del sample["img_paths"]

            # Store sequence
            label_seq = [symbol2id(sym) for sym in sample["expr"]]
            sample["image_sequence"] = img_seq
            sample["label_sequence"] = label_seq
            sample["len"] = len(sample["expr"])

            resulting_dataset.append(sample)
            if len(resulting_dataset) == num_samples:
                break

        self.dataset: typing.List[Dict[str, any]] = resulting_dataset

    def __len__(self):
        return len(self.dataset)


class MathExprDataset(AbstractMathExprDataset, ContextualizedTermDataset):
    def __init__(
        self,
        split: str = 'train',
        num_samples=None,
        random_seed=None,
        expression_length=None,
        expression_max_length=None,
        allow_division=None,
    ):
        super().__init__(
            split=split,
            num_samples=num_samples,
            random_seed=random_seed,
            expression_length=expression_length,
            expression_max_length=expression_max_length,
            allow_division=allow_division,
        )

        # Initialize terms used to denote token sequence
        max_length = (
            expression_max_length
            if expression_max_length
            else mathexpression_dataset_max_seq_length
        )
        terms = [Term("img" + str(i + 1)) for i in range(max_length)]

        self._img_token_sequences: Dict[int, typing.List[Term]] = dict()
        for length in range(1, max_length + 1, 2):
            self._img_token_sequences[length] = [terms[idx] for idx in range(length)]

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        item_dict = self.dataset[item]
        images = item_dict["image_sequence"]
        expression_result = item_dict["res"]
        context_dict = dict()
        img_sequence = self._img_token_sequences[len(images)]
        for idx, img_nr in enumerate(img_sequence):
            context_dict[img_nr] = self.tensors[images[idx]]
        return ContextualizedTerm(
            context=Context(context_dict),
            term=Term("expression", expression_result, term.List(*img_sequence)),
        )



class MathExprDataset(AbstractMathExprDataset, ContextualizedTermDataset):
    def __init__(
        self,
        split: str = 'train',
        num_samples=None,
        random_seed=None,
        expression_length=None,
        expression_max_length=None,
        allow_division=None,
    ):
        super().__init__(
            split=split,
            num_samples=num_samples,
            random_seed=random_seed,
            expression_length=expression_length,
            expression_max_length=expression_max_length,
            allow_division=allow_division,
        )

        # Initialize terms used to denote token sequence
        max_length = (
            expression_max_length
            if expression_max_length
            else mathexpression_dataset_max_seq_length
        )
        terms = [Term("img" + str(i + 1)) for i in range(max_length)]

        self._img_token_sequences: Dict[int, typing.List[Term]] = dict()
        for length in range(1, max_length + 1, 2):
            self._img_token_sequences[length] = [terms[idx] for idx in range(length)]

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        item_dict = self.dataset[item]
        images = item_dict["image_sequence"]
        expression_result = item_dict["res"]
        context_dict = dict()
        img_sequence = self._img_token_sequences[len(images)]
        for idx, img_nr in enumerate(img_sequence):
            context_dict[img_nr] = self.tensors[images[idx]]
        return ContextualizedTerm(
            context=Context(context_dict),
            term=Term("expression", expression_result, term.List(*img_sequence)),
        )

def create_our_splits():
    import os
    from shutil import copy2
    hwf_path = os.path.join("..", "..", "data", "raw", "hwf")
    if not os.path.exists(os.path.join(hwf_path, "expr_val.json")):
        # Moving the original ones into an "original" folder
        os.mkdir(os.path.join(hwf_path, "original"))
        os.rename(os.path.join(hwf_path, "expr_train.json"), os.path.join(hwf_path, "original","expr_train.json"))
        os.rename(os.path.join(hwf_path, "expr_test.json"), os.path.join(hwf_path, "original","expr_test.json"))

        # Copying our splits into the data folder
        copy2(os.path.join("splits_with_valid", "expr_train.json"), hwf_path)
        copy2(os.path.join("splits_with_valid", "expr_val.json"), hwf_path)
        copy2(os.path.join("splits_with_valid", "expr_test.json"), hwf_path)
