import json
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import typing

from deepstochlog.dataset import ContextualizedTermDataset
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.term import Term, List

data_root = Path(__file__).parent / ".." / ".." / "data" / "raw" / "wap"


class WapDataset(ContextualizedTermDataset):
    def __init__(
        self,
        split: str = "train",
        size: int = None,
    ):
        with open(data_root / "questions.json", "r") as questions_file:
            all_questions: typing.Dict = json.load(questions_file)
        with open(data_root / (split + ".txt"), "r") as split_file:
            question_answers: typing.List[typing.Tuple[int, str]] = [
                (int(float(el[0])), el[1])
                for el in [s.split("\t") for s in split_file.readlines()]
            ]

        # for i, q in enumerate(all_questions):
        #     assert i == q["iIndex"]

        with open(data_root / (split + ".tsv")) as ids_file:
            idxs = [int(idx) for idx in ids_file.readlines()]
            questions = [
                {
                    **all_questions[idx],
                    "tokenized_question": question_answers[i][1],
                }
                for i, idx in enumerate(idxs)
            ]

        if size is None:
            size = len(questions)

        self.ct_term_dataset = []
        for idx in range(0, size):
            question = questions[idx]

            example = create_term(question)
            self.ct_term_dataset.append(example)

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]


def get_number(question: str, alignment: int):
    number_str = ""
    while question[alignment].isdigit():
        number_str += question[alignment]
        alignment += 1
    return int(number_str)


def get_numbers(question: str, alignments: typing.List[int]):
    return tuple(get_number(question, alignment) for alignment in alignments)


sentence_token = List(Term("sentence_token"))


def create_term(question: typing.Dict) -> ContextualizedTerm:

    number1, number2, number3 = get_numbers(
        question["sQuestion"], question["lAlignments"]
    )

    correct_sequence = question["lEquations"][0]
    # Remove "X=(" and ")", and then replace all ".0" from numbers
    correct_sequence_fixed = correct_sequence[3:-1].replace(".0","")

    return ContextualizedTerm(
        context=Context(
            {Term("sentence_token"): question["tokenized_question"]},
            map_default_to_term=True,
        ),
        term=Term(
            "s",
            Term(str(int(question["lSolutions"][0]))),
            Term(str(number1)),
            Term(str(number2)),
            Term(str(number3)),
            sentence_token,
        ),
        meta=correct_sequence,
    )
