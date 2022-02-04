import random
from time import time
from typing import List, Dict, Iterable, Tuple, Sequence, Callable, Union

from numpy import argmax

# from sklearn.metrics import roc_curve

import numpy as np
import torch

import tempfile
import os

from deepstochlog.context import ContextualizedTerm
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.logic import LogicNode, NNLeaf, TermLeaf
from deepstochlog.term import Term

from io import StringIO
import sys

from deepstochlog.network import NetworkStore


def calculate_zipped_probabilities(
    model: DeepStochLogModel,
    possibilities=Iterable[ContextualizedTerm],
) -> Iterable[Tuple[ContextualizedTerm, float]]:
    probabilities = model.predict_sum_product(batch=possibilities)
    return zip(possibilities, probabilities)


def test_single_instance(
    model: DeepStochLogModel,
    test_term: Union[ContextualizedTerm, Iterable[ContextualizedTerm]],
    parse_is_nnleaf_outputs=False,
    generation_output_accuracy=True,
    create_parse: Callable[[Term, Iterable[LogicNode], NetworkStore], str] = None,
):

    if isinstance(test_term, Iterable):
        for t in test_term:
            test_single_instance(
                model,
                t,
                parse_is_nnleaf_outputs=parse_is_nnleaf_outputs,
                generation_output_accuracy=generation_output_accuracy,
                create_parse=create_parse,
            )
    else:
        masked_term = test_term.term.mask_generation_output()
        print("-" * 20)
        print("Query:\t", masked_term)
        if test_term.meta is not None:
            print("Meta:\t", test_term.meta)

        if generation_output_accuracy:
            print("Goal:\t", test_term.term.get_generation_output())

            with torch.no_grad():
                contextualized_possibilities = (
                    model.get_direct_contextualized_proof_possibilities(test_term)
                )

                predictions = [
                    (pred.term.get_generation_output(), prob)
                    for pred, prob in calculate_zipped_probabilities(
                        model, contextualized_possibilities
                    )
                ]

                predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
                for k in predictions[:5]:
                    print("{}\t{}".format(*k))

        # Predict most likely parse
        max_prod_prediction: Tuple[
            torch.Tensor, Iterable[LogicNode]
        ] = model.predict_max_product_parse(batch=[test_term.mask_generation_output()])[
            0
        ]

        parse_prob, most_likely_parse_raw = max_prod_prediction
        most_likely_parse_raw = list(most_likely_parse_raw)
        if create_parse:
            most_likely_parse = create_parse(
                test_term.term, most_likely_parse_raw, model.neural_networks
            )
        else:
            most_likely_parse = convert_most_likely_parse(
                model,
                test_term.term,
                most_likely_parse_raw,
                parse_is_nnleaf_outputs=parse_is_nnleaf_outputs,
            )
        most_likely_parse_label = convert_most_likely_parse_term(
            test_term.term.mask_generation_output(), most_likely_parse_raw
        )

        print("\nMost likely parse:\t", most_likely_parse_label)
        print("Probability:\t\t", parse_prob.item())
        print("Predicted Sequence:\t", most_likely_parse)

        print("-" * 20)
        print()


def calculate_correct_most_likely_parse(
    model: DeepStochLogModel,
    batch: List[ContextualizedTerm],
    create_parse: Callable[[Term, Iterable[LogicNode], NetworkStore], str] = None,
):
    most_likely_parses_raw: List[Iterable[LogicNode]] = list(
        el[1] for el in model.predict_max_product_parse(batch=batch)
    )
    if create_parse:
        most_likely_parses = [
            create_parse(
                batch[i].term, most_likely_parses_raw[i], model.neural_networks
            )
            for i in range(len(batch))
        ]

    else:
        most_likely_parses = [
            convert_most_likely_parse(model, batch[i].term, most_likely_parses_raw[i])
            for i in range(len(batch))
        ]
    return len(
        [
            i
            for i in range(len(batch))
            if batch[i].probability == 1 and batch[i].meta == most_likely_parses[i]
        ]
    )


def convert_most_likely_parse(
    model: DeepStochLogModel,
    term: Term,
    most_likely_parse_raw: Iterable[LogicNode],
    parse_is_nnleaf_outputs=False,
):
    nnleafs: Iterable[NNLeaf] = [
        el for el in most_likely_parse_raw if isinstance(el, NNLeaf)
    ]
    if parse_is_nnleaf_outputs:
        result = " | ".join(
            nnleaf.network
            + "->"
            + str(
                model.neural_networks.get_network(nnleaf.network).idx2term(nnleaf.index)
            )
            for nnleaf in nnleafs
        )
        return result

    else:
        mapping = dict()
        for nnleaf in nnleafs:
            token = nnleaf.inputs
            if len(token) == 1:
                token = token[0]
            network = model.neural_networks.get_network(nnleaf.network)
            mapping[token] = network.idx2term(nnleaf.index)

        token_sequence = term.arguments[-1]
        resulting_tokens = "".join([str(mapping[t]) for t in token_sequence])

        return resulting_tokens


def convert_most_likely_parse_term(
    masked_term: Term, most_likely_parse_raw: Iterable[LogicNode]
) -> Term:
    terms: Iterable[Term] = [
        el.term for el in most_likely_parse_raw if isinstance(el, TermLeaf)
    ]
    return next(el for el in terms if masked_term.covers(el) and not el.contains_mask())


def calculate_accuracy(
    model: DeepStochLogModel,
    test_data: DataLoader,
    generation_output_accuracy=True,
    most_probable_parse_accuracy=False,
    create_parse: Callable[[Term, Iterable[LogicNode], NetworkStore], str] = None,
) -> Tuple[float, float, float]:
    if len(test_data) == 0:
        return 0, 0, 0

    test_acc = 0
    label_probability = 0
    num_instances = 0
    if generation_output_accuracy:
        with torch.no_grad():
            for batch in test_data:
                num_instances += len(batch)

                other_possibilities: Dict[
                    ContextualizedTerm, List[ContextualizedTerm]
                ] = dict()
                for elem in batch:
                    other_possibilities[
                        elem
                    ] = model.get_direct_contextualized_proof_possibilities(elem)

                # flatmap
                all_to_evaluate = [
                    elem for poss in other_possibilities.values() for elem in poss
                ]

                # Map all results to queryable dict
                probabilities = dict()
                for possibility, prob in calculate_zipped_probabilities(
                    model, all_to_evaluate
                ):
                    probabilities[possibility] = prob

                # Now calculate for every possibility if its the top scoring one
                for elem in batch:

                    # Sum the probability of the labels
                    own_prob = probabilities[elem]
                    label_probability += own_prob

                    # Check if it is the highest prediction out of all other possibilities
                    is_highest = not any(
                        [
                            pos
                            for pos in other_possibilities[elem]
                            if probabilities[pos] > own_prob
                        ]
                    )
                    if is_highest:
                        test_acc += 1

    parse_acc = 0
    valid_parses = 0
    if most_probable_parse_accuracy:
        for batch in test_data:
            parse_acc += calculate_correct_most_likely_parse(
                model, batch, create_parse=create_parse
            )
            valid_parses += len([el for el in batch if el.probability == 1])

    return (
        0 if num_instances == 0 else (test_acc / num_instances),
        0 if num_instances == 0 else (label_probability / num_instances),
        0 if valid_parses == 0 else (parse_acc / valid_parses),
    )


def set_fixed_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def create_model_accuracy_calculator(
    model: DeepStochLogModel,
    test_dataloader: DataLoader,
    start_time,
    val_dataloader: DataLoader = None,
    generation_output_accuracy=True,
    most_probable_parse_accuracy=False,
    create_parse: Callable[[Term, Iterable[LogicNode], NetworkStore], str] = None,
) -> Tuple[str, Callable]:
    header, result_f = create_header(
        val_accuracy=val_dataloader is not None,
        generation_output_accuracy=generation_output_accuracy,
        most_probable_parse_accuracy=most_probable_parse_accuracy,
        average_right_prob=True,
        avg_prob=False,
        show_threshold=False,
    )

    def calculate_model_accuracy() -> str:
        val_acc = None
        val_average_right_prob = None
        val_parse_acc = None

        if val_dataloader is not None:
            val_acc, val_average_right_prob, val_parse_acc = calculate_accuracy(
                model,
                val_dataloader,
                generation_output_accuracy=generation_output_accuracy,
                most_probable_parse_accuracy=most_probable_parse_accuracy,
                create_parse=create_parse,
            )

        test_acc, test_average_right_prob, test_parse_acc = calculate_accuracy(
            model,
            test_dataloader,
            most_probable_parse_accuracy=most_probable_parse_accuracy,
            create_parse=create_parse,
        )
        return format_results(
            format_string=result_f,
            val_accuracy=val_acc,
            val_parse_acc=val_parse_acc,
            val_average_right_prob=val_average_right_prob,
            test_accuracy=test_acc,
            test_parse_acc=test_parse_acc,
            test_average_right_prob=test_average_right_prob,
            threshold=False,
            avg_prob=False,
            start_time=start_time,
        )

    return header, calculate_model_accuracy


def create_header(
    val_accuracy=False,
    generation_output_accuracy=True,
    most_probable_parse_accuracy=False,
    average_right_prob=False,
    show_threshold=None,
    avg_prob=False,
):
    header = ""
    result_f = ""
    if val_accuracy:
        if generation_output_accuracy:
            header += "Val acc\t"
            result_f += "{val_acc:.3f}\t"
            if average_right_prob:
                header += "Val P(cor)\t"
                result_f += "{val_average_right_prob:.7f}\t"
        if most_probable_parse_accuracy:
            header += "Val parse acc\t"
            result_f += "{val_parse_acc:.3f}\t\t\t"

    if generation_output_accuracy:
        header += "Test acc\t"
        result_f += "{test_acc:.3f}\t\t"
        if average_right_prob:
            header += "Test P(cor)\t"
            result_f += "{test_average_right_prob:.7f}\t"

    if most_probable_parse_accuracy:
        header += "Test parse acc\t"
        result_f += "{test_parse_acc:.3f}\t\t\t"

    if show_threshold:
        header += "Threshold\t"
        result_f += "{threshold:.3f}\t\t"
    if avg_prob:
        header += "Avg P\t"
        result_f += "{avg_prob:.3f}\t"
    header += "time"
    result_f += "{t:.3f}"
    return header, result_f


def format_results(
    format_string,
    val_accuracy: float,
    val_parse_acc,
    val_average_right_prob: float,
    test_accuracy: float,
    test_parse_acc,
    test_average_right_prob: float,
    threshold,
    avg_prob,
    start_time,
):
    result_dict = {
        "val_acc": val_accuracy,
        "val_parse_acc": val_parse_acc,
        "val_average_right_prob": val_average_right_prob,
        "test_acc": test_accuracy,
        "test_parse_acc": test_parse_acc,
        "test_average_right_prob": test_average_right_prob,
        "avg_prob": avg_prob,
        "threshold": threshold,
        "t": time() - start_time,
    }
    return format_string.format(**result_dict)


def create_run_test_query(
    model: DeepStochLogModel,
    test_data: Sequence,
    test_example_idx: Union[int, Iterable[int]] = None,
    verbose: bool = False,
    parse_is_nnleaf_outputs=False,
    generation_output_accuracy=True,
    create_parse: Callable[[Term, Iterable[LogicNode], NetworkStore], str] = None,
):
    def run_test_query():
        if len(test_data) > 0 and test_example_idx is not None and verbose:
            if isinstance(test_example_idx, Iterable):
                test_data_point = [test_data[idx] for idx in test_example_idx]
            else:
                test_data_point = test_data[test_example_idx]
            test_single_instance(
                model,
                test_data_point,
                parse_is_nnleaf_outputs=parse_is_nnleaf_outputs,
                generation_output_accuracy=generation_output_accuracy,
                create_parse=create_parse,
            )

    return run_test_query


def run_prolog(file_content):
    file_content = file_content.encode()
    with tempfile.NamedTemporaryFile() as fo:
        fo.write(file_content)
        fo.flush()
        with tempfile.NamedTemporaryFile() as log:
            cmd = f"swipl -q -f {fo.name} -t main > {log.name}"
            os.system(cmd)
            log.flush()
            res = [line.decode("UTF-8") for line in log]
    return res


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# ======================================================================================
# DEPRECATED CODE BELOW
# From when using Terms with probabilities.
# We can revive this if we want to use DeepStochLog as a feature calculator
# ======================================================================================

#
# def calculate_probability_predictions(model: DeepStochLogModel, dataloader: DataLoader):
#     with torch.no_grad():
#         all_expected_targets = torch.tensor([], device=model.device)
#         all_predicted_probabilities = torch.tensor([], device=model.device)
#
#         for batch in dataloader:
#             # Save expected
#             expected_targets = torch.as_tensor(
#                 [b.probability for b in batch], device=model.device
#             )
#             all_expected_targets = torch.cat(
#                 [all_expected_targets, expected_targets], 0
#             )
#
#             # Save prediction
#             predicted_probabilities = model.predict_sum_product(batch=batch).squeeze(-1)
#             all_predicted_probabilities = torch.cat(
#                 [all_predicted_probabilities, predicted_probabilities], 0
#             )
#     return all_expected_targets, all_predicted_probabilities
#
#
# def calculate_probability_threshold(model, validation_data):
#     (
#         all_expected_targets,
#         all_predicted_probabilities,
#     ) = calculate_probability_predictions(model, validation_data)
#     """ From https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/ """
#     # calculate roc curves
#     fpr, tpr, thresholds = roc_curve(
#         all_expected_targets.cpu().numpy(), all_predicted_probabilities.cpu().numpy()
#     )
#     # get the best threshold
#     difference = tpr - fpr
#     idx = argmax(difference)
#     best_thresh = thresholds[idx]
#     return best_thresh
#
#
# def create_run_test_query_probability(
#     model: DeepStochLogModel,
#     test_data: Sequence,
#     test_example_idx: Union[Iterable[int], int] = None,
#     verbose: bool = False,
#     parse_is_nnleaf_outputs=False,
# ):
#     def run_test_query():
#         if len(test_data) > 0 and test_example_idx is not None and verbose:
#             if isinstance(test_example_idx, Iterable):
#                 test_data_point = [test_data[idx] for idx in test_example_idx]
#             else:
#                 test_data_point = test_data[test_example_idx]
#             test_single_instance_probability(
#                 model, test_data_point, parse_is_nnleaf_outputs=parse_is_nnleaf_outputs
#             )
#
#     return run_test_query
#
#
# def test_single_instance_probability(
#     model: DeepStochLogModel,
#     test_term: Union[List[ContextualizedTerm], ContextualizedTerm],
#     parse_is_nnleaf_outputs=False,
# ):
#     if isinstance(test_term, Iterable):
#         for t in test_term:
#             test_single_instance_probability(model, t)
#     else:
#         print("-" * 20)
#         print("Query:\t", test_term.term)
#         if test_term.meta is not None:
#             print("Meta:\t", test_term.meta)
#         print("Goal:\t", test_term.probability)
#
#         with torch.no_grad():
#             prediction = model.predict_sum_product(batch=[test_term])
#             print("Total predicted probability: \t", prediction.item())
#
#             parse_prob, most_likely_parse_raw = model.predict_max_product_parse(
#                 batch=[test_term]
#             )[0]
#             most_likely_parse = convert_most_likely_parse(
#                 model,
#                 test_term.term,
#                 most_likely_parse_raw,
#                 parse_is_nnleaf_outputs=parse_is_nnleaf_outputs,
#             )
#             print("\nMost likely parse:")
#             print("Probability:\t\t", parse_prob.item())
#             print("Predicted Sequence:\t", most_likely_parse)
#         print("-" * 20)
#         print()
#
#
# def create_probability_accuracy_calculator(
#     model: DeepStochLogModel,
#     test_dataloader: DataLoader,
#     start_time,
#     validation_data: DataLoader = None,
#     threshold: float = None,
#     most_probable_parse_accuracy=False,
# ) -> Tuple[str, Callable]:
#
#     header, result_f = create_header(
#         most_probable_parse_accuracy=most_probable_parse_accuracy,
#         average_right_prob=False,
#         show_threshold=threshold is None,
#         avg_prob=True,
#     )
#
#     def calculate_model_accuracy() -> str:
#         if threshold is None:
#             if validation_data is not None:
#                 actual_threshold = calculate_probability_threshold(
#                     model, validation_data
#                 )
#             else:
#                 raise RuntimeError(
#                     "Please provide a threshold or a validation dataset to calculate the threshold with"
#                 )
#         else:
#             actual_threshold = threshold
#
#         acc, avg_prob, parse_acc = calculate_probability_accuracy(
#             model,
#             test_dataloader,
#             threshold=actual_threshold,
#             most_probable_parse_accuracy=most_probable_parse_accuracy,
#         )
#         return format_results(
#             format_string=result_f,
#             test_accuracy=acc,
#             threshold=actual_threshold,
#             avg_prob=avg_prob,
#             test_parse_acc=parse_acc,
#             start_time=start_time,
#             test_average_right_prob=0,
#         )
#
#     return header, calculate_model_accuracy
#
#
# def calculate_probability_accuracy(
#     model: DeepStochLogModel,
#     test_data: DataLoader,
#     threshold: float = 0.5,
#     most_probable_parse_accuracy=False,
# ) -> Tuple[float, float, float]:
#     (
#         all_expected_targets,
#         all_predicted_probabilities,
#     ) = calculate_probability_predictions(model, test_data)
#
#     threshold_tensor = Variable(torch.as_tensor([threshold], device=model.device))
#     num_instances = all_predicted_probabilities.size()[0]
#     total_probability = torch.sum(all_predicted_probabilities).item()
#     all_predictions = (all_predicted_probabilities > threshold_tensor).float() * 1
#     accurate_predictions = torch.sum((all_predictions == all_expected_targets)).item()
#
#     parse_acc = 0
#     valid_parses = 0
#     if most_probable_parse_accuracy:
#         for batch in test_data:
#             parse_acc += calculate_correct_most_likely_parse(model, batch)
#             valid_parses += len([el for el in batch if el.probability == 1])
#
#     if num_instances == 0:
#         return 0, 0, 0
#
#     return (
#         accurate_predictions / num_instances,
#         total_probability / num_instances,
#         0 if parse_acc == 0 else (parse_acc / valid_parses),
#     )
