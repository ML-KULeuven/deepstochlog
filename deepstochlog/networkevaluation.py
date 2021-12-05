from collections import defaultdict
from typing import Dict, Tuple, Iterable, List

import torch
from torch import Tensor

from deepstochlog.context import Context
from deepstochlog.network import NetworkStore


class RequiredEvaluation:
    def __init__(self, context: Context, network_name: str, input_args: Tuple):
        self.context = context
        self.network_name = network_name
        self.input_args = input_args

    def prepare(self) -> "PreparedEvaluation":
        """ Prepares the evaluation by mapping the input variables to the right tensors """
        mapped_input_args = self.context.get_all_tensor_representations(self.input_args)
        return PreparedEvaluation(self, mapped_input_args)

    def __str__(self):
        return (
            "RequiredEvaluation(<context>, "
            + self.network_name
            + ", "
            + str(self.input_args)
            + ")"
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, RequiredEvaluation):
            return (
                self.context == other.context
                and self.network_name == other.network_name
                and self.input_args == other.input_args
            )
        return False

    def __hash__(self):
        return hash((self.context, self.network_name, self.input_args))


class PreparedEvaluation:
    def __init__(
        self, required_evaluation: RequiredEvaluation, mapped_input_args: List[Tensor]
    ):
        self.required_evaluation = required_evaluation
        self.mapped_input_args = mapped_input_args

    def has_tensors(self):
        return len(self.mapped_input_args) > 0

    def __str__(self):
        return (
            "PreparedEvaluation("
            + str(self.required_evaluation)
            + ", tensor_list(len:"
            + str(len(self.mapped_input_args))
            + "))"
        )

    def __repr__(self):
        return str(self)


def extract_input_arguments(prepared_evaluations: Iterable[PreparedEvaluation]):
    return [
        torch.cat(pe.mapped_input_args, 0) if pe.has_tensors() else torch.tensor([])
        for pe in prepared_evaluations
    ]


class NetworkEvaluations:
    def __init__(self):
        self.evaluations: Dict[Context, Dict[str, Dict[Tuple, Tensor]]] = defaultdict(
            lambda: defaultdict(defaultdict)
        )

    def add_evaluation_result(
        self, context: Context, network_name: str, input_args: Tuple, output: Tensor
    ):
        self.evaluations[context][network_name][input_args] = output

    def get_evaluation_result(
        self, context: Context, network_name: str, input_args: Tuple
    ) -> Tensor:
        # print(context, network_name, input_args)
        return self.evaluations[context][network_name][input_args]

    @staticmethod
    def from_required_evaluations(
        required_evaluations: Iterable[RequiredEvaluation],
        networks: NetworkStore,
        device=None,
    ):
        """ Evaluates all networks for a list of required evaluations """
        # Group all required evaluations per network, and prepare required evaluation
        per_network: Dict[str, List[PreparedEvaluation]] = defaultdict(list)
        for req in required_evaluations:
            per_network[req.network_name].append(req.prepare())

        # Evaluate on network
        network_evaluations: NetworkEvaluations = NetworkEvaluations()
        for network_name, prepared_evaluations in per_network.items():
            # Convert to
            network = networks.get_network(network_name)

            if network.concat_tensor_input:
                all_to_evaluate: List[Tensor] = extract_input_arguments(
                    prepared_evaluations
                )

                # neural_input = torch.cat(all_to_evaluate, 0).unsqueeze(1)
                neural_input = torch.nn.utils.rnn.pad_sequence(
                    all_to_evaluate, batch_first=True
                )
                # neural_input = torch.stack(all_to_evaluate)

                if device:
                    neural_input = neural_input.to(device)
            else:
                neural_input = [pe.mapped_input_args for pe in prepared_evaluations]

            outputs = network.neural_model(neural_input)

            # Store result
            required_evaluations = [
                pe.required_evaluation for pe in prepared_evaluations
            ]
            for re, output in zip(required_evaluations, outputs):
                network_evaluations.add_evaluation_result(
                    context=re.context,
                    network_name=re.network_name,
                    input_args=re.input_args,
                    output=output,
                )

        return network_evaluations
