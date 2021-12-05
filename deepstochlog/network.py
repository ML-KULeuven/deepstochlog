from typing import List

import torch.nn as nn

from deepstochlog.term import Term


class Network(object):
    def __init__(
        self,
        name: str,
        neural_model: nn.Module,
        index_list: List[Term],
        concat_tensor_input=True,
    ):
        self.name = name
        self.neural_model = neural_model
        self.computation_graphs = dict()
        self.index_list = index_list
        self.index_mapping = dict()
        if index_list is not None:
            for i, elem in enumerate(index_list):
                self.index_mapping[elem] = i
        self.concat_tensor_input = concat_tensor_input

    def term2idx(self, term: Term) -> int:
        #TODO(giuseppe) index only with the functor

        # key = term #old
        key = Term(str(term.functor))
        if key not in self.index_mapping:
            raise Exception(
                "Index was not found, did you include the right Term list as keys? Error item: "
                + str(term)
                + " "
                + str(type(term))
                + ".\nPossible values: "
                + ", ".join([str(k) for k in self.index_mapping.keys()])
            )
        return self.index_mapping[key]

    def idx2term(self, index: int) -> Term:
        return self.index_list[index]

    def to(self, *args, **kwargs):
        self.neural_model.to(*args, **kwargs)


class NetworkStore:
    def __init__(self, *networks: Network):
        self.networks = dict()
        for n in networks:
            self.networks[n.name] = n

    def get_network(self, name: str) -> Network:
        return self.networks[name]

    def to_device(self, *args, **kwargs):
        for network in self.networks.values():
            network.to(*args, **kwargs)

    def get_all_net_parameters(self):
        all_parameters = list()
        for network in self.networks.values():
            all_parameters.extend(network.neural_model.parameters())
        return all_parameters

    def __add__(self, other: "NetworkStore"):
        return NetworkStore(
            *(list(self.networks.values()) + list(other.networks.values()))
        )
