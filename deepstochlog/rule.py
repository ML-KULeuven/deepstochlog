import typing
from collections import defaultdict
from typing import Dict

from deepstochlog.term import Term, List
from deepstochlog.network import Network, NetworkStore
from deepstochlog.nn_models import TrainableProbability


class ProbabilityAnnotation:
    pass



class VariableProbabilityAnnotation(ProbabilityAnnotation):
    def __init__(self, variable: str):
        self.variable = variable

    def __str__(self):
        return str(self.variable)

    def __repr__(self):
        return str(self)


class StaticProbabilityAnnotation(ProbabilityAnnotation):
    def __init__(self, probability: float):
        self.probability = probability

    def __str__(self):
        return str(self.probability)

    def __repr__(self):
        return str(self)


class TrainableProbabilityAnnotation(ProbabilityAnnotation):
    def __init__(self):
        pass

    def __str__(self):
        return "t(_)"

    def __repr__(self):
        return str(self)


class NeuralProbabilityAnnotation(ProbabilityAnnotation):
    def __init__(
        self,
        model_name: str,
        input_var: typing.List[Term],
        output_var: Term,
        # output_domain: List[Term],
        output_domain: str,
    ):
        self.model_name = model_name
        self.input_var = input_var
        self.output_var = output_var
        self.output_domain = output_domain

    def __str__(self):
        return (
            "nn("
            + str(self.model_name)
            + ", "
            + str(self.input_var)
            + ", "
            + str(self.output_var)
            + ", "
            + str(self.output_domain)
            + ")"
        )

    def __repr__(self):
        return str(self)


class Rule:
    def __init__(self, head: Term):
        self.head = head


class Fact(Rule):
    def __init__(self, head: Term):
        super().__init__(head)

    def __str__(self):
        return str(self.head) + "."

    def __repr__(self):
        return str(self)


class ClauseRule(Rule):
    def __init__(self, head: Term, body: str):
        super().__init__(head)
        self.body = body

    def __str__(self):
        return str(self.head) + " :- " + self.body + "."

    def __repr__(self):
        return str(self)


class NDCGRule(Rule):
    def __init__(self, probability: ProbabilityAnnotation, head: Term, body: str):
        super().__init__(head)
        self.probability = probability
        self.body = body

    def __str__(self):
        return (
            str(self.probability) + " :: " + str(self.head) + " --> " + self.body + "."
        )

    def __repr__(self):
        return str(self)


def check_trainable_probability_support(
    trainable_probability_rules: typing.List[NDCGRule], rules: typing.List[NDCGRule]
):
    if len(trainable_probability_rules) > 0:
        # Check if all rules are trainable if at least one of them is. Otherwise, not (yet) supported
        if len(trainable_probability_rules) != len(rules):
            raise RuntimeError(
                "Rules need to either have all trainable probabilities"
                "or none. Mix of probability types in rules: {}".format(
                    trainable_probability_rules
                )
            )
        # Checks if they all have the same arguments
        arguments = trainable_probability_rules[0].head.arguments
        for trainable_rule in trainable_probability_rules:
            if trainable_rule.head.arguments != arguments:
                raise RuntimeError(
                    "Trainable probability rules all need to have the same exact arguments & arguments names."
                    "Different arguments are not yet supported.\n"
                    "Conflicting rules:\n{}\n{}".format(
                        trainable_probability_rules[0], trainable_rule
                    )
                )


class ProgramRules:
    def __init__(self, rules: typing.List[Rule]):
        self.rules = rules

    # Divide over types of rules
    def get_ndcg_rules(self) -> typing.List[NDCGRule]:
        return [rule for rule in self.rules if isinstance(rule, NDCGRule)]

    def get_prolog_rules(self):
        return [
            rule
            for rule in self.rules
            if isinstance(rule, ClauseRule) or isinstance(rule, Fact)
        ]

    # Properties checking
    def has_trainable_probabilities(self):
        return any(
            rule
            for rule in self.get_ndcg_rules()
            if isinstance(rule.probability, TrainableProbabilityAnnotation)
        )

    # Transformations
    def remove_syntactic_sugar(self) -> typing.Tuple["ProgramRules", NetworkStore]:
        return self._transform_trainable_probabilities_to_switches()

    def _transform_trainable_probabilities_to_switches(
        self,
    ) -> typing.Tuple["ProgramRules", NetworkStore]:
        # Rules of the final program
        resulting_rules = []
        new_networks = []

        # Sort rules
        rules_per_head: Dict[
            typing.Tuple[str, int], typing.List[NDCGRule]
        ] = defaultdict(list)
        for rule in self.rules:
            if isinstance(rule, NDCGRule):
                functor_arity = rule.head.get_functor_and_arity()
                rules_per_head[functor_arity].append(rule)
            else:
                resulting_rules.append(rule)

        # Create switch for each functor&arity if there are trainable parameters
        for ((functor, arity), rules) in rules_per_head.items():
            trainable_probability_rules: typing.List[NDCGRule] = [
                rule
                for rule in rules
                if isinstance(rule.probability, TrainableProbabilityAnnotation)
            ]

            # Check if there are trainable parameters
            if len(trainable_probability_rules) == 0:
                resulting_rules.extend(rules)
            else:
                check_trainable_probability_support(trainable_probability_rules, rules)

                # Create new predicates/heads/names
                switch_functor_name = "switch_{}_{}".format(functor, arity)
                dom_head_functor = "domain_{}".format(switch_functor_name)
                neural_network_name = "nn_{}".format(switch_functor_name)
                number_of_switch_rules = len(trainable_probability_rules)

                # Create domain of the switch variable
                domain_variable = Term("X")
                domain_range = [Term(str(i)) for i in range(number_of_switch_rules)]
                switch_domain_rule = ClauseRule(
                    head=Term(dom_head_functor, domain_variable),
                    body=str(Term("member", domain_variable, List(*domain_range))),
                )
                resulting_rules.append(switch_domain_rule)

                # Create trainable probability
                switch_nn = Network(
                    neural_network_name,
                    TrainableProbability(N=number_of_switch_rules),
                    index_list=domain_range,
                )
                new_networks.append(switch_nn)

                # Create a rule mapping the original head to the switch, e.g.
                # nn(s_nn, [], Y, s_switch_dom) :: s --> s_switch(Y)
                switch_rule_variable = Term("Y")
                passed_arguments = trainable_probability_rules[0].head.arguments
                switch_arguments = [switch_rule_variable] + list(passed_arguments)

                switch_entry_rule = NDCGRule(
                    probability=NeuralProbabilityAnnotation(
                        model_name=neural_network_name.format(functor, arity),
                        input_var=[],
                        output_var=switch_rule_variable,
                        output_domain=dom_head_functor,
                    ),
                    head=Term(functor, *passed_arguments),
                    body=str(Term(switch_functor_name, *switch_arguments)),
                )
                switch_choice_rules = [
                    NDCGRule(
                        probability=StaticProbabilityAnnotation(1),
                        head=Term(
                            switch_functor_name,
                            *([switch_rule_idx] + list(passed_arguments)),
                        ),
                        body=trainable_probability_rules[i].body,
                    )
                    for i, switch_rule_idx in enumerate(domain_range)
                ]

                resulting_rules.append(switch_entry_rule)
                resulting_rules.extend(switch_choice_rules)

        return ProgramRules(resulting_rules), NetworkStore(*new_networks)

    # Buildins
    def __str__(self):
        return "\n".join([str(rule) for rule in self.rules])

    def __repr__(self):
        return str(self)
