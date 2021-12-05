import re
from typing import List

from pyparsing import (
    Word,
    alphas,
    nums,
    delimitedList,
    Suppress,
    Combine,
    Optional,
    OneOrMore,
    ParseResults,
    Group,
    Regex,
    Or,
    Forward,
    ParseException,
)

from deepstochlog.rule import (
    TrainableProbabilityAnnotation,
    VariableProbabilityAnnotation,
    StaticProbabilityAnnotation,
    NeuralProbabilityAnnotation,
    NDCGRule,
    ProbabilityAnnotation,
    Rule,
    Fact,
    ClauseRule,
    ProgramRules,
)
from deepstochlog.term import Term


def create_rules_parser():
    static_probability = (
        Combine(Optional("0") + "." + Word(nums))
        | Combine("1" + Optional("." + OneOrMore("0")))
    ).setResultsName("static_probability")

    variable_probability = Word(alphas.upper()).setResultsName("variable_probability")

    trainable_probability = Combine("t(_)").setResultsName("trainable_probability")

    any_alphanumeric = alphas + nums + "_"
    any_alphanumeric_word = Word(any_alphanumeric)
    variable = Word(alphas.upper() + "_", alphas + nums + "_")
    model_name = any_alphanumeric_word
    input_vars = (
        Suppress("[") + Group(Optional(delimitedList(Group(variable)))) + Suppress("]")
    )
    neural_probability = (
        Suppress("nn(")
        + model_name.setResultsName("model_name")
        + Suppress(",")
        + input_vars.setResultsName("input_vars")
        + Suppress(",")
        + Group(variable).setResultsName("output_var")
        + Suppress(",")
        + any_alphanumeric_word.setResultsName("output_domain")
        # + Suppress("[")
        # + Group(delimitedList(any_alphanumeric_word)).setResultsName("output_domain")
        # + Suppress("]")
        + Suppress(")")
    ).setResultsName("neural_probability")

    any_probability = static_probability | variable_probability | trainable_probability | neural_probability

    term_forward = Forward()
    term = any_alphanumeric_word + Optional(
        Suppress("(")
        + Group(delimitedList(Group(term_forward)).setResultsName("arguments"))
        + Suppress(")")
    )
    term_forward << term

    # clause = (
    #     term
    #     ^ (term + ";" + term)
    #     ^ (term + "," + term)
    #     ^ (
    #         term
    #         + Or(["=", "is"])
    #         + term
    #         + Or(["+", "-", "*", "/", "**", "%"])
    #         + term
    #     )
    #     ^ (term + Or(["<", ">", "=<", ">="]) + term)
    #     ^ ("{" + term + "}")
    # )
    # body = clause

    # Disallow --> and :- to occur in body, as this usually indicates a bug, e.g. forgotten period
    # Allow quoted dots, dots followed by a digit, and otherwise any character that isn't a dot
    body = Regex(
        r"((?!([-][-][>])|([:][-]))(([^\\][\"]([^\"])*[^\\][\"])|(.\d)|[^.]))*"
    )

    ndcg_rule = (
        Optional(
            Group(any_probability.setResultsName("probability")) + Suppress("::")
        ).setResultsName("probability_annotation")
        + Group(term.setResultsName("head"))
        + Suppress("-->")
        + Group(body.setResultsName("body"))
        + Suppress(".")
    ).setResultsName("ndcg_rule")

    fact = (Group(term.setResultsName("head")) + Suppress(".")).setResultsName("fact")
    clause_rule = (
        Group(term.setResultsName("head"))
        + Suppress(":-")
        + Group(body.setResultsName("body"))
        + Suppress(".")
    ).setResultsName("clause_rule")

    rule = ndcg_rule | clause_rule | fact

    rules = OneOrMore(Group(rule)).setResultsName("rules")
    return rules


rules_parser = create_rules_parser()


def parse_term(input_term: ParseResults) -> Term:
    functor = input_term[0]
    arguments = [] if len(input_term) == 1 else [parse_term(t) for t in input_term[1]]
    return Term(functor, *arguments)


def parse_probability(p_probability) -> ProbabilityAnnotation:
    if len(p_probability) == 1:
        el = p_probability[0]
        if str(el).startswith("t"):
            return TrainableProbabilityAnnotation()
        if str(el)[0].isupper():
            return VariableProbabilityAnnotation(el)
        else:
            return StaticProbabilityAnnotation(float(el))
    else:
        model, p_input_vars, output_var, p_output_domain = p_probability
        input_vars = [parse_term(t) for t in p_input_vars]
        # output_domain = [parse_term(t) for t in p_output_domain]
        return NeuralProbabilityAnnotation(
            str(model), input_vars, parse_term(output_var), p_output_domain
        )


def process_body(p_body: ParseResults):
    """ Joins the elements of the body together, and removes unnecessary extra whitespace"""
    return re.sub(r"\s+", " ", "".join([str(el) for el in p_body]))


def parse_rule(input_rule: ParseResults) -> Rule:
    # print("rule", input_rule)
    if "fact" in input_rule:
        (p_head,) = input_rule
        return Fact(parse_term(p_head))
    if "clause_rule" in input_rule:
        p_head, p_body = input_rule
        return ClauseRule(parse_term(p_head), process_body(p_body))
    if "ndcg_rule" in input_rule:
        if "probability_annotation" in input_rule:
            p_probability, p_head, p_body = input_rule
            probability = parse_probability(p_probability)
        else:
            # If no probability given, make it 1
            p_head, p_body = input_rule
            probability = StaticProbabilityAnnotation(1)

        return NDCGRule(
            probability=probability,
            head=parse_term(p_head),
            body=process_body(p_body),
        )

    raise NotImplementedError("Unsupported rule type: " + str(input_rule))


def parse_rules(rules_str: str) -> ProgramRules:
    try:
        parsed_rules = rules_parser.parseString(rules_str, parseAll=True)
    except ParseException as error:
        # Add common mistake warning
        if 'Expected {{[{Group:({{Combine:({["0"]' in error.msg:
            error.msg += (
                "\nDid you forget putting a closing period at the end of a rule?"
                "Check all lines above the given line number"
            )
        raise error

    resulting_rules = []

    for parsed_rule in parsed_rules:
        resulting_rules.append(parse_rule(parsed_rule))

    print("Parsed rules:", "\n".join([str(r) for r in resulting_rules]), "\n", sep="\n")

    return ProgramRules(resulting_rules)
