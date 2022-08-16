import atexit
import sys
import tempfile
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple, Dict, Callable
import typing
import ast

from deepstochlog.network import NetworkStore
from deepstochlog.tabled_and_or_trees import TabledAndOrTrees
from deepstochlog import term
from deepstochlog.term import Term, List
from deepstochlog.logic import (
    And,
    Or,
    NNLeaf,
    LogicNode,
    TermLeaf,
    AlwaysTrue,
    AlwaysFalse,
    StaticProbability,
)

# Declare some common parts for querying prolog later
from deepstochlog.rule import (
    Rule,
    ProgramRules,
    NDCGRule,
    StaticProbabilityAnnotation,
    VariableProbabilityAnnotation,
    NeuralProbabilityAnnotation,
)
from deepstochlog.inferences import DescendantTermMapper

file_root = Path(__file__).parent
with open(file_root / "_meta_program_suffix.pl", "r") as file:
    meta_program = file.read()


dcg_converter = """
translate_and_write(DCGClause) :- expand_term(DCGClause,[_,PrologClause]),
                                  numbervars(PrologClause, 0, _), 
                                  PrologClause = :-(Head,Body),
                                  comma_list(Body, LBody),
                                  flatten(LBody,FBody),
                                  comma_list(CBody,FBody),
                                  Rule = :-(Head,CBody),
                                  write_term(Rule,[numbervars(true), quoted(true)]), 
                                  nl.
                                  
main :- findall(D, to_expand(D), L), maplist(translate_and_write, L).
"""


def convert_ndcg_to_prolog(rule: NDCGRule):
    if isinstance(rule.probability, StaticProbabilityAnnotation):
        probability_representation = "p(" + str(rule.probability.probability) + ")"
    elif isinstance(rule.probability, VariableProbabilityAnnotation):
        probability_representation = "p(" + str(rule.probability.variable) + ")"
    elif isinstance(rule.probability, NeuralProbabilityAnnotation):
        probability_representation = (
            f"{rule.probability.output_domain}({rule.probability.output_var}), "
            f"nn({rule.probability.model_name},{rule.probability.input_var},{rule.probability.output_var})"
        )
    else:
        raise NotImplementedError(
            "Can't deal with this type of probability:", rule.probability
        )

    # The probability representation is used first so that all the variables in the body are unified.
    # This is useful for structure learning, where the output variables of neural predicates are non-terminals
    # that need to be unified first.
    return (
        str(rule.head) + " --> " + "{" + probability_representation + "}, " + rule.body
    )


class PrologSolver:
    def __init__(self, include_predicates=()):

        self.meta_program = meta_program
        self.dcg_converter = dcg_converter
        self.include_predicates = include_predicates
        self.inclusion_definition = (
            "\n".join(["include_predicate(%s)." % p for p in include_predicates]) + "\n"
        )

    def _run_prolog(self, program):
        # file_content = program.encode()
        fo_path = _temporary_filename(suffix="pl", text=True)
        # with tempfile.NamedTemporaryFile() as fo:
        with open(fo_path, "w") as fo:
            fo.write(program)
            fo.flush()
            log_path = _temporary_filename(suffix="pl", text=True)
            with open(log_path, "r") as log:
                cmd = f"swipl -q -f {fo.name} -t main > {log.name}"
                os.system(cmd)
                log.flush()
                res = [line.replace("\n", "") for line in log.readlines()]

        return res

    def convert_dcg(self, clauses):

        dcg_clauses = ["to_expand((%s))." % s.strip() for s in clauses if "-->" in s]
        prolog_clauses = [s for s in clauses if "-->" not in s and s.strip() != ""]
        program = "\n".join(dcg_clauses)

        dcg_converted_clauses = self._run_prolog(program + self.dcg_converter)
        converted_program = ".\n".join(prolog_clauses + dcg_converted_clauses) + ".\n"
        return converted_program

    def prove(self, program, query):

        program = (
            self.meta_program.format(query=query) + self.inclusion_definition + program
        )

        res = self._run_prolog(program)

        return res


def _create_deepstochlog_definition(domains, rules):
    """Define a number of utility predicates for DeepStochLog."""

    # Multiple ways of defining domains
    definition = "domain(X, L) :- member(X, L). \n"  # TODO(giuseppe): Likely depreated
    if domains is not None:
        for k, l in domains.items():
            definition += "{}(X) :- member(X, [{}]). \n".format(k, ",".join(l))

    # Neural Calls
    definition = definition + "nn(_,_,_,_).\n"
    definition = definition + "nn(_,_,_).\n"

    # Static probabilities
    definition = definition + "p(_).\n"

    # Trainable probabilities
    definition = definition + "t(_).\n"

    return definition


class TabledAndOrTreeBuilder:
    def __init__(
        self,
        program_rules: ProgramRules,
        prolog_executor: str = "swipl --table_space=1000000000000",
        verbose: bool = False,
        domains: Dict[str, typing.List[str]] = None,
        simplify_tree=True,
        prolog_facts: str = "",
        tabling=True,
    ):
        self.verbose = verbose
        self.prolog_executor = prolog_executor

        # Add prolog facts from program
        prolog_facts += "\n" + "\n".join(
            [str(rule) for rule in program_rules.get_prolog_rules()]
        )

        # Predicates that should be included in the and/or tree
        include_predicates = ["nn", "p", "t"]
        for rule in program_rules.get_ndcg_rules():
            include_predicates.append(rule.head.functor)

        # Engine
        self.solver = PrologSolver(include_predicates=include_predicates)

        self.prolog_program = self._refactor_to_difference_list_dcg(
            program_rules, tabling=tabling, domains=domains, prolog_facts=prolog_facts
        )
        self.simplify_tree = simplify_tree

        if self.verbose:
            print("----------Final Prolog Program-----------")
            print(self.prolog_program.strip())
            print("----------------------------------\n")

    def _refactor_to_difference_list_dcg(
        self,
        rules: ProgramRules,
        tabling: bool = True,
        domains: Dict[str, typing.List[str]] = None,
        prolog_facts: str = "",
    ) -> str:

        # Extract the DCG from the string
        dcg_clauses = [convert_ndcg_to_prolog(rule) for rule in rules.get_ndcg_rules()]

        transformed_program = self.solver.convert_dcg(dcg_clauses)

        # Add tabling header
        transformed_program = (
            _create_deepstochlog_definition(domains, rules.get_ndcg_rules())
            + "\n"
            + prolog_facts
            + "\n"
            + transformed_program
        )
        return transformed_program

    def build_and_or_trees(
        self,
        networks: NetworkStore,
        queries: typing.Optional[Union[Term, typing.Iterable]] = None,
        add_default_dynamic_grounding=True,
        add_default_zero_probability=False,
    ) -> TabledAndOrTrees:

        if add_default_dynamic_grounding and add_default_zero_probability:
            raise RuntimeError(
                "Can't allow both default dynamic grounding and default zero probability"
            )

        if isinstance(queries, Term):
            queries = [queries]

        def ground_all(input_terms: typing.Iterable[Term]):
            # If there are multiple queries, add comma's in between
            queries = ",".join([str(q.to_dsl_input()) for q in input_terms])
            prolog_result = self.solver.prove(self.prolog_program, queries)

            if not prolog_result:
                raise RuntimeWarning(
                    "There was a problem when proving the terms, as the Prolog output is empty.\n"
                    "Please check if your DeepStochLog Prolog program is valid, "
                    "and if these terms can be proved with it:"
                    "\nTerms:\n" + str(input_terms)
                )

            """The program writes on each line of the file a list with two elements (key, value) of the table.
            Each key value of the table is a complex Prolog term represented as a nested structures of lists.
            Each list represents:
            1) a Prolog list if its first element is "list". All the other elements are the elements of the list.
            2) a Prolog compound term if its first element is different from "list". In this case, the first element
               is the functor of the term and all the other elements its arguments.
            This is valid Python syntax and can be parsed with "eval()".  
            """

            and_or_tree_builder: Dict[Term, typing.List] = defaultdict(list)

            for k, raw_line in enumerate(prolog_result):
                line_parts = ast.literal_eval(raw_line)
                parent_raw = line_parts[0]
                conjunction_raw = line_parts[1]
                parent = _parse_term_from_lists(parent_raw).without_difference_list()
                # Parse the conjunection, and turn terms into differencelist-less representation, and neural leafs
                conjunction = _parse_logic_node(
                    conjunction_raw, networks=networks, simplify_tree=self.simplify_tree
                ).accept_visitor(DescendantTermMapper(lambda t: t.without_difference_list()))

                and_or_tree_builder[parent].append(conjunction)

            if add_default_zero_probability:
                and_or_tree: Dict[Term, LogicNode] = defaultdict(lambda: AlwaysFalse())
            else:
                and_or_tree: Dict[Term, LogicNode] = dict()

            for key, values in and_or_tree_builder.items():
                if self.simplify_tree and len(values) == 1:
                    and_or_tree[key] = values[0]
                else:
                    and_or_tree[key] = Or(*values)

            return and_or_tree

        ground_tree = ground_all(queries) if queries is not None else dict()
        # Allow for grounding unseen terms later
        if add_default_dynamic_grounding:
            return TabledAndOrTrees(ground_tree, ground_all)

        return TabledAndOrTrees(ground_tree)

    def _prolog_only(
        self,
        queries: typing.Optional[Union[Term, typing.Iterable]] = None,
        add_default_dynamic_grounding=True,
        add_default_zero_probability=False,
    ):

        if add_default_dynamic_grounding and add_default_zero_probability:
            raise RuntimeError(
                "Can't allow both default dynamic grounding and default zero probability"
            )

        if isinstance(queries, Term):
            queries = [queries]
            queries = ",".join([str(q.to_dsl_input()) for q in queries])
            prolog_result = self.solver.prove(self.prolog_program, queries)
            return prolog_result


class RawNNLeaf(TermLeaf):
    def __init__(self, nn_term: Term, networks: NetworkStore):
        # TODO: Remove this soon! This is just a fix to still support old syntax with single input argument.
        nn_term = Term(
            nn_term.functor,
            nn_term.arguments[0],
            nn_term.arguments[2],
            nn_term.arguments[1],
            List(),
        )

        super().__init__(nn_term)
        # TODO Network is saved for mapping to index id. Should maybe be mapped using Prolog instead
        self.networks = networks

    def accept_visitor(self, visitor: "DescendantTermMapper") -> "LogicNode":
        """Overwrites this so it can return a NNLeaf once the difference list is removed.
        We thus assume that the mapper is removing the difference list
        Maybe not the cleanest solution though, but it prevents going over all elements again"""
        return visitor.accept_raw_nn_leaf(self)


always_true = AlwaysTrue()


def _parse_logic_node(
    term_list: typing.List, networks: NetworkStore = None, simplify_tree=True
) -> LogicNode:
    functor = term_list[0]

    if functor == "conj":
        if simplify_tree and len(term_list) == 2:
            return _parse_logic_node(term_list[1], networks=networks)
        if len(term_list) == 1:
            return always_true
        return And(*[_parse_logic_node(i, networks=networks) for i in term_list[1:]])

    # Parse term
    inner_term = Term(
        functor,
        *[_parse_term_from_lists(i, networks=networks) for i in term_list[1:]],
    )
    if functor == "nn":
        return RawNNLeaf(nn_term=inner_term, networks=networks)
    if functor == "p" and len(inner_term.arguments) == 1:
        probability = float(inner_term.arguments[0].functor)
        # print("Adding static probability",probability)
        if probability < 0 or probability > 1:
            raise RuntimeError(
                "Given static probability is not a valid probability", term_list
            )
        return StaticProbability(probability)
    return TermLeaf(term=inner_term)


def _parse_term_from_lists(
    term_list: typing.List, networks: NetworkStore = None
) -> Term:
    """Term is a list where the first element is either a valid functor or the keyword 'list'.
    If the first element is list, then this term is a list, we parse recursively all its elements and return them.
    If the first element is a valid functor, we parse recursively all the other elements and create a term with
    that functor and the other elements as arguments."""
    functor = term_list[0]

    remaining_arguments = [
        _parse_term_from_lists(i, networks=networks) for i in term_list[1:]
    ]

    if functor == "list":
        return term.List(*remaining_arguments)
    return Term(functor, *remaining_arguments)


def _temporary_filename(
    prefix=None, suffix="tmp", dir=None, text=False, remove_on_exit=True
):
    """Returns a temporary filename that, like mkstemp(3), will be secure in
    its creation.  The file will be closed immediately after it's created, so
    you are expected to open it afterwards to do what you wish.  The file
    will be removed on exit unless you pass removeOnExit=False.  (You'd think
    that amongst the myriad of methods in the tempfile module, there'd be
    something like this, right?  Nope.)"""

    if prefix is None:
        prefix = "%s_%d_" % (os.path.basename(sys.argv[0]), os.getpid())

    (fileHandle, path) = tempfile.mkstemp(
        prefix=prefix, suffix=suffix, dir=dir, text=text
    )
    os.close(fileHandle)

    def remove_file(path):
        os.remove(path)

    if remove_on_exit:
        atexit.register(remove_file, path)

    return path
