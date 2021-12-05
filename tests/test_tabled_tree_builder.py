from collections import defaultdict
from pathlib import Path
from typing import Union

import typing

import torch
from torch import Tensor
import time

from deepstochlog.context import Context, ContextualizedTerm
from examples.mathexpression.mathexpression import (
    load_expression_networks,
    create_expression_sentence_query,
)
from examples.addition.addition_data import AdditionDataset
from examples.models import MNISTNet
from deepstochlog.model import DeepStochLogModel
from deepstochlog.logic import And, Or, NNLeaf, TermLeaf, StaticProbability
from deepstochlog.network import Network, NetworkStore
from deepstochlog.tabled_tree_builder import TabledAndOrTreeBuilder
from deepstochlog.term import Term, List
import unittest

from deepstochlog.parser import parse_rules

root = Path(__file__).parent
with open(
    root / ".." / "examples" / "mathexpression" / "mathexpression.pl", "r"
) as file:
    expression_program = file.read()

with open(root / ".." / "examples" / "addition" / "addition.pl", "r") as file:
    addition_program = file.read()


def create_mock_expression_model(query):
    # Read expression grammar
    # Create neural networks
    networks = load_expression_networks()

    # Create model
    model: DeepStochLogModel = DeepStochLogModel.from_string(
        program_str=expression_program, networks=networks, query=query
    )
    return model


def create_mock_expression_context():
    mock_context = Context(defaultdict(lambda: torch.ones([1, 45, 45])))
    return mock_context


mnist_network = MNISTNet()


def create_addition_program(
    query: Union[Term, typing.List[Term]] = Term(
        "addition", Term("_"), List(Term("t1"), Term("t2"))
    )
):
    builder = TabledAndOrTreeBuilder(parse_rules(addition_program), verbose=True)
    fake_network = Network(
        "number",
        neural_model=mnist_network,
        index_list=[Term(str(i)) for i in range(10)],
    )
    networks = NetworkStore(fake_network)
    tabled_program = builder.build_and_or_trees(networks=networks, queries=query)
    return tabled_program



def create_mock_addition_model_with_loss_probability(
    query: Union[Term, typing.List[Term]] = Term(
        "addition", Term("_"), List(Term("t1"), Term("t2")),
    ),
    normalization=None

):
    addition_odd_program = """
digit(Y) :- member(Y,[0,1,2,3,4,5,6,7,8,9]).
nn(number, [X], Y, digit) :: is_number(Y) --> [X].
addition(N) --> is_number(N1),
                is_number(N2),
                {N is N1 + N2, N < 15}.
"""
    fake_network = Network(
        "number",
        neural_model=mnist_network,
        index_list=[Term(str(i)) for i in range(10)],
    )
    networks = NetworkStore(fake_network)

    # Create model
    model: DeepStochLogModel = DeepStochLogModel.from_string(
        program_str=addition_odd_program, networks=networks, query=query, normalization=normalization
    )
    return model


class AndOrTreeTest(unittest.TestCase):
    def test_addition(self):
        """ FIRST WE DEFINE THE PROGRAM"""
        tabled_program = create_addition_program()
        and_or_tree = tabled_program._and_or_tree

        for k, v in and_or_tree.items():
            print(k, v)

        self.assertTrue(len(and_or_tree) > 0)

        # Check top result
        top = Term("addition", Term("1"), List(Term("t1"), Term("t2")))
        print("Top", top)
        for key in and_or_tree:
            if key.functor == "addition":
                print("key", key)

        self.assertTrue(top in and_or_tree)
        self.assertEqual(
            Or(
                *(
                    And(
                        StaticProbability(1.0),
                        TermLeaf(Term("is_number", Term("0"), List(Term("t1")))),
                        TermLeaf(Term("is_number", Term("1"), List(Term("t2")))),
                    ),
                    And(
                        StaticProbability(1.0),
                        TermLeaf(Term("is_number", Term("1"), List(Term("t1")))),
                        TermLeaf(Term("is_number", Term("0"), List(Term("t2")))),
                    ),
                ),
            ),
            and_or_tree[top],
        )

        # Check if nn leaf is correctly in there result
        element = Term("is_number", Term("1"), List(Term("t2")))
        self.assertTrue(element in and_or_tree)
        self.assertEqual(
            NNLeaf("number", 1, List(Term("t2"))),
            and_or_tree[element],
        )

    def test_probability_expression_single_element(self):
        # Create simple expression to ground
        query = create_expression_sentence_query(number_img=3, total_sum=1)

        # Create model
        model = create_mock_expression_model(query)

        context = create_mock_expression_context()

        term = Term("factor", Term("5"), List(Term("img1")))
        ct = ContextualizedTerm(context, term)
        self.assertIn(term, model.trees._and_or_tree)

        total_probability: Tensor = model.calculate_probability(contextualized_term=ct)
        print("Total Probability:", total_probability)

        self.assertTrue(total_probability.item() >= 0)

    def test_probability_expression_multiple_elements(self):
        # Create simple expression to ground
        query = create_expression_sentence_query(number_img=3, total_sum=1)

        # Create model
        model = create_mock_expression_model(query)
        context = create_mock_expression_context()

        term = Term(
            "expression", Term("1"), List(Term("img1"), Term("img2"), Term("img3"))
        )
        ct = ContextualizedTerm(context, term)
        self.assertIn(term, model.trees._and_or_tree)

        total_probability = model.calculate_probability(contextualized_term=ct)
        print("Total Probability:", total_probability)
        self.assertTrue(total_probability.item() >= 0)

    def test_multiple_queries(self):
        queries = [
            Term("addition", Term("1"), List(Term("t1"), Term("t2"))),
            Term(
                "addition",
                Term("2"),
                List(Term("t1"), Term("t2")),
            ),
        ]
        tabled_program = create_addition_program(queries)
        and_or_tree = tabled_program._and_or_tree

        for k, v in and_or_tree.items():
            print(k, v)

        # Check top result
        addition1 = Term("addition", Term("1"), List(Term("t1"), Term("t2")))
        addition2 = Term("addition", Term("2"), List(Term("t1"), Term("t2")))
        self.assertTrue(addition1 in and_or_tree)
        self.assertTrue(addition2 in and_or_tree)

    def test_wildcard_top_query_addition(self):
        """ FIRST WE DEFINE THE PROGRAM"""
        tabled_program = create_addition_program(
            Term("addition", Term("_"), List(Term("t1"), Term("t2")))
        )
        and_or_tree = tabled_program._and_or_tree

        # Check top result
        top = Term("addition", Term("_"), List(Term("t1"), Term("t2")))
        self.assertTrue(top in and_or_tree)
        self.assertTrue(len(and_or_tree[top]) == 19)

    def test_domains(self):
        program = parse_rules(
            """
               nn(number, [X], Y, class) :: is_number(Y) --> [X], {image(X)}.
               addition(N) --> is_number(N1), is_number(N2), {N is N1+N2}.
               """.strip()
        )

        domains = {"image": ["t1", "t2"], "class": [str(i) for i in range(10)]}

        query = Term("addition", Term("_"), List(Term("t1"), Term("t2")))

        builder = TabledAndOrTreeBuilder(program, verbose=True, domains=domains)
        fake_network = Network(
            "number",
            neural_model=mnist_network,
            index_list=[Term(str(i)) for i in range(10)],
        )
        networks = NetworkStore(fake_network)

        self.assertTrue("image(X)" in builder.prolog_program)

        tabled_program = builder.build_and_or_trees(networks=networks, queries=query)
        and_or_tree = tabled_program._and_or_tree

        # Check if this is superset of hardcoding domains
        hardcoded_domains_and_or_tree = create_addition_program()._and_or_tree

        for node, values in hardcoded_domains_and_or_tree.items():
            self.assertIn(node, and_or_tree)
            self.assertEqual(values, and_or_tree[node])

        self.assertEqual(and_or_tree, hardcoded_domains_and_or_tree)

    def test_static_probabilities(self):
        program_str = """
        0.1 :: is_number-->[3].
        0.9 :: is_number-->[4].
        two_numbers --> is_number, is_number.
        """

        query = Term("two_numbers", List(Term("3"), Term("4")))

        model: DeepStochLogModel = DeepStochLogModel.from_string(
            program_str, query=query, networks=NetworkStore()
        )

        query_term = ContextualizedTerm(
            context=Context({}), term=Term("two_numbers", List(Term("3"), Term("4")))
        )
        res = model.calculate_probability(query_term).numpy().tolist()
        self.assertTrue(abs(res - 0.09) < 1e-4)

    def test_normalization_addition(self):
        model_no_norm = create_mock_addition_model_with_loss_probability()
        model_with_norm = create_mock_addition_model_with_loss_probability(normalization=DeepStochLogModel.FULL_NORM)
        data = AdditionDataset(
            True,
            digit_length=1,
            size=1
        )

        # Create fake queries
        P_no_norm =[]
        P_with_norm = []
        for term in data:
            context = term.context
            for i in range(15):
                term =  Term("addition", Term(str(i)), List(Term("t1"), Term("t2")))
                contextualized_term = ContextualizedTerm(term=term, context = context)
                P_no_norm.append(model_no_norm.calculate_probability(contextualized_term))
                P_with_norm.append(model_with_norm.calculate_probability(contextualized_term))

        p_no_norm = sum(P_no_norm).detach().numpy()
        p_with_norm = sum(P_with_norm).detach().numpy()

        self.assertLess(p_no_norm, 1)
        self.assertAlmostEqual(p_with_norm, 1., delta=0.001)







if __name__ == "__main__":
    unittest.main()
