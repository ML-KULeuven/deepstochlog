import unittest
from typing import List

from pyparsing import ParseException

from deepstochlog.parser import parse_rules
from deepstochlog.rule import (
    NDCGRule,
    NeuralProbabilityAnnotation,
    StaticProbabilityAnnotation,
    TrainableProbabilityAnnotation,
    Rule,
    Fact,
    ClauseRule,
)


class ParserTestCase(unittest.TestCase):
    def test_static_probability(self):
        test_program = "0.2 :: foo --> a."
        result = parse_rules(test_program).rules
        self.assertEqual("0.2", str(result[0].probability))

    def test_body_period_exceptions_parsing(self):
        body = 'a(0.4), 0.3, ["hello."], {b(X,Y)}'
        test_program = "0.2 :: foo --> " + body + ".\n" "0.1 :: bar --> a."
        resulting_rules: List[NDCGRule] = parse_rules(test_program).get_ndcg_rules()

        print("Resulting rules:\n", resulting_rules)

        self.assertEqual(2, len(resulting_rules))
        self.assertEqual(body, resulting_rules[0].body)

        program_str = "\n".join([str(r) for r in resulting_rules])
        print(program_str)
        self.assertEqual(
            test_program.replace(" ", ""),
            program_str.replace(" ", ""),
        )

    def test_all_types_probability(self):
        test_program = (
            "0.2 :: foo --> a.\n"
            "nn(mnist, [X], Y, digit) :: bla(A,B) --> bla, h.\n"
            "t(_) :: bar(X) --> a(0,1), c(_,X)."
        )
        program = parse_rules(test_program)
        resulting_rules: List[NDCGRule] = program.get_ndcg_rules()
        self.assertEqual(3, len(resulting_rules))
        self.assertEqual(
            StaticProbabilityAnnotation, type(resulting_rules[0].probability)
        )
        self.assertEqual(
            NeuralProbabilityAnnotation, type(resulting_rules[1].probability)
        )
        self.assertEqual(
            TrainableProbabilityAnnotation, type(resulting_rules[2].probability)
        )

        program_str = "\n".join([str(r) for r in resulting_rules])
        print(program_str)
        self.assertEqual(
            test_program.replace(" ", ""),
            program_str.replace(" ", ""),
        )

    def test_fact(self):
        fact_rule = parse_rules("a.").rules
        print(fact_rule)
        self.assertEqual(1, len(fact_rule))
        self.assertEqual("a.", str(fact_rule[0]))
        self.assertTrue(type(fact_rule[0]) == Fact)

    def test_all_types_rules(self):
        test_program = """
        a.
        b :- c.
        1.0 :: e --> f.
        """
        resulting_rules: List[Rule] = parse_rules(test_program).rules
        self.assertEqual(3, len(resulting_rules))
        self.assertEqual(Fact, type(resulting_rules[0]))
        self.assertEqual(ClauseRule, type(resulting_rules[1]))
        self.assertEqual(NDCGRule, type(resulting_rules[2]))

        program_str = "\n".join([str(r) for r in resulting_rules])
        print(program_str)
        self.assertEqual(
            test_program.replace(" ", "").strip(),
            program_str.replace(" ", "").strip(),
        )

    def test_term_arguments(self):
        program_str = """
        1.0 :: rep(s(N), C) --> [X].
        """
        program = parse_rules(program_str)
        self.assertEqual(1, len(program.rules))
        self.assertTrue(isinstance(program.rules[0], NDCGRule))

        program_stringified = str(program)
        self.assertEqual(
            program_str.replace(" ", "").strip(),
            program_stringified.replace(" ", "").strip(),
        )

    def test_forgotten_period_exception_raised(self):
        program_str = """
        letter(X) :- member(X,[a,b,c])
        nn(mnist, [X], C, letter) :: rep(s(N), C) --> [X], rep(N, C).
        """
        self.assertRaises(ParseException, parse_rules, program_str)

    def test_anbncn(self):
        program_str = """
        0.5 :: s(0) --> akblcm(K,L,M),{K\\=L; L\\=M; M\\=K}, {K \\= 0, L \\= 0, M \\= 0}.
        0.5 :: s(1) --> akblcm(N,N,N).
        akblcm(K,L,M) --> rep(K,A), rep(L,B), rep(M,C),{A\\=B, B\\=C, C\\=A}.
        rep(0, _) --> [].
        rep(s(N), C) --> [X], rep(N,C), {domain(C, [a,b,c]), nn(mnist, X, C)}.
        """
        program = parse_rules(program_str)
        self.assertEqual(5, len(program.rules))

    def test_trainable_probability(self):
        program_str = """
        t(_) :: a --> b.
        t(_) :: a --> c.
        t(_) :: a --> d.
        """
        program = parse_rules(program_str)

        # Basic sanity checks in parser
        self.assertEqual(3, len(program.rules))
        for i in range(3):
            rule: Rule = program.rules[i]
            self.assertTrue(isinstance(rule, NDCGRule))
            if isinstance(rule, NDCGRule):
                self.assertTrue(
                    isinstance(rule.probability, TrainableProbabilityAnnotation)
                )
            else:
                self.fail("Not a proper NDCG rule: " + str(rule))
        self.assertTrue(program.has_trainable_probabilities())

        # Transform program
        program, networks = program._transform_trainable_probabilities_to_switches()
        self.assertFalse(program.has_trainable_probabilities())
        self.assertEqual(1, len(networks.networks))
        print(program)


    def test_list_argument(self):
        program_str = """
        t(_) :: a([0|Rest]) --> a(Rest).
        t(_) :: a([0,1|Rest]) --> a(Rest).
        """
        program = parse_rules(program_str)
        self.assertEqual(2, len(program.rules))

    def test_list_argument_string(self):
        program_str = """
        t(_) :: a(['a'|Rest]) --> a(Rest).
        t(_) :: a(["a"]) --> a(Rest).
        t(_) :: a(["hello world!"]) --> a(Rest).
        """
        program = parse_rules(program_str)
        self.assertEqual(3, len(program.rules))

    def test_matrix(self):
        program_str = """
        t(_) :: make_matrix([0, 3|Rest], [['x', 'x', 'x']|Rest1]) --> make_matrix(Rest, Rest1).
        """
        program = parse_rules(program_str)
        self.assertEqual(1, len(program.rules))


if __name__ == "__main__":
    unittest.main()
