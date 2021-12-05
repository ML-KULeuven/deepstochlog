import unittest

import examples.bracket.bracket
from examples.bracket.bracket_data import (
    create_bracket_language_of_length,
    create_bracket_language,
    BracketDataset,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
# from utils import calculate_probability_predictions


class BracketTest(unittest.TestCase):
    def test_bracket_creation_exact_length(self):
        self.assertEqual(["()"], create_bracket_language_of_length(2))
        self.assertEqual({"()()", "(())"}, set(create_bracket_language_of_length(4)))
        self.assertEqual(
            {"(()())", "((()))", "()()()", "()(())", "(())()"},
            set(create_bracket_language_of_length(6)),
        )
        self.assertEqual(
            {
                "((()()))",
                "(((())))",
                "(()()())",
                "(()(()))",
                "((())())",
                "(()())()",
                "((()))()",
                "()()()()",
                "()(())()",
                "(())()()",
                "()(()())",
                "()((()))",
                "()()()()",
                "()()(())",
                "()(())()",
                "()()()()",
                "(())()()",
                "()()(())",
                "(())(())",
            },
            set(create_bracket_language_of_length(8)),
        )

    def test_bracket_creation_max_length(self):
        self.assertEqual(["()"], create_bracket_language(0, 2))
        self.assertEqual({"()", "()()", "(())"}, set(create_bracket_language(0, 4)))
        self.assertEqual(
            len(set(create_bracket_language(0, 10))),
            len(create_bracket_language(0, 10)),
        )

    def test_lower_than_one_probability(self):
        max_length = 10

        # Train model
        results = examples.bracket.bracket.run(
            min_length=2,
            max_length=max_length,
            allow_uneven=False,
            epochs=1,
            train_size=500,
            test_size=None,
            seed=42,
        )
        model: DeepStochLogModel = results["model"]

        # Create test data
        test_size = 100
        test_data = BracketDataset(
            split="test",
            size=test_size,
            min_length=2,
            max_length=max_length,
            allow_uneven=False,
            seed=42,
        )

        # Predict for test data
        # all_expected, all_predicted = calculate_probability_predictions(
        #     model, DataLoader(test_data, batch_size=test_size)
        # )

        # for predicted in all_predicted:
        #     self.assertGreaterEqual(1, predicted)
        #     self.assertLessEqual(0, predicted)


if __name__ == "__main__":
    unittest.main()
