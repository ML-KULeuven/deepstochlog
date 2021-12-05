import unittest

from examples.bracket import bracket
from examples.addition import addition
from examples.mathexpression import mathexpression
from examples.wap import wap


class DeepStochLogTest(unittest.TestCase):
    def test_no_exception_addition(self):
        result = addition.run(epochs=1, train_size=1, test_size=1, verbose=False)
        self.assertTrue(result["proving_time"] > 0)

    def test_no_exception_brackets(self):
        result = bracket.run(
            max_length=4,
            train_size=1,
            epochs=1,
            test_size=5,
            test_example_idx=[0, 1],
            verbose=True,
            allow_uneven=False,
        )
        self.assertTrue(result["grounding_time"] > 0)

    def test_no_exception_expression(self):
        time = mathexpression.run(
            epochs=1,
            train_size=1,
            test_size=1,
            verbose=False,
            expression_max_length=3,
            device_str="cpu",
        )
        self.assertTrue(time > 0)

    def test_no_exception_wap(self):
        result = wap.run(
            epochs=1,
            train_size=1,
            test_size=1,
            verbose=False,
        )
        self.assertTrue(result["grounding_time"] > 0)
