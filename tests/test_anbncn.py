import unittest

from examples.anbncn.anbncn_data import (
    create_anbncn_language,
    create_non_anbncn_language,
)


class BracketTest(unittest.TestCase):
    def test_anbncn_creation(self):
        self.assertEqual(["abc"], create_anbncn_language(min_length=3, max_length=3))
        self.assertEqual(
            ["abc", "aabbcc"], create_anbncn_language(min_length=3, max_length=6)
        )
        self.assertEqual(
            ["abc", "aabbcc", "aaabbbccc"],
            create_anbncn_language(min_length=3, max_length=9),
        )

    def test_non_anbncn_creation_empty(self):
        self.assertEqual(
            [],
            create_non_anbncn_language(
                min_length=3, max_length=3, allow_non_threefold=False
            ),
        )
        self.assertEqual(
            [],
            create_non_anbncn_language(
                min_length=3, max_length=5, allow_non_threefold=False
            ),
        )

    def test_non_anbncn_creation(self):
        expected = list(
            [
                "aaaabc",
                "aaabbc",
                "aaabcc",
                "aabbbc",
                "aabccc",
                "abbbbc",
                "abbbcc",
                "abbccc",
                "abcccc",
            ]
        )
        expected.sort()
        calculated = create_non_anbncn_language(
            min_length=3, max_length=6, allow_non_threefold=False
        )
        calculated.sort()
        self.assertEqual(expected, calculated)

    def test_non_anbncn_creation_larger(self):
        calculated = create_non_anbncn_language(
            min_length=3, max_length=12, allow_non_threefold=False
        )
        print(calculated)
        self.assertLess(20, len(calculated))


if __name__ == "__main__":
    unittest.main()
