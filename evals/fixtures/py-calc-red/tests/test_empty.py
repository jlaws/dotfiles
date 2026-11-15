import unittest

from calc.ops import mean


class EmptyMeanTests(unittest.TestCase):
    def test_mean_of_empty_list_is_zero(self):
        self.assertEqual(mean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
