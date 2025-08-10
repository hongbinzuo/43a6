import unittest

from pi_200 import format_pi


class TestPiComputation(unittest.TestCase):
    def test_first_50_digits_exact(self):
        # Known 50 decimal places of pi
        expected_50 = (
            "3.14159265358979323846264338327950288419716939937510"
        )
        self.assertEqual(format_pi(50), expected_50)

    def test_200_digits_length_and_prefix(self):
        pi_200 = format_pi(200)
        # 1 for '3', 1 for '.', plus 200 decimal digits
        self.assertEqual(len(pi_200), 202)
        self.assertTrue(pi_200.startswith("3.14159265358979323846264338327950288419716939937510"))


if __name__ == "__main__":
    unittest.main()