from unittest import TestCase

import numpy as np

from pyglimmpse.finv import finv


class TestFinv(TestCase):

    def test_largedf1(self):
        actual = finv(0.05, 10**8, 1)
        self.assertTrue(np.isnan(actual))

    # def test_largedf2(self):
    #     actual = finv(0.05, 1, 10**9.5)
    #     self.assertTrue(np.isnan(actual))

    def test_negativedf1(self):
        actual = finv(0.05, -1, 1)
        self.assertTrue(np.isnan(actual))

    def test_negativedf2(self):
        actual = finv(0.05, 1, -1)
        self.assertTrue(np.isnan(actual))

    def test_finv(self):
        expected = 0.7185356
        actual = finv(0.05, 100, 100)
        result = round(actual, 7)
        self.assertEqual(expected, result)

    def test_finv_large_df2(self):
        expected = 6.6348966
        actual = finv(0.99, 1, 10000000000)
        result = round(actual, 7)
        self.assertEqual(expected, result)