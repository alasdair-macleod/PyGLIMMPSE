from unittest import TestCase
import numpy as np

from pyglimmpse import samplesize
from pyglimmpse.multirep import hlt_two_moment_null_approximator


# class TestSamplesize(TestCase):
#     def test_samplesize(self):
#         """
#
#         :return:
#         """
#         test = hlt_two_moment_null_approximator
#         rank_C = 1
#         rank_U = 1
#         rank_X = 1
#         expected = 10
#         eval_HINVE = np.array([0.5102041])
#         alpha = 0.05
#         sigmaScale = 1
#         betaScale = 1
#         sigma = np.matrix([[1]])
#         beta = np.matrix([[1]])
#         targetPower = 0.514351069870
#         actual = samplesize.samplesize(test, rank_C, rank_U, alpha, sigmaScale, sigma, betaScale, beta, targetPower, rank_X, eval_HINVE)
        self.assertEqual(expected, actual)