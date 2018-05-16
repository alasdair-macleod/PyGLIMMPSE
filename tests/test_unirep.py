from unittest import TestCase
import numpy as np
from pyglimmpse.finv import finv

from pyglimmpse import unirep
from pyglimmpse.constants import Constants
from pyglimmpse.input import Option
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.unirep import _err_checking, _calc_multipliers_est_sigma, _calc_undf1_undf2, \
    geisser_greenhouse_muller_edwards_simpson_taylor_2007


class TestUnirep(TestCase):
    

    def test_hfexeps(self):
        """ should return expected value """
        expected = 0.2901679
        actual = unirep.hyuhn_feldt_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_cmexeps(self):
        """ should return expected value """
        expected = 0.2757015
        actual = unirep.chi_muller_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_ggexeps(self):
        """ should return expected value """
        expected = 0.2975125
        actual = unirep.geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, delta=0.0000001)

