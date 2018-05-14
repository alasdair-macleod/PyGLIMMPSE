from unittest import TestCase
import numpy as np
from pyglimmpse.finv import finv

from pyglimmpse import unirep
from pyglimmpse.constants import Constants
from pyglimmpse.input import Option
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.unirep import _err_checking, _calc_multipliers_est_sigma, _calc_defaults, _calc_undf1_undf2, \
    geisser_greenhouse_muller_edwards_simpson_taylor_2007


class TestUnirep(TestCase):
    

    def test_hfexeps(self):
        """ should return expected value """
        expected = 0.2901679
        actual = unirep.hyuhn_feldt_muller_edwards_simpson_taylor_2004(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_cmexeps(self):
        """ should return expected value """
        expected = 0.2757015
        actual = unirep.chi_muller_muller_edwards_simpson_taylor_2004(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_ggexeps(self):
        """ should return expected value """
        expected = 0.2975125
        actual = unirep.geisser_greenhouse_muller_edwards_simpson_taylor_2004(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                rank_U=3,
                                total_N=20,
                                rank_X=5)
        self.assertAlmostEqual(actual, expected, delta=0.0000001)

    def test_intermediates(self):
        rank_U = 3
        total_N = 20
        rank_X = 5
        rank_C = 1
        sigma_star = np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        rep_n = 1

        essh = (theta - theta_zero).T * np.linalg.solve(m_matrix, np.identity(np.shape(m_matrix)[0])) * (theta - theta_zero)

        error_sum_square = sigma_star * (total_N - rank_X)  # E
        hypo_sum_square = rep_n * essh  # H

        epsilon = Epsilon(sigma_star, rank_U)

        expected_epsilon = geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star, rank_U, total_N, rank_X)
        # exeps = 0.2757015

        nue = total_N - rank_X
        undf1, undf2 = _calc_undf1_undf2(Option, expected_epsilon, nue, rank_C, rank_U)
        # Create defaults - same for either SIGMA known or estimated
        q1, q2, q3, q4, q5, lambar = _calc_defaults(error_sum_square, hypo_sum_square, nue, rank_U)

        cl1df, e_1_2, e_3_5, e_4, omegaua = _calc_multipliers_est_sigma(None, Option, epsilon.eps, nue, q1, q2, q3, q4, q5,
                                                                        rank_C,
                                                                        rank_U, Constants.UCDF_MULLER2004_APPROXIMATION)
        # Error checking
        e_1_2 = _err_checking(e_1_2, rank_U)

        # Obtain noncentrality and critical value for power point estimate
        omega = e_3_5 * q2 / lambar
        if Option.opt_calc_cm:
            omega = omegaua

        fcrit = finv(1 - 0.05, undf1 * e_1_2, undf2 * e_1_2)
        a = 1

