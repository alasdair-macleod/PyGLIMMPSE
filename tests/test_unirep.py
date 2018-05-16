from unittest import TestCase
import numpy as np
from pyglimmpse.finv import finv

from pyglimmpse import unirep
from pyglimmpse.constants import Constants
from pyglimmpse.input import Option
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.unirep import _err_checking, _calc_multipliers_est_sigma, _calc_undf1_undf2, \
    geisser_greenhouse_muller_edwards_simpson_taylor_2007, _calc_epsilon


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

    def test_unirep_power_estimated_sigma_hf(self):
        """ case 1: should return expected value, for hf method """
        expected = 0.98471
        approximation = Constants.HF
        unirepmethod = Constants.UCDF_MULLER2004_APPROXIMATION
        alpha = 0.05
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20

        n_est = 10
        rank_est = 1
        tolerance = 0.000000000000001

        alpha_cl = 0.01
        alpha_cu = 0.01

        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                     [0.625, 1.25, -1.25, 0.625],
                                     [-0.625, -1.25, 1.25, -0.625],
                                     [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                      [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                      [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                      [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        exeps = 0.7203684
        eps = 0.7203684
        result = unirep.unirep_power_estimated_sigma(rank_C, rank_U, total_N, rank_X, error_sum_square, hypo_sum_square, exeps, eps, alpha,
                                                    approximation, unirepmethod, n_est, rank_est, alpha_cl, alpha_cu, tolerance)
        actual = result.power
        self.assertAlmostEqual(actual, expected, places=5)

    def test_unirep_power_estimated_sigma_gg(self):
        """ case 2: should return expected value, for gg method """
        expected = 0.97442
        alpha = 0.05
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20

        alpha_cl = 0.01
        alpha_cu = 0.01
        tolerance = 0.000000000000001

        approximation = Constants.GG
        unirepmethod = Constants.UCDF_MULLER2004_APPROXIMATION

        n_est=10
        rank_est=1

        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                     [0.625, 1.25, -1.25, 0.625],
                                     [-0.625, -1.25, 1.25, -0.625],
                                     [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                      [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                      [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                      [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        sigmastareval = np.matrix([[0.23555], [0.17123], [0.05561], [0.04721]])
        sigmastarevec = np.matrix([[-1, 4.51e-17, -2.01e-16, -4.61e-18],
                                   [2.776e-17, 1, -3.33e-16, -2.39e-16],
                                   [-2.74e-16, 2.632e-16, 1, 2.001e-16],
                                   [-4.61e-18, 2.387e-16, -2e-16, 1]])

        sigma_star = np.multiply(sigmastareval, sigmastarevec)
        e = geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star, rank_U, total_N, rank_X)
        ep = _calc_epsilon(sigma_star, rank_U)

        exeps = 0.7203684
        eps = 0.7203684
        result = unirep.unirep_power_estimated_sigma(rank_C, rank_U, total_N, rank_X, error_sum_square, hypo_sum_square, exeps, eps, alpha,
                                                    approximation, unirepmethod, n_est, rank_est, alpha_cl, alpha_cu, tolerance)
        actual = result.power
        self.assertAlmostEqual(actual, expected, places=5)
