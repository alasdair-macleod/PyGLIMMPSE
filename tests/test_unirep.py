from unittest import TestCase
import numpy as np

from pyglimmpse import unirep
from pyglimmpse.model import epsilon

from pyglimmpse.constants import Constants
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.unirep import _geisser_greenhouse_muller_edwards_simpson_taylor_2007, _calc_epsilon, OptionalArgs


class ConfidenceInterval(object):
    """
    Class holding required information to calculate confidence intervals.
    """

    def __init__(self,
                 beta_known: bool = True,
                 lower_tail: float = 0.05,
                 upper_tail: float = 0.0,
                 rank_est: int = 1,
                 n_est: int = 1):
        """
        Class describing a Confidence Interval
        :param beta_known:
        :param lower_tail:
        :param upper_tail:
        :param rank_est:
        :param n_est:
        :param kwargs:
        """
        self.beta_known = beta_known
        self.lower_tail = lower_tail
        self.upper_tail = upper_tail
        self.rank_est = rank_est
        self.n_est = n_est

class TestUnirep(TestCase):

    def test_geisser_greenhouse_muller_barton_1989(self):
        expected = 0.2871105857
        actual = unirep._geisser_greenhouse_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                               rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_geisser_greenhouse_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2975124504
        actual = unirep._geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                               rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)
    def test_hfexeps(self):
        """ should return expected value """
        expected = 0.2901679
        actual = unirep._hyuhn_feldt_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                        rank_U=3,
                                                                        total_N=20,
                                                                        rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_chi_muller_muller_barton_1989(self):
        expected = 0.3412303
        actual = unirep._chi_muller_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                       rank_U=3,
                                                       total_N=20,
                                                       rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_chi_muller_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2757015
        actual = unirep._chi_muller_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                       rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)
        actual = unirep._chi_muller_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                       rank_U=3,
                                                                       total_N=20,
                                                                       rank_X=5)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_hyuhn_feldt_muller_barton_1989(self):
        expected = 0.3591350780
        actual = unirep._hyuhn_feldt_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                        rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_hyuhn_feldt_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2901678808
        actual = unirep._hyuhn_feldt_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                        rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_hf_derivs_functions_eigenvalues(self):
        """ should return expected value """
        expected_bh_i = np.matrix([[0.0597349430], [0.8662479536], [0.9186891354]])[::-1]
        expected_bh_ii = np.matrix([[-0.1092718067], [0.3256505266], [0.5747431889]])[::-1]
        expected_h1 = 17.7024794
        expected_h2 = 16.2314045
        eps = epsilon.Epsilon(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]), rank_U=3)
        actual = unirep._hf_derivs_functions_eigenvalues(rank_U=3, rank_X=5, total_N=20, epsilon=eps)
        self.assertTrue((actual[0] == expected_bh_i).all)
        self.assertTrue((actual[1] == expected_bh_ii).all)
        self.assertAlmostEqual(actual[2], expected_h1, places=5)
        self.assertAlmostEqual(actual[3], expected_h2, places=5)
        expected = 0.2975125
        actual = unirep._geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                               rank_U=3,
                                                                               total_N=20,
                                                                               rank_X=5)
        self.assertAlmostEqual(actual, expected, delta=0.0000001)

    def test_gg_derivs_functions_eigenvalues(self):
        """ should return expected value """
        expected_f_i = np.matrix([[0.0400189375], [0.5803357469], [0.6154682886]])[::-1]
        expected_f_ii = np.matrix([[-0.0738858283], [0.0751513782], [0.2241890031]])[::-1]
        eps = epsilon.Epsilon(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]), rank_U=3)
        actual = unirep._gg_derivs_functions_eigenvalues(epsilon=eps, rank_U=3)
        self.assertTrue((actual[0] == expected_f_i).all)
        self.assertTrue((actual[1] == expected_f_ii).all)
    def test_unirep_power_estimated_sigma_hf(self):
        """ case 1: should return expected value, for hf method """
        expected = 0.98471
        unirep_method = Constants.HF
        approximation_method = Constants.UCDF_MULLER2004_APPROXIMATION
        alpha = 0.05
        rank_C = 1
        rank_U = 4
        rank_X = 1
        total_N = 20

        confidence_interval = ConfidenceInterval(beta_known=False,
                                                 lower_tail=0.01,
                                                 upper_tail=0.01,
                                                 n_est=10,
                                                 rank_est=1)

        tolerance = 0.000000000000001


        hypo_sum_square = np.matrix([[0.3125, 0.625, -0.625, 0.3125],
                                     [0.625, 1.25, -1.25, 0.625],
                                     [-0.625, -1.25, 1.25, -0.625],
                                     [0.3125, 0.625, -0.625, 0.3125]])
        error_sum_square = np.matrix([[4.47545, -3.3e-17, 1.055e-15, 1.648e-17],
                                      [-3.3e-17, 3.25337, -8.24e-18, 5.624e-16],
                                      [1.055e-15, -8.24e-18, 1.05659, -3.19e-17],
                                      [1.648e-17, 5.624e-16, -3.19e-17, 0.89699]])
        exeps = 0.7203684
        # eps = 0.7203684
        sigma_star = error_sum_square / (total_N - rank_X)
        eps = _calc_epsilon(sigma_star=sigma_star, rank_U=rank_U)

        result = unirep._unirep_power_estimated_sigma(rank_C=rank_C,
                                                      rank_U=rank_U,
                                                      total_N=total_N,
                                                      rank_X=rank_X,
                                                      sigma_star=error_sum_square/(total_N-rank_X),
                                                      hypo_sum_square=hypo_sum_square,
                                                      expected_epsilon=exeps,
                                                      epsilon=eps,
                                                      alpha=alpha,
                                                      unirep_method=unirep_method,
                                                      epsilon_estimator=Constants.EPSILON_MULLER2004,
                                                      unirepmethod=Constants.SIGMA_KNOWN,
                                                      confidence_interval=confidence_interval,
                                                      n_ip=1,
                                                      rank_ip=1,
                                                      tolerance=tolerance)
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

        confidence_interval = ConfidenceInterval(beta_known=False,
                                                 lower_tail=0.01,
                                                 upper_tail=0.01,
                                                 n_est=10,
                                                 rank_est=1)

        tolerance = 0.000000000000001

        unirep_method = Constants.GG
        approximation_method = Constants.UCDF_MULLER2004_APPROXIMATION

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
        e = _geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star, rank_U, total_N, rank_X)
        ep = _calc_epsilon(sigma_star, rank_U)

        exeps = 0.7203684
        # eps = 0.7203684
        sigma_star = error_sum_square / (total_N - rank_X)
        eps = _calc_epsilon(sigma_star=sigma_star, rank_U=rank_U)

        result = unirep._unirep_power_estimated_sigma(rank_C=rank_C,
                                                      rank_U=rank_U,
                                                      total_N=total_N,
                                                      rank_X=rank_X,
                                                      sigma_star=error_sum_square/(total_N-rank_X),
                                                      hypo_sum_square=hypo_sum_square,
                                                      expected_epsilon=exeps,
                                                      epsilon=eps,
                                                      alpha=alpha,
                                                      unirep_method=unirep_method,
                                                      epsilon_estimator=Constants.EPSILON_MULLER2004.value,
                                                      unirepmethod=Constants.SIGMA_KNOWN.value,
                                                      confidence_interval=confidence_interval,
                                                      n_ip=1,
                                                      rank_ip=1,
                                                      tolerance=tolerance)
        actual = result.power
        self.assertAlmostEqual(actual, expected, places=5)
