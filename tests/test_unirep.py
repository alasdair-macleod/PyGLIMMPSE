from unittest import TestCase
import numpy as np
from pyglimmpse import unirep
from pyglimmpse.model import epsilon


class TestUnirep(TestCase):

    def test_geisser_greenhouse_muller_barton_1989(self):
        expected = 0.2871105857
        actual = unirep.geisser_greenhouse_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                              rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_geisser_greenhouse_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2975124504
        actual = unirep.geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                              rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_chi_muller_muller_barton_1989(self):
        expected = 0.3412303
        actual = unirep.chi_muller_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                      rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_chi_muller_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2757015
        actual = unirep.chi_muller_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                                      rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_hyuhn_feldt_muller_barton_1989(self):
        expected = 0.3591350780
        actual = unirep.hyuhn_feldt_muller_barton_1989(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
                                                       rank_U=3, total_N=20, rank_X=5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_hyuhn_feldt_muller_edwards_simpson_taylor_2007(self):
        expected = 0.2901678808
        actual = unirep.hyuhn_feldt_muller_edwards_simpson_taylor_2007(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]),
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

    def test_gg_derivs_functions_eigenvalues(self):
        """ should return expected value """
        expected_f_i = np.matrix([[0.0400189375], [0.5803357469], [0.6154682886]])[::-1]
        expected_f_ii = np.matrix([[-0.0738858283], [0.0751513782], [0.2241890031]])[::-1]
        eps = epsilon.Epsilon(sigma_star=np.matrix([[1, 2, 3], [3, 4, 5], [4, 5, 6]]), rank_U=3)
        actual = unirep._gg_derivs_functions_eigenvalues(epsilon=eps, rank_U=3)
        self.assertTrue((actual[0] == expected_f_i).all)
        self.assertTrue((actual[1] == expected_f_ii).all)