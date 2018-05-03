from unittest import TestCase
import numpy as np
import pyglimmpse.multirep as multirep
from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power


class TestMultirep(TestCase):
    def test_hlt_one_moment_null_approximator(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.5])
        alpha = 0.05
        expected = 0.138179071626
        actual = multirep.hlt_one_moment_null_approximator(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_hlt_one_moment_null_approximator_min_rakn_C_U_2(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.5])
        alpha = 0.05
        expected = 0.138179071626
        actual = multirep.hlt_one_moment_null_approximator(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_hlt_two_moment_null_approximator(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 5
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 1
        actual = multirep.hlt_two_moment_null_approximator(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 3), round(actual.power, 3))

    def test_hlt_one_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 3
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.326487905774
        actual = multirep.hlt_one_moment_null_approximator_obrien_shieh(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))

    def test_hlt_two_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 5
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.9998
        actual = multirep.hlt_two_moment_null_approximator_obrien_shieh(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_pbt_one_moment_null_approx(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 3
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.326487905774
        actual = multirep.pbt_one_moment_null_approx(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))

    def test_pbt_two_moment_null_approx(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.5])
        alpha = 0.05
        expected = 0.138179071626
        actual = multirep.pbt_two_moment_null_approx(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_pbt_one_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 3
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.326487905774
        actual = multirep.pbt_one_moment_null_approx_obrien_shieh(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))

    def test_pbt_two_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.05])
        alpha = 0.05
        expected = 0.058149316 # 0.138179071626
        actual = multirep.pbt_two_moment_null_approx_obrien_shieh(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 3), round(actual.power, 3))

    def test_wlk(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.5])
        alpha = 0.05
        expected = 0.107399189 # 0.138179071626
        actual = multirep.wlk_two_moment_null_approx(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 3), round(actual.power, 3))

    def test_wlk_os(self):
        """
        This should return the expected value
        """

        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 3
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.326487905774
        actual = multirep.wlk_two_moment_null_approx_obrien_shieh(rank_C, rank_U, rank_X, total_N, eval_HINVE, alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))


    def test_special(self):
        """
        This should return the expected value
        """
        rank_C = 1
        rank_U = 2
        rank_X = 1
        total_N = 3
        eval_HINVE = np.array([35])
        alpha = 0.05
        expected = 0.326487905774
        actual = multirep.special(rank_C, rank_U, rank_X,total_N,eval_HINVE,alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))

    def test_special_2(self):
        """
        This should return the expected value
        """
        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        eval_HINVE = np.array([0.5])
        alpha = 0.05
        expected = 0.138179071626
        actual = multirep.special(rank_C, rank_U, rank_X,total_N,eval_HINVE,alpha)
        self.assertEqual(round(expected, 2), round(actual.power, 2))



    def test_df1_rank_c_u(self):
        rank_C = 2
        rank_U = 3
        expected = 6
        actual = multirep._df1_rank_c_u(rank_C, rank_U)
        self.assertEqual(expected, actual)

    def test_multi_power(self):
        alpha = 0.05
        df1 = 3
        df2 = 4
        omega = 3

        expected = Power(0.138179071626, 3, Constants.FMETHOD_NORMAL_LR)
        actual = multirep._multi_power(alpha, df1, df2, omega)
        self.assertEqual(round(expected.power, 4), round(actual.power, 4))
        self.assertEqual(expected.noncentrality_parameter, actual.noncentrality_parameter)

    def test_trace(self):
        eval_HINVE = np.array([0.5])
        rank_X = 5
        total_N = 10
        expected = np.array([0.25])
        actual = multirep._trace(eval_HINVE, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_calc_omega(self):
        min_rank_C_U = 1
        eval_HINVE = np.array([0.5])
        rank_X = 5
        total_N = 10
        expected = 2.5
        actual = multirep._calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_calc_hlt_omega(self):
        min_rank_C_U = 1
        eval_HINVE = np.array([0.5])
        rank_X = 5
        total_N = 10
        expected = 2.5
        df2 = 2
        actual = multirep._calc_hlt_omega(min_rank_C_U, eval_HINVE, rank_X, total_N, df2)
        self.assertEqual(expected, actual)

    def test_hlt_one_moment_df2(self):
        rank_C = 1
        rank_U = 2
        rank_X = 3
        total_N = 10
        expected = 6
        actual = multirep._hlt_one_moment_df2(rank_C, rank_U, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_hlt_two_moment_df2(self):
        rank_C = 1
        rank_U = 2
        rank_X = 3
        total_N = 10
        expected = 6
        actual = multirep._hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_valid_df2_eigenvalues(self):
        eval_HINVE = [1]
        df2 = 1
        fail_hinve = [float('nan')]
        fail_df2 = 0

        self.assertTrue(multirep._valid_df2_eigenvalues(eval_HINVE,df2))
        self.assertFalse(multirep._valid_df2_eigenvalues(fail_hinve,df2))
        self.assertFalse(multirep._valid_df2_eigenvalues(eval_HINVE,fail_df2))


    def test_pbt_one_moment_df2(self):
        rank_C = 1
        rank_U = 2
        rank_X = 3
        total_N = 10
        expected = 6
        actual = multirep._pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_pbt_two_moment_df1_df2(self):
        rank_C = 1
        rank_U = 3
        rank_X = 2
        total_N = 8
        exp_df1, exp_df2 = 3, 4
        actual_df1, actual_df2 = multirep._pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)
        self.assertEqual(round(exp_df1, 12), round(actual_df1, 12))
        self.assertEqual(round(exp_df2, 12), round(actual_df2, 12))

    def test_pbt_population_value(self):
        evalt = 1
        min_rank_C_U  =1
        expected = 0.5
        actual = multirep._pbt_population_value(evalt, min_rank_C_U)
        self.assertEqual(expected, actual)

    def test_pbt_uncorrected_evalt(self):
        eval_HINVE =np.array([35])
        rank_C = 2
        rank_U = 2
        rank_X = 1
        total_N = 3
        expected = np.array([35])
        actual = multirep._pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_undefined_power(self):
        actual = multirep._undefined_power()
        self.assertTrue(np.isnan(actual.power))




