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
        rank_X = 1
        total_N = 10
        error_sum_square = np.matrix([[18]])
        hypothesis_sum_square = np.matrix([[10]])
        relative_group_sizes = [1]
        rep_N = 10
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        alpha = 0.05
        expected = 0.514351069870
        actual = multirep.hlt_one_moment_null_approximator(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def calc_sigma_star(self, total_n, rank_x, error_sum_square):
        return error_sum_square / (total_n-rank_x)

    def calc_delta_es(self, rep_N, hypothesis_sum_square):
        return hypothesis_sum_square / rep_N

    def test_hlt_one_moment_null_approximator_min_rakn_C_U_2(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850], [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.203028693139
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.hlt_one_moment_null_approximator(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_hlt_two_moment_null_approximator(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850], [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.132606902708
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.hlt_two_moment_null_approximator(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_hlt_one_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850], [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        # expected = 0.229495804549
        expected = 0.13261
        rep_N = 10
        relative_group_sizes = [1, 1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.hlt_two_moment_null_approximator(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_hlt_two_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850], [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.209176173376
        expected = 0.13261
        rep_N = 10
        relative_group_sizes = [1, 1]
        sigma_star = self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es = self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.hlt_two_moment_null_approximator(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_pbt_one_moment_null_approx(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.209723893717
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.pbt_one_moment_null_approx(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_pbt_two_moment_null_approx(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected =0.222786795810
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.pbt_two_moment_null_approx(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)

        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_pbt_one_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.214237639104
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.pbt_one_moment_null_approx_obrien_shieh(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_pbt_two_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.205398851860
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.pbt_two_moment_null_approx_obrien_shieh(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)

        self.assertEqual(round(expected, 5), round(actual.power, 5))

    def test_wlk(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.2221 # 0.138179071626
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.wlk_two_moment_null_approx(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_wlk_os(self):
        """
        This should return the expected value
        """

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.222089414105
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.wlk_two_moment_null_approx_obrien_shieh(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))


    def test_special(self):
        """
        This should return the expected value
        """
        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        expected = 0.197974983838
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.special(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 4), round(actual.power, 4))

    def test_special_2(self):
        """
        This should return the expected value
        """
        rank_C = 1
        rank_U = 1
        rank_X = 1
        total_N = 10
        error_sum_square = np.matrix([[18]])
        hypothesis_sum_square = np.matrix([[10]])
        alpha = 0.05
        expected = 0.514351069870
        rep_N = total_N
        relative_group_sizes = [1]
        sigma_star=self.calc_sigma_star(total_N, rank_X, error_sum_square)
        delta_es=self.calc_delta_es(rep_N, hypothesis_sum_square)
        actual = multirep.pbt_one_moment_null_approx(rank_C=rank_C,
                                                           rank_X=rank_X,
                                                           relative_group_sizes=relative_group_sizes,
                                                           rep_N=rep_N,
                                                           alpha=alpha,
                                                           sigma_star=sigma_star,
                                                           delta_es=delta_es)
        self.assertEqual(round(expected, 5), round(actual.power, 5))



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
        total_N = 10

        expected = Power(0.138179071626, 3, Constants.FMETHOD_NORMAL_LR)
        actual = multirep._multi_power(alpha, df1, df2, omega, total_N)
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




