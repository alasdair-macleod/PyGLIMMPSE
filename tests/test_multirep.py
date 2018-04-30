from unittest import TestCase
import numpy as np
import pyglimmpse.multirep as multirep
from pyglimmpse.constants import Constants
from pyglimmpse.input import Scalar, CL
from pyglimmpse.model.power import Power


class TestMultirep(TestCase):
    def test_hlt_one_moment_null_approximator(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.hlt_one_moment_null_approximator(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
                alphatest=0.05,
                dfh=20,   # df1
                n2=5,    # total_N ??? what is this
                dfe2=28,  # df2
                cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                n_est=5,
                rank_est=2,
                alpha_cl=0.025,
                alpha_cu=0.025,
                tolerance=1e-12,
                omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_hlt_two_moment_null_approximator(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.hlt_two_moment_null_approximator(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
                alphatest=0.05,
                dfh=20,   # df1
                n2=5,    # total_N ??? what is this
                dfe2=28,  # df2
                cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                n_est=5,
                rank_est=2,
                alpha_cl=0.025,
                alpha_cu=0.025,
                tolerance=1e-12,
                omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_hlt_one_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.hlt_one_moment_null_approximator_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
                alphatest=0.05,
                dfh=20,   # df1
                n2=5,    # total_N ??? what is this
                dfe2=28,  # df2
                cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                n_est=5,
                rank_est=2,
                alpha_cl=0.025,
                alpha_cu=0.025,
                tolerance=1e-12,
                omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_hlt_two_moment_null_approximator_obrien_shieh(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.hlt_one_moment_null_approximator_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
                alphatest=0.05,
                dfh=20,   # df1
                n2=5,    # total_N ??? what is this
                dfe2=28,  # df2
                cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                n_est=5,
                rank_est=2,
                alpha_cl=0.025,
                alpha_cu=0.025,
                tolerance=1e-12,
                omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_pbt_one_moment_null_approx(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.pbt_one_moment_null_approx(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_pbt_two_moment_null_approx(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.pbt_two_moment_null_approx(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_pbt_one_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.pbt_one_moment_null_approx_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_pbt_two_moment_null_approx_obrien_shieh(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.pbt_two_moment_null_approx_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_wlk(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.wlk_two_moment_null_approx(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_wlk_os(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.wlk_two_moment_null_approx_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))


    def test_special(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = multirep.special(2, 1, 2, 5, eval_HINVE, 0.05)
        actual.glmmpcl(
            alphatest=0.05,
            dfh=20,  # df1
            n2=5,  # total_N ??? what is this
            dfe2=28,  # df2
            cl_type=Constants.CLTYPE_DESIRED_KNOWN,
            n_est=5,
            rank_est=2,
            alpha_cl=0.025,
            alpha_cu=0.025,
            tolerance=1e-12,
            omega=200)
        self.assertEqual(expected[0], np.round(actual.lower_bound.power, 5))
        self.assertEqual(expected[1], np.round(actual.power, 5))
        self.assertEqual(expected[1], np.round(actual.upper_bound.power, 4))

    def test_df1_rank_c_u(self):
        rank_C = 2
        rank_U = 3
        expected = 6
        actual = multirep._df1_rank_c_u(rank_C, rank_U)
        self.assertEqual(expected, actual)

    def test_multi_power(self):
        alpha = 0.9
        df1 = 3
        df2 = 4
        omega = 0.05

        expected = Power(1, 0.05, Constants.FMETHOD_NORMAL_LR)
        actual = multirep._multi_power(alpha, df1, df2, omega)
        self.assertEqual(expected.power, actual.power)
        self.assertEqual(expected.noncentrality_parameter, actual.noncentrality_parameter)
        self.assertEqual(expected.fmethod, actual.fmethod)

    def test_trace(self):
        eval_HINVE = [0.5]
        rank_X = 5
        total_N = 10
        expected = [0.025]
        actual = multirep._trace(eval_HINVE, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_calc_omega(self):
        min_rank_C_U = 1
        eval_HINVE = [0.5]
        rank_X = 5
        total_N = 10
        expected = [0.025]
        actual = multirep._calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_calc_hlt_omega(self):
        min_rank_C_U = 1
        eval_HINVE = [0.5]
        rank_X = 5
        total_N = 10
        expected = [0.025]
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
        rank_U = 2
        rank_X = 3
        total_N = 10
        exp_df1, exp_df2 = 1, 2
        actual_df1, actual_df2 = multirep._pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)
        self.assertEqual(exp_df1, actual_df1)
        self.assertEqual(exp_df2, actual_df2)

    def test_pbt_population_value(self):
        evalt = 1
        min_rank_C_U  =1
        expected = 1
        actual = multirep._pbt_population_value(evalt, min_rank_C_U)
        self.assertEqual(expected, actual)

    def test_pbt_uncorrected_evalt(self):
        eval_HINVE =[1]
        rank_C = 1
        rank_U = 2
        rank_X = 3
        total_N = 10
        expected = 1
        actual = multirep._pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N)
        self.assertEqual(expected, actual)

    def test_undefined_power(self):
        expected = Power(float('nan'), float('nan'), Constants.FMETHOD_MISSING)
        actual = multirep._undefined_power()
        self.assertTrue(np.math.isnan(actual.power))
        self.assertEqual(expected.noncentrality_parameter, actual.noncentrality_parameter)
        self.assertEqual(expected.fmethod, actual.fmethod)


