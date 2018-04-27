from unittest import TestCase
import numpy as np
from pyglimmpse.multirep import *
from pyglimmpse.constants import Constants
from pyglimmpse.input import Scalar, CL

class TestMultirep(TestCase):
    def test_hlt_one_moment_null_approximator(self):
        """
        This should return the expected value
        """

        expected = (np.array([0.05421]), np.array([0.10994]), np.array([0.2395]))
        eval_HINVE = np.array([0.6])
        actual = hlt_one_moment_null_approximator(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = hlt_two_moment_null_approximator(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = hlt_one_moment_null_approximator_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = hlt_one_moment_null_approximator_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = pbt_one_moment_null_approx(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = pbt_two_moment_null_approx(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = pbt_one_moment_null_approx_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
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
        actual = pbt_two_moment_null_approx_obrien_shieh(2, 1, 2, 5, eval_HINVE, 0.05)
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
