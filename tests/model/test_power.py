from unittest import TestCase

from pyglimmpse.finv import finv

from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power


class TestGlmmpcl(TestCase):

    def test_glmmpcl(self):
        """
        This should correctly calculate the confidence intervals for power base on???
        """
        #todo where is this example from? whay are we rounding?
        expected = Power(0.9, 0.05, Constants.FMETHOD_NOAPPROXIMATION)
        dfe1 = 20 - 1
        alphatest = 0.05
        dfh = 20
        dfe2 = 28
        fcrit = finv(1 - alphatest, dfh, dfe2)
        actual = expected.glmmpcl(is_multirep=True,
                                  alphatest=0.05,
                                  dfh=20,  # df1
                                  n2=30,  # total_N ??? what is this
                                  dfe2=28,  # df2
                                  cl_type=Constants.CLTYPE_DESIRED_KNOWN,
                                  n_est=20,
                                  rank_est=1,
                                  alpha_cl=0.048,
                                  alpha_cu=0.052,
                                  fcrit=fcrit,
                                  tolerance=0.01,
                                  omega=200)

        power_l = 0.9999379
        noncen_l=105.66408
        power_u = 1
        noncen_u=315.62306
        fmethod = Constants.FMETHOD_NOAPPROXIMATION

        self.assertEqual(round(expected.lower_bound.power, 7), power_l)
        self.assertEqual(round(expected.lower_bound.noncentrality_parameter, 5), noncen_l)
        self.assertEqual(expected.lower_bound.fmethod, fmethod)
        self.assertEqual(expected.upper_bound.power, power_u)
        self.assertEqual(round(expected.upper_bound.noncentrality_parameter, 5), noncen_u)
        self.assertEqual(expected.upper_bound.fmethod, fmethod)
