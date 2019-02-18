from unittest import TestCase
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import *
from pyglimmpse.chisquareterm import *


class TestUnirep(TestCase):

    def test_geisser_greenhouse_muller_barton_1989(self):
        a = WeightedSumOfNoncentralChiSquaresDistribution(
            [ChiSquareTerm(7, 1, 10), ChiSquareTerm(-3, 2, 2), ChiSquareTerm(5, 1, 1)], 0.1, 0.001)
        print(a.cdf(10))
        print(a.cdf(20))
        print(a.cdf(30))
        print(a.cdf(40))
        print(a.cdf(50))
        print(a.cdf(60))
        print(a.cdf(70))
        print(a.cdf(80))
        print(a.cdf(90))
        print(a.cdf(100))