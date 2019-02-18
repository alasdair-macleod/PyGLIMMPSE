from unittest import TestCase

from pyglimmpse.constants import Constants
from pyglimmpse.NonCentralityDistribution import NonCentralityDistribution
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import *


class TestUnirep(TestCase):

    def test_noncentralitydistribution(self):
        a = NonCentralityDistribution(
            test=Constants.HLT.value,
            FEssence=np.matrix([[1, 0],[0, 2]]),
            FtFinverse=1,
            perGroupN=10,
            CFixed=np.matrix([[1, -1, 0],[1, 0, -1]]),
            CRand=np.matrix([[1], [1]]),
            U=np.matrix([[1, 0],[0, 1]]),
            thetaNull=np.matrix([[0, 0], [0, 0]]),
            beta=np.matrix([[1, 0], [0, 0], [0, 0]]),
            sigmaError=np.matrix([[1, 0],[0, 1]]) - (np.matrix([[0.9], [0]]) * np.matrix([[0.9], [0]]).T),
            sigmaG=np.matrix([1]),
            exact=False)
        print(a.cdf(10))