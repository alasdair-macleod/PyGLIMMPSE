from unittest import TestCase

from pyglimmpse.constants import Constants
from pyglimmpse.NonCentralityDistribution import NonCentralityDistribution
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import *


class TestUnirep(TestCase):

    def test_noncentralitydistribution(self):
        a = NonCentralityDistribution(
            test=Constants.HLT.value,
            FEssence=np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            FtFinverse=1.0,
            perGroupN=5,
            CFixed=np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]]),
            CRand=np.matrix([[1.0], [1.0]]),
            U=np.matrix([[1.0, 0.0],[0.0, 1.0]]),
            thetaNull=np.matrix([[0.0, 0.0], [0.0, 0.0]]),
            beta=np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]),
            sigmaError=np.matrix([[0.19, 0.0],[0.0, 1.0]]) - (np.matrix([[0.9], [0.0]]) * np.matrix([[0.9], [0]]).T),
            sigmaG=np.matrix([1.0]),
            sigmaStar= np.matrix([[0.19,0.0],[0.0,1.0]]),
            exact=False)
        print("Answer: ", a.cdf(14))
        print("H0", a.getH0())
        print("H1", a.getH1())