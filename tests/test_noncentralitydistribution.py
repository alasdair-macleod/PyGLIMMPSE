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
            thetaDiff=(np.concatenate((np.array(np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]])), np.array(np.matrix([[1.0], [1.0]]))), axis=1) * np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]) * np.matrix([[1.0, 0.0],[0.0, 1.0]])) -np.matrix([[0.0, 0.0], [0.0, 0.0]]),
            stddevG=np.matrix([1.0]),
            sigmaStar= np.matrix([[0.19,0.0],[0.0,1.0]]),
            exact=False)

        # U=np.matrix([[1.0, 0.0],[0.0, 1.0]]),
        # thetaNull=np.matrix([[0.0, 0.0], [0.0, 0.0]]),
        # beta=np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]),
        # sigmaError=np.matrix([[0.19, 0.0],[0.0, 1.0]]) - (np.matrix([[0.9], [0.0]]) * np.matrix([[0.9], [0]]).T),

        print("Answer: ", a.cdf(1))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(2))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(3))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(4))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(5))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(6))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(7))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(8))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(9))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(10))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(11))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(12))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(13))
        print("H0", a.getH0())
        print("H1", a.getH1())

        print("Answer: ", a.cdf(14))
        print("H0", a.getH0())
        print("H1", a.getH1())