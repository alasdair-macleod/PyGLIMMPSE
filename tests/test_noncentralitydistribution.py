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

        print("H0", a.getH0())
        self.assertAlmostEqual(0.0, a.getH0(), 7)
        print("H1", a.getH1())
        self.assertAlmostEqual(63.33333333333333, a.getH1(), 7)
        print("Answer: ", a.cdf(1))
        print("Answer: ", a.cdf(2))
        print("Answer: ", a.cdf(3))
        print("Answer: ", a.cdf(4))
        print("Answer: ", a.cdf(5))
        print("Answer: ", a.cdf(6))
        print("Answer: ", a.cdf(7))
        print("Answer: ", a.cdf(8))
        print("Answer: ", a.cdf(9))
        print("Answer: ", a.cdf(10))
        print("Answer: ", a.cdf(11))
        print("Answer: ", a.cdf(12))
        print("Answer: ", a.cdf(13))
        print("Answer: ", a.cdf(14))
        self.assertAlmostEqual(0.0000002106, a.cdf(1),  4)
        self.assertAlmostEqual(0.0000003420, a.cdf(2),  4)
        self.assertAlmostEqual(0.0000014004, a.cdf(3),  4)
        self.assertAlmostEqual(0.0000057429, a.cdf(4),  4)
        self.assertAlmostEqual(0.0000182151, a.cdf(5),  4)
        self.assertAlmostEqual(0.0000470019, a.cdf(6),  4)
        self.assertAlmostEqual(0.0001043387, a.cdf(7),  4)
        self.assertAlmostEqual(0.0002070572, a.cdf(8),  4)
        self.assertAlmostEqual(0.0003769546, a.cdf(9),  4)
        self.assertAlmostEqual(0.0006409878, a.cdf(10), 4)
        self.assertAlmostEqual(0.0010313008, a.cdf(11), 4)
        self.assertAlmostEqual(0.0015850977, a.cdf(12), 4)
        self.assertAlmostEqual(0.0023443774, a.cdf(13), 4)
        self.assertAlmostEqual(0.0033555463, a.cdf(14), 4)