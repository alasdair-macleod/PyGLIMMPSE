from unittest import TestCase

from pyglimmpse.constants import Constants
from pyglimmpse.NonCentralityDistribution import NonCentralityDistribution
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import *
from pyglimmpse.probf import probf


class TestUnirep(TestCase):

    def test_noncentralitydistribution(self):
        a = NonCentralityDistribution(
            test=Constants.HLT.value,
            FEssence=np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            FtFinverse=np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T*np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            perGroupN=5,
            CFixed=np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]]),
            CRand=np.matrix([[1.0], [1.0]]),
            thetaDiff=(np.concatenate((np.array(np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]])), np.array(np.matrix([[1.0], [1.0]]))), axis=1) * np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]) * np.matrix([[1.0, 0.0],[0.0, 1.0]])) -np.matrix([[0.0, 0.0], [0.0, 0.0]]),
            stddevG=np.matrix([1.0]),
            sigmaStar= np.matrix([[0.19,0.0],[0.0,1.0]]),
            exact=False)
        print("ThetaDiff", (np.concatenate((np.array(np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]])), np.array(np.matrix([[1.0], [1.0]]))), axis=1) * np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]) * np.matrix([[1.0, 0.0],[0.0, 1.0]])) -np.matrix([[0.0, 0.0], [0.0, 0.0]]))
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
        self.assertAlmostEqual(0.0000002106, a.cdf(1),  7)
        self.assertAlmostEqual(0.0000003420, a.cdf(2),  7)
        self.assertAlmostEqual(0.0000014004, a.cdf(3),  7)
        self.assertAlmostEqual(0.0000057429, a.cdf(4),  7)
        self.assertAlmostEqual(0.0000182151, a.cdf(5),  7)
        self.assertAlmostEqual(0.0000470019, a.cdf(6),  7)
        self.assertAlmostEqual(0.0001043387, a.cdf(7),  7)
        self.assertAlmostEqual(0.0002070572, a.cdf(8),  7)
        self.assertAlmostEqual(0.0003769546, a.cdf(9),  7)
        self.assertAlmostEqual(0.0006409878, a.cdf(10), 7)
        self.assertAlmostEqual(0.0010313008, a.cdf(11), 7)
        self.assertAlmostEqual(0.0015850977, a.cdf(12), 7)
        self.assertAlmostEqual(0.0023443774, a.cdf(13), 7)
        self.assertAlmostEqual(0.0033555463, a.cdf(14), 7)

    def test_probf(self):
        print(probf(0.05, 0.5, 0.5, 10.0))