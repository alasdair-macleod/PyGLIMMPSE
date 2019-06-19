from unittest import TestCase

from pyglimmpse.constants import Constants
from pyglimmpse.NonCentralityDistribution import NonCentralityDistribution
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import *
from pyglimmpse.probf import probf


class TestNoncentralityDist(TestCase):

    def test_noncentralitydistribution(self):
        a = NonCentralityDistribution(
            test=Constants.HLT.value,
            FEssence=np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            perGroupN=5,
            CFixed=np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]]),
            CGaussian=np.matrix([[1.0], [1.0]]),
            thetaDiff=(np.concatenate((np.array(np.matrix([[1.0, -1.0, 0.0],[1.0, 0.0, -1.0]])), np.array(np.matrix([[1.0], [1.0]]))), axis=1) * np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]]) * np.matrix([[1.0, 0.0],[0.0, 1.0]])) -np.matrix([[0.0, 0.0], [0.0, 0.0]]),
            stddevG=np.matrix([1.0]),
            sigmaStar= np.matrix([[0.19,0.0],[0.0,1.0]]),
            exact=False)
        # self.assertAlmostEqual(0.0, a.getH0(), 7)
        self.assertAlmostEqual(63.33333333333333, a.getH1(), 7)
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

    def test_noncentralitydistribution_2(self):

        Cf = np.matrix([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        Cg = np.matrix([[1.0], [1.0]])
        C = np.concatenate((Cf, Cg), axis=1)
        B = np.matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.9, 0.0]])
        U = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        Theta = C*B*U
        Theta0=np.matrix([[0.0, 0.0],[0.0, 0.0]])
        thetaDiff = Theta - Theta0
        FEssence = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        SigmaError= np.matrix([[0.19, 0.0], [0.0, 1.0]])
        sigmaStar = np.matrix([[0.19,0.0],[0.0,1.0]])
        stdevG = 1.0
        perGroupN = 5

        a = NonCentralityDistribution(
            test=Constants.HLT.value,
            FEssence=FEssence,
            perGroupN=perGroupN,
            CFixed=Cf,
            CGaussian=Cg,
            thetaDiff=thetaDiff,
            stddevG=stdevG,
            sigmaStar= sigmaStar,
            exact=False)

        self.assertAlmostEqual(30.07007667770580, a.inverseCDF(0.1))
        self.assertAlmostEqual(36.147160249831344, a.inverseCDF(0.2))
        self.assertAlmostEqual(40.66756344933491, a.inverseCDF(0.3))
        self.assertAlmostEqual(44.52591765312671, a.inverseCDF(0.4))
        self.assertAlmostEqual(48.04932724557611, a.inverseCDF(0.5))
        self.assertAlmostEqual(51.41016454949209, a.inverseCDF(0.6))
        self.assertAlmostEqual(54.72284790812153, a.inverseCDF(0.7))
        self.assertAlmostEqual(58.06316056229026, a.inverseCDF(0.8))
        self.assertAlmostEqual(61.361079235392985, a.inverseCDF(0.9))
        self.assertAlmostEqual(63.33333333333333, a.inverseCDF(1.0))

    def test_unconditional(self):

        Cf = np.matrix([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        Cg = np.matrix([[0.0], [0.0]])
        C = np.concatenate((Cf, Cg), axis=1)

        Bf = np.matrix([[1.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]])
        Bg = np.matrix([[0.5,0.5,0.5]])
        beta_scale = [0.4997025, 0.8075886, 1.097641]
        B = [np.concatenate((Bf*scale, Bg), axis=0) for scale in beta_scale]

        U = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0 ,0.0], [0.0, 0.0, 1.0]])

        Theta = [C*b*U for b in B]
        Theta0=np.matrix([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        thetaDiff = [t - Theta0 for t in Theta]

        FEssence = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        SigmaError= np.matrix([[0.75,-0.25,-0.25],[-0.25,0.75,-0.25],[-0.25,-0.25,0.75]])
        sigmaStar = U.T * SigmaError * U
        stdevG = 1.0

        perGroupN = 5
        tests = [Constants.UN, Constants.HF, Constants.GG, Constants.BOX]

        noncen_dists = []
        for test in tests:
            for theta_diff in thetaDiff:
                noncen_dists.append(NonCentralityDistribution(
                    test=test,
                    FEssence=FEssence,
                    perGroupN=perGroupN,
                    CFixed=Cf,
                    CGaussian=Cg,
                    thetaDiff=theta_diff,
                    stddevG=stdevG,
                    sigmaStar=sigmaStar,
                    exact=False))

        r = noncen_dists[3].unconditional_power_simpson(fcrit=2.15720777985222,
                                                        df1=7.29946278309409,
                                                        df2=37.93877551020408)

        print(r)

    def test_probf(self):
        print(probf(0.05, 0.5, 0.5, 10.0))

    def test_unconditional_power_simpson(self):
        """
        Test for model as described in .....
        :return:
        """
        Cf = np.matrix([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        Cg = np.matrix([[0.0], [0.0]])
        C = np.concatenate((Cf, Cg), axis=1)

        Bf = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        Bg = np.matrix([[0.5, 0.5, 0.5, 0.0]])
        beta_scale = [0.4997025, 0.8075886, 1.097641]
        B = [np.concatenate((Bf * scale, Bg), axis=0) for scale in beta_scale]

        U = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        Theta = [C * b * U for b in B]
        Theta0 = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        thetaDiff = [t - Theta0 for t in Theta]

        FEssence = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        SigmaError = np.matrix([[0.75, -0.25, -0.25, 0.0], [-0.25, 0.75, -0.25, 0.0], [-0.25, -0.25, 0.75, 0.0], [0, 0, 0, 1.0]])
        sigmaStar = U.T * SigmaError * U
        stdevG = 1.0

        perGroupN = 5
        tests = [Constants.HLT]

        noncen_dists = []
        for test in tests:
            for theta_diff in thetaDiff:
                noncen_dists.append(NonCentralityDistribution(
                    test=test,
                    FEssence=FEssence,
                    perGroupN=perGroupN,
                    CFixed=Cf,
                    CGaussian=Cg,
                    thetaDiff=theta_diff,
                    stddevG=stdevG,
                    sigmaStar=sigmaStar,
                    exact=False))

        fcrit = 3.1639338863418063
        df1 = 8.0
        df2 = 9.384615384615383

        # integral result 0.02625710980203323
        # result 0.8045605423465413

        # 0.5127311459742605
        # 0.21502204564774394

        expected = 0.8045605423465413
        a = noncen_dists[0].unconditional_power_simpson(fcrit, df1, df2)
        b = noncen_dists[1].unconditional_power_simpson(fcrit, df1, df2)
        c = noncen_dists[2].unconditional_power_simpson(fcrit, df1, df2)
        self.assertAlmostEqual(expected, a[0], 4)
        self.assertAlmostEqual(0.5127311459742605, b[0], 4)
        self.assertAlmostEqual(0.21502204564774394, c[0], 4)