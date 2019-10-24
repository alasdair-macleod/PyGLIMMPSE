from unittest import TestCase
import numpy as np

from pyglimmpse import samplesize
from pyglimmpse.constants import Constants
from pyglimmpse.unirep import uncorrected
from pyglimmpse.multirep import hlt_two_moment_null_approximator_obrien_shieh


class TestSamplesize(TestCase):

    def test_samplesize_hlt(self):
        c_matrix=np.matrix([[1]])
        m=np.matrix([[1]])
        sigma_star=np.matrix([[312.5]])
        theta=np.matrix([[52]])
        theta_zero=np.matrix([[42]])
        alpha = 0.01
        groups = [1]

        target_power = 0.9
        essence_design_matrix=np.matrix([[1]])
        test = hlt_two_moment_null_approximator_obrien_shieh
        delta = np.transpose(theta - theta_zero) * np.linalg.inv(m) * (theta-theta_zero)

        size, power = samplesize.samplesize(test=test,
                                            rank_C=np.linalg.matrix_rank(c_matrix),
                                            alpha=alpha,
                                            sigma_star=sigma_star,
                                            targetPower=target_power,
                                            rank_X=np.linalg.matrix_rank(essence_design_matrix),
                                            delta_es=delta,
                                            relative_group_sizes=groups,
                                            starting_smallest_group_size=10)
        self.assertEqual(50, size)
        self.assertEqual(round(power.power, 8), round(0.90101951, 8))

    def test_samplesize_hlt_multi_group(self):
        m=np.matrix([[1.16666667, 0.16666667], [0.16666667, 0.66666667]])
        t = np.matrix([[-5], [3.5]])
        delta=t.T * np.linalg.inv(m) * t
        sigma_star=np.matrix([[112.5]])
        alpha = 0.01
        groups = [6, 1,2]

        target_power = 0.9
        test = hlt_two_moment_null_approximator_obrien_shieh
        size, power = samplesize.samplesize(test=test,
                                            rank_C=2,
                                            alpha=alpha,
                                            sigma_star=sigma_star,
                                            targetPower=target_power,
                                            rank_X=3,
                                            delta_es=delta,
                                            relative_group_sizes=groups)

        self.assertEqual(round(0.90459334, 8), round(power.power, 8))
        self.assertEqual(369, size)


    # def test_samplesize_uncorrected(self):
    #     test =
    #     """
    #     :return:
    #     """
    #     test = uncorrected
    #     cmatrix=np.matrix([[1]])
    #     umatrix=np.matrix([[1]])
    #     essence_design_matrix=np.matrix([[1]])
    #     hypothesis_beta = np.matrix([[1]])
    #     hypothesis_sum_square = np.matrix([[1]])
    #     sigma_star = np.matrix([[625]])
    #     error_sum_square = np.matrix([[1]])
    #     rank_C = np.linalg.matrix_rank(cmatrix)
    #     rank_U = np.linalg.matrix_rank(umatrix)
    #     rank_X = np.linalg.matrix_rank(essence_design_matrix)
    #     expected = 20
    #     alpha = 0.01
    #     sigma_scale = 1
    #     beta_scale = 1
    #     target_power = 0.9
    #     args = {'approximation': 'uncorrected univariate approach to repeated measures',
    #             'epsilon_estimator': 'Muller, Edwards and Taylor (2004) approximation',
    #             'unirepmethod': Constants.SIGMA_KNOWN,
    #             'n_est': 33,
    #             'rank_est': 1,
    #             'alpha_cl': 0.025,
    #             'alpha_cu': 0.025,
    #             'n_ip': 33,
    #             'rank_ip': 1,
    #             'tolerance': 1e-10}
    #
    #     result = samplesize.samplesize(test=test,
    #                                    rank_C=rank_C,
    #                                    rank_U=rank_U,
    #                                    alpha=alpha,
    #                                    sigmaScale=sigma_scale,
    #                                    sigma_star=sigma_star,
    #                                    betaScale=beta_scale,
    #                                    beta=hypothesis_beta,
    #                                    targetPower=target_power,
    #                                    rank_X=rank_X,
    #                                    error_sum_square=error_sum_square,
    #                                    hypothesis_sum_square=hypothesis_sum_square,
    #                                    starting_sample_size=2,
    #                                    optional_args=args)
    #     self.assertTrue(target_power <= result[1])
    #     self.assertEqual(expected, result[0])
