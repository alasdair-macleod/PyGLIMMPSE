from unittest import TestCase
import numpy as np

from pyglimmpse import samplesize
from pyglimmpse.constants import Constants
from pyglimmpse.unirep import uncorrected


class TestSamplesize(TestCase):

    def test_samplesize_uncorrected(self):
        """
        :return:
        """
        test = uncorrected
        cmatrix=np.matrix([[1]])
        umatrix=np.matrix([[1]])
        essence_design_matrix=np.matrix([[1]])
        hypothesis_beta = np.matrix([[1]])
        hypothesis_sum_square = np.matrix([[1]])
        sigma_star = np.matrix([[625]])
        error_sum_square = np.matrix([[1]])
        rank_C = np.linalg.matrix_rank(cmatrix)
        rank_U = np.linalg.matrix_rank(umatrix)
        rank_X = np.linalg.matrix_rank(essence_design_matrix)
        expected = 17
        alpha = 0.01
        sigma_scale = 1
        beta_scale = 1
        target_power = 0.9
        args = {'approximation': 'uncorrected univariate approach to repeated measures',
                'epsilon_estimator': 'Muller, Edwards and Taylor (2004) approximation',
                'unirepmethod': Constants.SIGMA_KNOWN,
                'n_est': 33,
                'rank_est': 1,
                'alpha_cl': 0.025,
                'alpha_cu': 0.025,
                'n_ip': 33,
                'rank_ip': 1,
                'tolerance': 1e-10}

        size = samplesize.samplesize(test=test,
                                     rank_C=rank_C,
                                     rank_U=rank_U,
                                     alpha=alpha,
                                     sigmaScale=sigma_scale,
                                     sigma=sigma_star,
                                     betaScale=beta_scale,
                                     beta=hypothesis_beta,
                                     targetPower=target_power,
                                     rank_X=rank_X,
                                     error_sum_square=error_sum_square,
                                     hypothesis_sum_square=hypothesis_sum_square,
                                     starting_sample_size=2,
                                     optional_args=args)
        self.assertEqual(expected, size)