import warnings
import numpy as np

from pyglimmpse.constants import Constants
from pyglimmpse.finv import finv
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.model.power import Power
from pyglimmpse.probf import probf
from scipy.stats import chi2

def uncorrected(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def hyuhn_feldt_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def hyuhn_feldt_muller_edwards_simpson_taylor_2004(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def chi_muller_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def chi_muller_muller_edwards_simpson_taylor_2004(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def geisser_greenhouse(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def box(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def _first_uni(sigma_star: np.matrix, rank_U: float) -> Epsilon:
    """
    This module produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.

    :param sigma_star: U` * (SIGMA # SIGSCALTEMP) * U
    :param rank_U: rank of U matrix

    :return:
        d, number of distinct eigenvalues
        mtp, multiplicities of eigenvalues
        eps, epsilon calculated from U`*SIGMA*U
        deigval, first eigenvalue
        slam1, sum of eigenvalues squared
        slam2, sum of squared eigenvalues
        slam3, sum of eigenvalues
    """

    #todo is this true for ALL epsilon? If so build into the class and remove this method.
    if rank_U != np.shape(sigma_star)[0]:
        raise Exception("rank of U should equal to nrows of sigma_star")

    # Get eigenvalues of covariance matrix associated with E. This is NOT
    # the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
    # The covariance matrix is normalized to minimize numerical problems
    epsilon = Epsilon(sigma_star, rank_U)
    return epsilon