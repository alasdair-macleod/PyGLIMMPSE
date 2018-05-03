import warnings
import numpy as np

from pyglimmpse.constants import Constants
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.model.power import Power

def uncorrected(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass


def geisser_greenhouse_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    epsilon = _calc_epsilon(sigma_star, rank_U)
    fk, fkk = _calc_gg_fk_fkk(epsilon, rank_U)
    sum1, sum2, t1, t2 = _calc_sum1_sum2(epsilon, fk, fkk)
    exeps = epsilon.eps + (sum1 + sum2) / (total_N - rank_X)
    return exeps


def geisser_greenhouse_muller_edwards_simpson_taylor_2004(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    epsilon = _calc_epsilon(sigma_star, rank_U)

    nu = total_N - rank_X
    expt1 = 2 * nu * epsilon.slam2 + nu ** 2 * epsilon.slam1
    expt2 = nu * (nu + 1) * epsilon.slam2 + nu * epsilon.nameME()

    # Define GG Approx E(.) for Method 1
    exeps = (1 / rank_U) * (expt1 / expt2)
    return exeps


def chi_muller_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    """
        Univariate, HF STEP 2 with Chi-Muller:
        This function computes the approximate expected value of
        the Huynh-Feldt estimate with the Chi-Muller results

        :param sigma_star:
        :param rank_U:
        :param total_N:
        :param rank_X:
        :return:
        """
    exeps = hyuhn_feldt_muller_barton_1989(
                    sigma_star=sigma_star,
                    rank_U=rank_U,
                    total_N=total_N,
                    rank_X=rank_X
    )

    exeps = _calc_cm_exeps(exeps, rank_X, total_N)

    return exeps


def chi_muller_muller_edwards_simpson_taylor_2004(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    """
            Univariate, HF STEP 2 with Chi-Muller:
            This function computes the approximate expected value of
            the Huynh-Feldt estimate with the Chi-Muller results

            :param sigma_star:
            :param rank_U:
            :param total_N:
            :param rank_X:
            :return:
            """
    exeps = hyuhn_feldt_muller_edwards_simpson_taylor_2004(
        sigma_star=sigma_star,
        rank_U=rank_U,
        total_N=total_N,
        rank_X=rank_X
    )

    exeps = _calc_cm_exeps(exeps, rank_X, total_N)

    return exeps

def hyuhn_feldt_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    """

        Univariate, HF STEP 2:
        This function computes the approximate expected value of
        the Huynh-Feldt estimate.

          FK  = 1st deriv of FNCT of eigenvalues
          FKK = 2nd deriv of FNCT of eigenvalues
          For HF, FNCT is epsilon tilde

        :param sigma_star:
        :param rank_U:
        :param total_N:
        :param rank_X:
        :param UnirepUncorrected:
        :return:
        """
    epsilon = _calc_epsilon(sigma_star, rank_U)

    # Compute approximate expected value of Huynh-Feldt estimate
    fk, fkk, h1, h2 = _calc_hf_fk_fkk(rank_U, rank_X, total_N, epsilon)
    sum1, sum2, t1, t2 = _calc_sum1_sum2(epsilon, fk, fkk)
    # Define HF Approx E(.) for Method 0
    exeps = h1 / (rank_U * h2) + (sum1 + sum2) / (total_N - rank_X)

    return exeps


def hyuhn_feldt_muller_edwards_simpson_taylor_2004(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    epsilon = _calc_epsilon(sigma_star, rank_U)
    # Computation of EXP(T1) and EXP(T2)
    nu = total_N - rank_X
    expt1 = 2 * nu * epsilon.slam2 + nu ** 2 * epsilon.slam1
    expt2 = nu * (nu + 1) * epsilon.slam2 + nu * epsilon.nameME()
    num01 = (1 / rank_U) * ((nu + 1) * expt1 - 2 * expt2)
    den01 = nu * expt2 - expt1
    exeps = num01 / den01
    return exeps


def box(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float) -> Power:
    pass
def _calc_epsilon(sigma_star: np.matrix, rank_U: float) -> Epsilon:
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


def _calc_sum1_sum2(epsilon, fk, fkk):
    t1 = np.multiply(np.multiply(fkk, np.power(epsilon.deigval, 2)), epsilon.mtp)
    sum1 = np.sum(t1)
    if epsilon.d == 1:
        sum2 = 0
    else:
        t2 = np.multiply(np.multiply(fk, epsilon.deigval), epsilon.mtp)
        t3 = np.multiply(epsilon.deigval, epsilon.mtp)
        tm1 = t2 * t3.T
        t4 = epsilon.deigval * np.full((1, epsilon.d), 1)
        tm2 = t4 - t4.T
        tm2inv = 1 / (tm2 + np.identity(d)) - np.identity(d)
        tm3 = np.multiply(tm1, tm2inv)
        sum2 = np.sum(tm3)

    return sum1, sum2, t1, t2


def _calc_hf_fk_fkk(rank_U, rank_X, total_N, epsilon):
    h1 = total_N * epsilon.slam1 - 2 * epsilon.slam2
    h2 = (total_N - rank_X) * epsilon.slam2 - epsilon.slam1
    derh1 = np.full((epsilon.d, 1), 2 * total_N * epsilon.slam3) - 4 * epsilon.deigval
    derh2 = 2 * (total_N - rank_X) * epsilon.deigval - np.full((epsilon.d, 1), 2 * np.sqrt(epsilon.slam1))
    fk = (derh1 - h1 * derh2 / h2) / (rank_U * h2)
    der2h1 = np.full((epsilon.d, 1), 2 * total_N - 4)
    der2h2 = np.full((epsilon.d, 1), 2 * (total_N - rank_X) - 2)
    fkk = (np.multiply(-derh1, derh2) / h2 +
           der2h1 - np.multiply(derh1, derh2) / h2 +
           2 * h1 * np.power(derh2, 2) / h2 ** 2
           - h1 * der2h2 / h2) / (h2 * rank_U)
    return fk, fkk, h1, h2


def _calc_gg_fk_fkk(epsilon, rank_U):
    fk = np.full((epsilon.d, 1), 1) * 2 * epsilon.slam3 / (epsilon.slam2 * rank_U) \
         - 2 * epsilon.deigval * epsilon.slam1 / (rank_U * epsilon.slam2 ** 2)
    c0 = 1 - epsilon.slam1 / epsilon.slam2
    c1 = -4 * epsilon.slam3 / epsilon.slam2
    c2 = 4 * epsilon.slam1 / epsilon.slam2 ** 2
    fkk = 2 * (c0 * np.full((epsilon.d, 1), 1)
               + c1 * epsilon.deigval
               + c2 * np.power(epsilon.deigval, 2)) / (rank_U * epsilon.slam2)
    return fk, fkk


def _calc_cm_exeps(exeps, rank_X, total_N):
    if total_N - rank_X == 1:
        uefactor = 1
    else:
        nu_e = total_N - rank_X
        nu_a = (nu_e - 1) + nu_e * (nu_e - 1) / 2
        uefactor = (nu_a - 2) * (nu_a - 4) / (nu_a ** 2)
    exeps = uefactor * exeps
    return exeps
