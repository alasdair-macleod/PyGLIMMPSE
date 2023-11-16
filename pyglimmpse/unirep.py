import math
import warnings
import inspect
import numpy as np

from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException
from pyglimmpse.finv import finv

from pyglimmpse.constants import Constants
from pyglimmpse.model.epsilon import Epsilon
from pyglimmpse.model.hypothesis_error import HypothesisError
from pyglimmpse.model.power import Power
from pyglimmpse.multirep import calc_properties, __calc_quantile_omega
from pyglimmpse.probf import probf

class OptionalArgs(object):
    """
    Class to hold optional args
    """
    def __init__(self):
        self.unirep_method = Constants.UN
        self.epsilon_estimator = Constants.EPSILON_MULLER2004
        self.sigma_source = Constants.SIGMA_KNOWN
        self.n_est = None
        self.rank_est = None
        self.alpha_cl = None
        self.alpha_cu = None
        self.n_ip = None
        self.rank_ip = None
        self.tolerance = 0.00000000000001

def uncorrected(rank_C: float,
                rank_X: float,
                relative_group_sizes,
                rep_N: float,
                alpha: float,
                sigma_star: np.matrix,
                delta_es: np.matrix,
                **kwargs) -> Power:
    return _unirep_power(epsilon_estimator=_uncorrected,
                         rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=rep_N,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         unirep_method=Constants.UN,
                         **kwargs)


def chi_muller(rank_C: float,
               rank_X: float,
               relative_group_sizes,
               rep_N: float,
               alpha: float,
               sigma_star: np.matrix,
               delta_es: np.matrix,
               **kwargs) -> Power:
    epsilon_estimator = _chi_muller_muller_edwards_simpson_taylor_2007
    if 'epsilon_estimator' in kwargs.keys() and kwargs['epsilon_estimator'] == Constants.EPSILON_MULLER1989:
        epsilon_estimator = _chi_muller_muller_barton_1989
    return _unirep_power(epsilon_estimator=epsilon_estimator,
                         rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=rep_N,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         unirep_method=Constants.CM,
                         **kwargs)


def geisser_greenhouse(rank_C: float,
                       rank_X: float,
                       relative_group_sizes,
                       rep_N: float,
                       alpha: float,
                       sigma_star: np.matrix,
                       delta_es: np.matrix,
                       **kwargs):
    epsilon_estimator = _geisser_greenhouse_muller_edwards_simpson_taylor_2007
    if 'epsilon_estimator' in kwargs.keys() and kwargs['epsilon_estimator'] == Constants.EPSILON_MULLER1989:
        epsilon_estimator = _geisser_greenhouse_muller_barton_1989
    return _unirep_power(epsilon_estimator=epsilon_estimator,
                         rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=rep_N,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         unirep_method=Constants.GG,
                         **kwargs)


def hyuhn_feldt(rank_C: float,
                rank_X: float,
                relative_group_sizes,
                rep_N: float,
                alpha: float,
                sigma_star: np.matrix,
                delta_es: np.matrix,
                **kwargs):
    epsilon_estimator = _hyuhn_feldt_muller_edwards_simpson_taylor_2007
    if 'epsilon_estimator' in kwargs.keys() and kwargs['epsilon_estimator'] == Constants.EPSILON_MULLER1989:
        epsilon_estimator = _hyuhn_feldt_muller_barton_1989
    return _unirep_power(epsilon_estimator=epsilon_estimator,
                         rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=rep_N,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         unirep_method=Constants.HF,
                         **kwargs)


def box(rank_C: float,
        rank_X: float,
        relative_group_sizes,
        rep_N: float,
        alpha: float,
        sigma_star: np.matrix,
        delta_es: np.matrix,
        **kwargs):
    return _unirep_power(epsilon_estimator=_box,
                         rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=rep_N,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         unirep_method=Constants.BOX,
                         **kwargs)

def _unirep_power(epsilon_estimator,
                  rank_C: float,
                  rank_X: float,
                  relative_group_sizes,
                  rep_N: float,
                  alpha: float,
                  sigma_star: np.matrix,
                  delta_es: np.matrix,
                  unirep_method,
                  **kwargs):
    error_sum_square, hypo_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    if len(inspect.signature(epsilon_estimator).parameters) == 0:
        expected_epsilon = epsilon_estimator()
    elif len(inspect.signature(epsilon_estimator).parameters) == 1:
        expected_epsilon = epsilon_estimator(rank_U=rank_U)
    else:
        expected_epsilon = epsilon_estimator(sigma_star=sigma_star, rank_U=rank_U, total_N=total_N, rank_X=rank_X)
    epsilon = _calc_epsilon(sigma_star, rank_U)
    power = Power(power='Not Calculable.')

    sigma_source = Constants.SIGMA_KNOWN

    confidence_interval = None
    if 'confidence_interval' in kwargs.keys():
        confidence_interval = kwargs['confidence_interval']
    if confidence_interval:
        sigma_source = Constants.SIGMA_ESTIMATED

    if sigma_source == Constants.SIGMA_KNOWN:
        power = _unirep_power_known_sigma(rank_C=rank_C,
                                          rank_U=rank_U,
                                          total_N=total_N,
                                          rank_X=rank_X,
                                          sigma_star=sigma_star,
                                          hypo_sum_square=hypo_sum_square,
                                          expected_epsilon=expected_epsilon,
                                          epsilon=epsilon.eps,
                                          alpha=alpha,
                                          unirep_method=unirep_method,
                                          **kwargs)

    if sigma_source == Constants.SIGMA_ESTIMATED:
        power = _unirep_power_estimated_sigma(rank_C=rank_C,
                                              rank_U=rank_U,
                                              total_N=total_N,
                                              rank_X=rank_X,
                                              sigma_star=sigma_star,
                                              hypo_sum_square=hypo_sum_square,
                                              expected_epsilon=expected_epsilon,
                                              epsilon=epsilon,
                                              alpha=alpha,
                                              unirep_method=unirep_method,
                                              **kwargs)

    if sigma_source == Constants.INTERNAL_PILOT:
        # get the eigenvalues of sigma_star using a singular value decomposition
        # sigmastareval is an array of dimension 1 x b
        sigmastareval = np.linalg.svd(sigma_star, full_matrices=False, compute_uv=False, hermitian=True)
        power = _unirep_power_known_sigma_internal_pilot(rank_C,
                                                         rank_U,
                                                         total_N,
                                                         rank_X,
                                                         sigma_star,
                                                         hypo_sum_square,
                                                         expected_epsilon,
                                                         epsilon,
                                                         alpha,
                                                         sigmastareval,
                                                         unirep_method,
                                                         **kwargs)
    return power


def _unirep_power_known_sigma(rank_C,
                              rank_U,
                              total_N,
                              rank_X,
                              sigma_star,
                              hypo_sum_square,
                              expected_epsilon,
                              epsilon,
                              alpha,
                              unirep_method,
                              **kwargs):
    """
    This function calculates power for univariate repeated measures power calculations with known Sigma.

    Parameters
    ----------
    rank_C: float
        rank of the C matrix
    rank_U: float
        rank of the U matrix
    total_N: float
        total number of observations
    rank_X:
        rank of the X matrix
    error_sum_square: np.matrix
        error sum of squares
    hypo_sum_square: np.matrix
        hypothesis sum of squares
    expected_epsilon: float
        expected value epsilon estimator
    epsilon:
        epsilon calculated from U`*SIGMA*U
    alpha:
        Significance level for target GLUM test
    unirep_method:
        Which method was used to find the expected value of epsilon.

        One of:

        * Uncorrected

        * geisser_greenhouse

        * chi_muller

        * hyuhn_feldt

        * box
    approximation_method:
        approximation used for cdf:

        * Muller and Barton (1989) approximation

        * Muller, Edwards and Taylor (2004) approximation


    Returns
    -------
    power: Power
        power for the univariate test.
    """
    # optional_args = __process_optional_args(**kwargs)
    approximation_method = Constants.UCDF_MULLER2004_APPROXIMATION
    noncentrality_dist = None
    quantile = None
    tolerance = 1e-12
    for key, value in kwargs.items():
        if key == 'approximation_method':
            approximation_method = value
        if key == 'noncentrality_distribution':
            noncentrality_dist = value
        if key == 'quantile':
            quantile = value
        if key == 'tolerance':
            tolerance = value

    nue = total_N - rank_X
    undf1, undf2 = _calc_undf1_undf2(unirep_method, expected_epsilon, nue, rank_C, rank_U)
    # Create defaults - same for either SIGMA known or estimated
    hypothesis_error = HypothesisError(hypo_sum_square, sigma_star, rank_U)
    e_1_2, e_3_5, e_4 = _calc_multipliers_known_sigma(epsilon, expected_epsilon, hypothesis_error, rank_C, rank_U, Constants.SIGMA_KNOWN)
    omega = e_3_5 * hypothesis_error.q2 / hypothesis_error.lambar
    # Error checking
    e_1_2 = _err_checking(e_1_2, rank_U)
    fcrit = finv(1 - alpha, undf1 * e_1_2, undf2 * e_1_2)

    if noncentrality_dist and quantile:
        omega = __calc_quantile_omega(noncentrality_dist, quantile)
        df1, df2, power = _calc_power_muller_approx(undf1, undf2, omega, alpha, e_3_5, e_4, fcrit)
    elif noncentrality_dist and not quantile:
        df1 = undf1 * e_3_5
        df2 = undf2 * e_4
        power = noncentrality_dist.unconditional_power_simpson(fcrit=fcrit, df1=df1, df2=df2)
    else:
        # 2. Muller, Edwards & Taylor 2002 and Muller Barton 1989 CDF approx
        # UCDFTEMP[]=4 reverts to UCDFTEMP[]=2 if exact CDF fails
        df1, df2, power = _calc_power_muller_approx(undf1, undf2, omega, alpha, e_3_5, e_4, fcrit)

    power = Power(power, omega, Constants.SIGMA_KNOWN)

    return power


def _get_cl_type(confidence_interval):
    cl_type = Constants.CLTYPE_DESIRED_KNOWN
    if not confidence_interval.beta_known:
        cl_type = Constants.CLTYPE_DESIRED_ESTIMATE
    return cl_type

def _unirep_power_estimated_sigma(rank_C,
                                  rank_U,
                                  total_N,
                                  rank_X,
                                  sigma_star,
                                  hypo_sum_square,
                                  expected_epsilon,
                                  epsilon,
                                  alpha,
                                  unirep_method,
                                  **kwargs):
    """
    This function calculates power for univariate repeated measures power calculations with known Sigma.

    Parameters
    ----------
    rank_C: float
        rank of the C matrix
    rank_U: float
        rank of the U matrix
    total_N: float
        total number of observations
    rank_X:
        rank of the X matrix
    error_sum_square: float
        error sum of squares
    hypo_sum_square: float
        hypothesis sum of squares
    expected_epsilon: float
        expected value epsilon estimator
    epsilon:
        epsilon calculated from U`*SIGMA*U
    alpha:
        Significance level for target GLUM test
    unirep_method:
        Which method was used to find the expected value of epsilon.

        One of:

        * Uncorrected

        * geisser_greenhouse

        * chi_muller

        * hyuhn_feldt

        * box
    approximation_method:
        approximation used for cdf:

        * Muller and Barton (1989) approximation

        * Muller, Edwards and Taylor (2004) approximation
    or the univariate test.
    n_est:
        total N from estimate study
    rank_est:
        rank of WHAT??? from estimate study
    alpha_cl:
        type one error (alpha) for lower confidence bound
    alpha_cu:
        type one error (alpha) for lower confidence bound
    tolerance:
        value below which, numbers are considered zero

    Returns
    -------
    power: Power
        power for the univariate test.
    """
    approximation_method = Constants.UCDF_MULLER2004_APPROXIMATION
    confidence_interval = None
    noncentrality_dist = None
    quantile = None
    tolerance = 1e-12
    for key, value in kwargs.items():
        if key == 'approximation_method':
            approximation_method = value
        if key == 'confidence_interval':
            confidence_interval = value
        if key == 'noncentrality_distribution':
            noncentrality_dist = value
        if key == 'quantile':
            quantile = value
        if key == 'tolerance':
            tolerance = value

    # optional_args = __process_optional_args(**kwargs)
    # E = SIGMASTAR # (N - rX)
    nue = total_N - rank_X
    undf1, undf2 = _calc_undf1_undf2(unirep_method, expected_epsilon, nue, rank_C, rank_U)
    # Create defaults - same for either SIGMA known or estimated
    hypothesis_error = HypothesisError(hypo_sum_square, sigma_star, rank_U)
    cl1df, e_1_2, e_3_5, e_4, omegaua = _calc_multipliers_est_sigma(unirep_method=unirep_method,
                                                                    eps=epsilon.eps,
                                                                    hypothesis_error=hypothesis_error,
                                                                    nue=nue,
                                                                    rank_C=rank_C,
                                                                    rank_U=rank_U,
                                                                    approximation_method=approximation_method,
                                                                    n_est=confidence_interval.n_est,
                                                                    rank_est=confidence_interval.rank_est)
    # Error checking
    e_1_2 = _err_checking(e_1_2, rank_U)
    omega = e_3_5 * hypothesis_error.q2 / hypothesis_error.lambar
    if unirep_method == Constants.CM:
        omega = omegaua
    fcrit = finv(1 - alpha, undf1 * e_1_2, undf2 * e_1_2)

    df1, df2, power = _calc_power_muller_approx(undf1, undf2, omega, alpha, e_3_5, e_4, fcrit)
    power = Power(power, omega, Constants.SIGMA_ESTIMATED)
    if confidence_interval:
        cl_type = _get_cl_type(confidence_interval)
        power.glmmpcl(is_multirep=False,
                      alphatest=alpha,
                      dfh=cl1df,
                      n2=total_N,
                      rank_est=confidence_interval.rank_est,
                      dfe2=df2,
                      cl_type=cl_type,
                      n_est=confidence_interval.n_est,
                      alpha_cl=confidence_interval.lower_tail,
                      alpha_cu=confidence_interval.upper_tail,
                      fcrit=fcrit,
                      tolerance=tolerance,
                      omega=omega,
                      df1_unirep=df1)

    return power


def _unirep_power_known_sigma_internal_pilot(rank_C,
                                             rank_U,
                                             total_N,
                                             rank_X,
                                             sigma_star,
                                             hypo_sum_square,
                                             expected_epsilon,
                                             epsilon,
                                             alpha,
                                             sigmastareval,
                                             unirep_method,
                                             **kwargs):
    """
    This function calculates power for univariate repeated measures power calculations with known Sigma.

    Parameters
    ----------
    rank_C: float
        rank of the C matrix
    rank_U: float
        rank of the U matrix
    total_N: float
        total number of observations
    rank_X:
        rank of the X matrix
    error_sum_square: float
        error sum of squares
    hypo_sum_square: float
        hypothesis sum of squares
    expected_epsilon: float
        expected value epsilon estimator
    epsilon:
        epsilon calculated from U`*SIGMA*U
    alpha:
        Significance level for target GLUM test
    unirep_method:
        Which method was used to find the expected value of epsilon.

        One of:

        * Uncorrected

        * geisser_greenhouse

        * chi_muller

        * hyuhn_feldt

        * box
    approximation_method:
        approximation used for cdf:

        * Muller and Barton (1989) approximation

        * Muller, Edwards and Taylor (2004) approximation
    or the univariate test.
    sigmastareval:
        eigenvalues  of SIGMASTAR=U`*SIGMA*U
    sigmastarevec:
        eigenvectors of SIGMASTAR=U`*SIGMA*U
    n_ip
        total N from internal pilot study
    rank_ip
        rank of WHAT??? in internal pilot

    Returns
    -------
    power: Power
        power for the univariate test.
    """
    approximation_method = Constants.UCDF_MULLER2004_APPROXIMATION
    internal_pilot = None
    noncentrality_dist = None
    quantile = None
    tolerance = 1e-12
    for key, value in kwargs.items():
        if key == 'approximation_method':
            approximation_method = value
        if key == 'internal_pilot':
            internal_pilot = value
        if key == 'noncentrality_distribution':
            noncentrality_dist = value
        if key == 'quantile':
            quantile = value
        if key == 'tolerance':
            tolerance = value
    # optional_args = __process_optional_args(**kwargs)
    # E = SIGMASTAR # (N - rX)
    nue = total_N - rank_X
    undf1, undf2 = _calc_undf1_undf2(unirep_method, expected_epsilon, nue, rank_C, rank_U)
    # Create defaults - same for either SIGMA known or estimated
    hypothesis_error = HypothesisError(hypo_sum_square, sigma_star, rank_U)
    e_1_2, e_3_5, e_4 = _calc_multipliers_internal_pilot(unirep_method, expected_epsilon, epsilon, hypothesis_error, sigmastareval, rank_C, rank_U, internal_pilot.n_ip, internal_pilot.rank_ip)

    # Error checking
    e_1_2 = _err_checking(e_1_2, rank_U)
    omega = e_3_5 * hypothesis_error.q2 / hypothesis_error.lambar
    fcrit = finv(1 - alpha, undf1 * e_1_2, undf2 * e_1_2)

    df1, df2, power = _calc_power_muller_approx(undf1, undf2, omega, alpha, e_3_5, e_4, fcrit)

    power = Power(power, omega, Constants.BOX)

    return power


def _uncorrected():
    expected_epsilon = 1
    return expected_epsilon


def _geisser_greenhouse_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes the approximate expected value of the Geisser-Greenhouse estimate using the approximator
    detailed in Muller and Barton 1989.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Chi-Muller test.
    """
    epsilon = _calc_epsilon(sigma_star, rank_U)
    f_i, f_ii = _gg_derivs_functions_eigenvalues(epsilon, rank_U)
    g_1 = _calc_g_1(epsilon, f_i, f_ii)
    expected_epsilon = epsilon.eps + g_1 / (total_N - rank_X)
    return expected_epsilon


def _geisser_greenhouse_muller_edwards_simpson_taylor_2007(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes the approximate expected value of the Geisser-Greenhouse estimate using the approximator
    detailed in Muller, Edwards, Simpson and Taylor 2007.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Chi-Muller test.
    """
    epsilon = _calc_epsilon(sigma_star, rank_U)

    nu = total_N - rank_X
    expt1 = 2 * nu * epsilon.slam2 + nu ** 2 * epsilon.slam1
    expt2 = nu * (nu + 1) * epsilon.slam2 + nu * epsilon.esigEvals()

    # Define GG Approx E(.) for Method 1
    expected_epsilon = (1 / rank_U) * (expt1 / expt2)
    return expected_epsilon


def _chi_muller_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes the approximate expected value of the Huynh-Feldt estimate with the Chi-Muller results via
    the approximate expected value of the Huynh-Feldt estimate using the approximator detailed in
    Muller and Barton 1989.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Chi-Muller test.
    """
    expected_epsilon_hf = _hyuhn_feldt_muller_barton_1989(
                    sigma_star=sigma_star,
                    rank_U=rank_U,
                    total_N=total_N,
                    rank_X=rank_X
    )

    expected_epsilon_cm = _calc_cm_expected_epsilon_estimator(expected_epsilon_hf, rank_X, total_N)

    return expected_epsilon_cm


def _chi_muller_muller_edwards_simpson_taylor_2007(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes the approximate expected value of the Huynh-Feldt estimate with the Chi-Muller results via
    the approximate expected value of the Huynh-Feldt estimate using the approximator detailed in
    Muller, Edwards, Simpson and Taylor 2007.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Chi-Muller test.
    """
    expected_epsilon = _hyuhn_feldt_muller_edwards_simpson_taylor_2007(
        sigma_star=sigma_star,
        rank_U=rank_U,
        total_N=total_N,
        rank_X=rank_X
    )

    expected_epsilon = _calc_cm_expected_epsilon_estimator(expected_epsilon, rank_X, total_N)

    return expected_epsilon


def _hyuhn_feldt_muller_barton_1989(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes power via the approximate expected value of the Huynh-Feldt estimate using the
    approximator detailed in Muller and Barton 1989.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Huyhn-Feldt test.
    """
    epsilon = _calc_epsilon(sigma_star, rank_U)

    # Compute approximate expected value of Huynh-Feldt estimate
    bh_i, bh_ii, h1, h2 = _hf_derivs_functions_eigenvalues(rank_U, rank_X, total_N, epsilon)
    g_1 = _calc_g_1(epsilon, bh_i, bh_ii)
    # Define HF Approx E(.) for Method 0
    expected_epsilon = h1 / (rank_U * h2) + g_1 / (total_N - rank_X)

    return expected_epsilon


def _hyuhn_feldt_muller_edwards_simpson_taylor_2007(sigma_star: np.matrix, rank_U: float, total_N: float, rank_X: float):
    """
    This function computes power via the approximate expected value of the Huynh-Feldt estimate using the
    approximator detailed in Muller, Edwards, Simpson and Taylor 2007

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U () matrix
    total_N: float
        total N, the sample size
    rank_X: float
        rank of X matrix (design/essence)

    Returns
    -------
    power: Power
        power as calculated by the Huyhn-Feldt test.
    """
    epsilon = _calc_epsilon(sigma_star, rank_U)
    # Computation of EXP(T1) and EXP(T2)
    nu = total_N - rank_X
    # for valid error degrees of freedom, nu must be strictly greater than 4
    if nu < 4:
        return np.nan
    expt1 = 2 * nu * epsilon.slam2 + nu ** 2 * epsilon.slam1
    expt2 = nu * (nu + 1) * epsilon.slam2 + nu * epsilon.esigEvals()
    num01 = (1 / rank_U) * ((nu + 1) * expt1 - 2 * expt2)
    den01 = nu * expt2 - expt1
    expected_epsilon = num01 / den01
    return expected_epsilon


def _box(rank_U: float):
    expected_epsilon = (1 / rank_U)
    return expected_epsilon


def _calc_epsilon(sigma_star: np.matrix, rank_U: float) -> Epsilon:
    """
    This module produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.

    Parameters
    ----------
    sigma_star: np.matrix
        The covariance matrix, :math:`\Sigma_*`,  defined as: :math:`\Sigma_* = U\'\Sigma U`
        This should be scaled in advance by multiplying :math:`\Sigma` by a constant SIGMASCALARTEMP
    rank_U: float
        rank of U matrix

    Returns
    -------
    epsilon
        :class:`.Epsilon` object containing the following
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
        raise GlimmpseValidationException("rank of U should be equal to the number of rows in sigma_star")

    # Get eigenvalues of covariance matrix associated with E. This is NOT
    # the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
    # The covariance matrix is normalized to minimize numerical problems
    epsilon = Epsilon(sigma_star, rank_U)
    return epsilon


def _calc_g_1(epsilon, f_i, f_ii):
    """
    This calculates :math:`g_1` as defined in Muller and Barton 1989

    .. math::
        g_1 = \sum_{i=1}^{d}f_ii\lambda_i^2m_i + \mathop{\sum \sum}_{i \\neq j} \dfrac {f_i\lambda_i \lambda_jm_im_j}{\lambda_i - \lambda_j}


    :math:`m_i, m_j` are the the multiplicities if the :math:`i^{th}, j^{th}` distinct eigenvalues.

    :math:`f_i` is :math:`\dfrac{\partial f}{\partial \lambda}` and :math:`f_{ii}` is :math:`\dfrac{\partial f^{(2)}}{\partial \lambda^{(2)}}`

    Parameters
    ----------
    epsilon: :class:`.Epsilon`
        The :class:`.Epsilon` object calculated for this test
    f_i: float
        the value of the first derivative of the function of the eigenvalues wrt :math:`\lambda`
    f_ii: float
        the value of the second derivative of the function of the eigenvalues wrt :math:`\lambda`

    Returns
    -------
    g_i: float
        the value of :math:`g_i` as defined above

    """

    t1 = np.multiply(np.multiply(f_ii, np.power(epsilon.deigval, 2)), epsilon.mtp)
    sum1 = np.sum(t1)
    if epsilon.d == 1:
        sum2 = 0
    else:
        t2 = np.multiply(np.multiply(f_i, epsilon.deigval), epsilon.mtp)
        t3 = np.multiply(epsilon.deigval, epsilon.mtp)
        tm1 = t2 * t3.T
        t4 = epsilon.deigval * np.full((1, epsilon.d), 1)
        tm2 = t4 - t4.T
        tm2inv = 1 / (tm2 + np.identity(epsilon.d)) - np.identity(epsilon.d)
        tm3 = np.multiply(tm1, tm2inv)
        sum2 = np.sum(tm3)
    g_1 = sum1 + sum2

    return g_1


def _hf_derivs_functions_eigenvalues(rank_U: float, rank_X: float, total_N: float, epsilon: Epsilon):
    """
    This function computes the derivatives of the functions of eigenvalues for the Huyhn_Feldt test. For HF, FNCT is epsilon tilde

    For Huyhn-Feldt:

    .. math::

        h(\lambda) = \dfrac{(Nb_\epsilon - 2)}{b(N - r -b_\epsilon)}  = \dfrac{h_1(\lambda)}{h_2(\lambda)b}

    with:

    .. math::
        h_1 = N(\Sigma\lambda_k)^2 - 2\Sigma\lambda_k^2

        h_2 = (N - r)\Sigma\lambda_k^2 - (\Sigma\lambda_k)^2

    In turn:

    .. math::
        \partial h_1 = 2N(\Sigma\lambda_k) - 4\lambda_i

        \partial h_2 = 2(N - r)\lambda_i - 2\Sigma\lambda_k

    and:

    .. math::

        \partial h_1^{(2)} = 2N - 4

        \partial h_2^{(2)} = 2(N - r) - 2

    The necessary derivatives are:

    .. math::

        bh_i = \dfrac{\partial h_1}{h_2} - \dfrac{h_1 \partial h_2}{h_2^2}

    and:

    .. math::

        bh_{ii} = \dfrac{\partial h_1^{(2)}}{h_2} -  \dfrac{ 2 \partial h_1\partial h_2}{h_2^2} + \dfrac{ 2h_1(\partial h_2)^2}{h_2^3} - \dfrac{h_1\partial h_2^{2}}{h_2^2}


    Parameters
    ----------
    rank_U: float
        rank of U matrix
    rank_X: float
        rank of X matrix
    total_N: float
        total N
    epsilon: :class:`.Epsilon`
        The :class:`.Epsilon` object calculated for this test

    Returns
    -------
    bh_i: float
        the value of the first derivative of the function of the eigenvalues wrt :math:`\lambda`
    bh_ii: float
        the value of the second derivative of the function of the eigenvalues wrt :math:`\lambda`
    h_1: float
        the value of h_1 as defined above
    h_2: float
        the value of h_2 as defined above

    """
    h1 = total_N * epsilon.slam1 - 2 * epsilon.slam2
    h2 = (total_N - rank_X) * epsilon.slam2 - epsilon.slam1
    derh1 = np.full((epsilon.d, 1), 2 * total_N * epsilon.slam3) - 4 * epsilon.deigval
    derh2 = 2 * (total_N - rank_X) * epsilon.deigval - np.full((epsilon.d, 1), 2 * np.sqrt(epsilon.slam1))
    bh_i = (derh1 - h1 * derh2 / h2) / (rank_U * h2)
    der2h1 = np.full((epsilon.d, 1), 2 * total_N - 4)
    der2h2 = np.full((epsilon.d, 1), 2 * (total_N - rank_X) - 2)
    bh_ii = (
           (np.multiply(-derh1, derh2) / h2 + der2h1 - np.multiply(derh1, derh2) / h2 + 2 * h1 * np.power(derh2, 2) / h2 ** 2 - h1 * der2h2 / h2)
           / (h2 * rank_U))
    return bh_i, bh_ii, h1, h2


def _gg_derivs_functions_eigenvalues(epsilon: Epsilon, rank_U: float):
    """
    This function computes the derivatives of the functions of eigenvalues for the Geisser-Greenhouse test.

    For Geisser-Greenhouse test :math:`f( \lambda) = \epsilon` so :math:`f_i` the first derivative, with respect to :math:`\lambda` is:

    .. math::

        f_i = \partial f = 2(\Sigma\lambda_k)(\Sigma\lambda_k^2)^{-1} - 2\lambda_i(\Sigma\lambda_k)^2(\Sigma\lambda_k^2)^{-2}b^{-1}

    and :math:`f_{ii}` the second derivative, with respect to :math:`\lambda` is:

    .. math::

        f_{ii} = \partial f^{(2)} = 2(\Sigma\lambda_k^2)^{-1}b^{-1} - 8\lambda_i(\Sigma\lambda_k)(\Sigma\lambda_k^2)^{-2}b^{-1} +  8\lambda_i^2(\Sigma\lambda_k)^2(\Sigma\lambda_k^2)^{-3}b^{-1} -2(\Sigma\lambda_k)^2(\Sigma\lambda_k^2)^{-2}b^{-1}

    Parameters
    ----------
    epsilon:class:`.Epsilon`
        The :class:`.Epsilon` object calculated for this test
    rank_U: float
        rank of U matrix

    Returns
    -------
    f_i: float
        the value of the first derivative of the function of the eigenvalues wrt :math:`\lambda`
    f_ii: float
        the value of the second derivative of the function of the eigenvalues wrt :math:`\lambda`

    """
    f_i = np.full((epsilon.d, 1), 1) * 2 * epsilon.slam3 / (epsilon.slam2 * rank_U) \
         - 2 * epsilon.deigval * epsilon.slam1 / (rank_U * epsilon.slam2 ** 2)
    c0 = 1 - epsilon.slam1 / epsilon.slam2
    c1 = -4 * epsilon.slam3 / epsilon.slam2
    c2 = 4 * epsilon.slam1 / epsilon.slam2 ** 2
    f_ii = 2 * (c0 * np.full((epsilon.d, 1), 1)
               + c1 * epsilon.deigval
               + c2 * np.power(epsilon.deigval, 2)) / (rank_U * epsilon.slam2)
    return f_i, f_ii


def _calc_cm_expected_epsilon_estimator(exeps, rank_X, total_N):
    """
    This function computes the approximate expected value of the Chi-Muller estimate.

    Parameters
    ----------
    exeps: float
        the expected value of the epsilon estimator calculated using the approximate expected value of
        the Huynh-Feldt estimate
    rank_X: float
        rank of X matrix
    total_N: float
        total N

    Return
    ------
    epsilon:class:`pyglimmpse.model.Epsilon`
        The :class:`.Epsilon` object calculated for this test
    """
    if total_N - rank_X == 1:
        uefactor = 1
    else:
        nu_e = total_N - rank_X
        nu_a = (nu_e - 1) + nu_e * (nu_e - 1) / 2
        uefactor = (nu_a - 2) * (nu_a - 4) / (nu_a ** 2)
    exeps = uefactor * exeps
    return exeps


def _calc_multipliers_known_sigma(eps, exeps, hypothesis_error, rank_C, rank_U, approximation_method):

    epsn_num = hypothesis_error.q3 + hypothesis_error.q1 * hypothesis_error.q2 * 2 / rank_C
    epsn_den = hypothesis_error.q4 + hypothesis_error.q5 * 2 / rank_C
    epsn = epsn_num / (rank_U * epsn_den)
    e_1_2 = exeps
    e_4 = eps
    if approximation_method == Constants.UCDF_MULLER1989_APPROXIMATION:
        e_3_5 = eps
    else:
        e_3_5 = epsn

    # Obtain noncentrality and critical value for power point estimate
    return e_1_2, e_3_5, e_4


def _calc_multipliers_est_sigma(unirep_method, eps, hypothesis_error, nue, rank_C, rank_U, approximation_method, n_est, rank_est):
    # Case 2
    # Enter loop to compute E1-E5 based on estimated SIGMA
    nu_est = n_est - rank_est
    if nu_est <= 1:
        raise GlimmpseValidationException("ERROR 81: Too few estimation df in LASTUNI. df = N_EST - RANK_EST <= 1.")
    # For POWERCALC =6=HF, =7=CM, =8=GG critical values
    epstilde_r = ((nu_est + 1) * hypothesis_error.q3 - 2 * hypothesis_error.q4) / (rank_U * (nu_est * hypothesis_error.q4 - hypothesis_error.q3))
    epstilde_r_min = min(epstilde_r, 1)
    mult = np.power(nu_est, 2) + nu_est - 2
    epsnhat_num = hypothesis_error.q3 * nu_est * (nu_est + 1) + hypothesis_error.q1 * hypothesis_error.q2 * 2 * mult / rank_C - hypothesis_error.q4 * 2 * nu_est
    epsnhat_den = hypothesis_error.q4 * nu_est * nu_est + hypothesis_error.q5 * 2 * mult / rank_C - hypothesis_error.q3 * nu_est
    epsnhat = epsnhat_num / (rank_U * epsnhat_den)
    nua0 = (nu_est - 1) + nu_est * (nu_est - 1) / 2
    tau10 = nu_est * ((nu_est + 1) * hypothesis_error.q1 * hypothesis_error.q1 - 2 * hypothesis_error.q4) / (nu_est * nu_est + nu_est - 2)
    tau20 = nu_est * (nu_est * hypothesis_error.q4 - hypothesis_error.q1 * hypothesis_error.q1) / (nu_est * nu_est + nu_est - 2)
    epsda = tau10 * (nua0 - 2) * (nua0 - 4) / (rank_U * nua0 * nua0 * tau20)
    epsda = max(min(epsda, 1), 1 / rank_U)
    epsna = (1 + 2 * (hypothesis_error.q2 / rank_C) / hypothesis_error.q1) / (1 / epsda + 2 * rank_U * (hypothesis_error.q5 / rank_C) / (hypothesis_error.q1 * hypothesis_error.q1))
    omegaua = hypothesis_error.q2 * epsna * (rank_U / hypothesis_error.q1)
    # Set E_1_2 for all tests
    # for UN or Box critical values
    if unirep_method == Constants.UN or unirep_method == Constants.BOX:
        e_1_2 = epsda

    # for HF crit val
    if unirep_method == Constants.HF:
        if rank_U <= nue:
            e_1_2 = epstilde_r_min
        else:
            e_1_2 = epsda

    # for CM crit val
    if unirep_method == Constants.CM:
        e_1_2 = epsda

    # for GG crit val
    if unirep_method == Constants.GG:
        e_1_2 = eps

    # Set E_3_5 for all tests
    if approximation_method == Constants.UCDF_MULLER1989_APPROXIMATION:
        e_3_5 = eps
    else:
        e_3_5 = epsnhat

    # Set E_4 for all tests
    if unirep_method == Constants.CM:
        e_4 = epsda
    else:
        e_4 = eps

    # Compute DF for confidence limits for all tests
    cl1df = rank_U * nu_est * e_4 / e_3_5
    return cl1df, e_1_2, e_3_5, e_4, omegaua


def _calc_multipliers_internal_pilot(unirep_method, exeps, eps, hypothesis_error, sigmastareval, rank_C, rank_U, n_ip, rank_ip):
    nu_ip = n_ip - rank_ip
    e_1_2 = exeps
    e_4 = eps

    if unirep_method == Constants.HF or unirep_method == Constants.CM or unirep_method == Constants.GG:
        lambdap = np.concatenate((sigmastareval,
                                  np.power(sigmastareval, 2),
                                  np.power(sigmastareval, 3),
                                  np.power(sigmastareval, 4)), axis=1)
        sumlam = np.matrix(np.sum(lambdap, axis=0)).T
        kappa = np.multiply(np.multiply(np.matrix([[1], [2], [8], [48]]), nu_ip), sumlam)
        muprime2 = np.asscalar(kappa[1] + np.power(kappa[0], 2))
        meanq2 = np.asscalar(np.multiply(np.multiply(nu_ip, nu_ip + 1), sumlam[1]) + np.multiply(nu_ip, np.sum(
            sigmastareval * sigmastareval.T)))

        et1 = muprime2 / np.power(nu_ip, 2)
        et2 = meanq2 / np.power(nu_ip, 2)
        ae_epsn_up = et1 + 2 * hypothesis_error.q1 * hypothesis_error.q2
        ae_epsn_dn = rank_U * (et2 + 2 * hypothesis_error.q5)
        aex_epsn = ae_epsn_up / ae_epsn_dn
        e_3_5 = aex_epsn
    else:
        epsn_num = hypothesis_error.q3 + hypothesis_error.q1 * hypothesis_error.q2 * 2 / rank_C
        epsn_den = hypothesis_error.q4 + hypothesis_error.q5 * 2 / rank_C
        epsn = epsn_num / (rank_U * epsn_den)
        e_3_5 = epsn
    return e_1_2, e_3_5, e_4


def _calc_power_muller_approx(undf1, undf2, omega, alpha, e_3_5, e_4, fcrit):
    df1 = undf1 * e_3_5
    df2 = undf2 * e_4
    prob, fmethod = probf(fcrit, df1, df2, omega)
    if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
        power = alpha
    else:
        power = 1 - prob

    return df1, df2, power


def _calc_undf1_undf2(unirep_method, exeps, nue, rank_C, rank_U):
    if rank_U > nue and (unirep_method == Constants.UN or unirep_method == Constants.GG or unirep_method == Constants.BOX):
        warnings.warn('Power is missing, because Uncorrected, Geisser-Greenhouse and Box tests are '
                      'poorly behaved (super low power and test size) when B > N-R, i.e., HDLSS.')
        '''During the sample size searching process, the smaller sample size can raise this error but we dont want to 
        stop searching '''
        # raise GlimmpseValidationException('Power is missing, because Uncorrected, Geisser-Greenhouse and Box tests are'
        #               'poorly behaved (super low power and test size) when B > N-R, i.e., HDLSS.')
    if np.isnan(exeps) or nue <= 0:
        raise GlimmpseValidationException(Constants.ERR_ERROR_DEG_FREEDOM.value)
    undf1 = rank_C * rank_U
    undf2 = rank_U * nue
    return undf1, undf2


def _err_checking(e_1_2, rank_U):
    if e_1_2 < 1 / rank_U:
        e_1_2 = 1 / rank_U
        warnings.warn('PowerWarn17: The approximate expected value of estimated epsilon was truncated up to 1/B.')
    if e_1_2 > 1:
        e_1_2 = 1
        warnings.warn('PowerWarn18: The approximate expected value of estimated epsilon was truncated down to 1.')

    return e_1_2

