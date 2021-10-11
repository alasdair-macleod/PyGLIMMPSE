import warnings

import numpy as np
from scipy import linalg

from pyglimmpse.constants import Constants
from pyglimmpse.finv import finv
from pyglimmpse.model.power import Power
from pyglimmpse.probf import probf


def hlt_one_moment_null_approximator(rank_C: float,
                                     rank_X: float,
                                     relative_group_sizes,
                                     rep_N: float,
                                     alpha: float,
                                     sigma_star: np.matrix,
                                     delta_es: np.matrix,
                                     tolerance=1e-12,
                                     **kwargs) -> Power:
    """
    This function calculates power for Hotelling-Lawley trace
    based on the Pillai F approximation. HLT is the "population value"
    Hotelling Lawley trace. F1 and DF2 are the hypothesis and
    error degrees of freedom, OMEGA is the non-centrality parameter, and
    FCRIT is the critical value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """
    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 1  Pillai (1954, 55) 1 moment null approx
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)

    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _hlt_one_moment_df2(min_rank_C_U, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_hlt_omega(min_rank_C_U, eval_HINVE, rank_X, total_N, df2)
        return _multi_power(alpha, df1, df2, omega, total_N)
    return _undefined_power()


def hlt_two_moment_null_approximator(rank_C: float,
                                     rank_X: float,
                                     relative_group_sizes,
                                     rep_N: float,
                                     alpha: float,
                                     sigma_star: np.matrix,
                                     delta_es: np.matrix,
                                     tolerance=1e-12,
                                     **kwargs) -> Power:
    """
    This function calculates power for Hotelling-Lawley trace
    based on the Pillai F approximation. HLT is the "population value"
    Hotelling Lawley trace. F1 and DF2 are the hypothesis and
    error degrees of freedom, OMEGA is the non-centrality parameter, and
    FCRIT is the critical value from the F distribution.


    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """  # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 2  McKeon (1974) two moment null approx
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)

    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_hlt_omega(min_rank_C_U, eval_HINVE, rank_X, total_N, df2)
        return _multi_power(alpha, df1, df2, omega, total_N)
    else:
        return _undefined_power()


def hlt_one_moment_null_approximator_obrien_shieh(rank_C: float,
                                                  rank_X: float,
                                                  relative_group_sizes,
                                                  rep_N: float,
                                                  alpha: float,
                                                  sigma_star: np.matrix,
                                                  delta_es: np.matrix,
                                                  tolerance=1e-12,
                                                  **kwargs) -> Power:
    """
    This function calculates power for Hotelling-Lawley trace
    based on the Pillai F approximation. HLT is the "population value"
    Hotelling Lawley trace. F1 and DF2 are the hypothesis and
    error degrees of freedom, OMEGA is the non-centrality parameter, and
    FCRIT is the critical value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 3  Pillai (1959) one moment null approx+ OS noncen mult
    df2 = _hlt_one_moment_df2(min_rank_C_U, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    # df2 need to > 0 and eigenvalues not missing
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        return _multi_power(alpha, df1, df2, omega, total_N)
    else:
        return _undefined_power()


def hlt_two_moment_null_approximator_obrien_shieh(rank_C: float,
                                                  rank_X: float,
                                                  relative_group_sizes,
                                                  rep_N: float,
                                                  alpha: float,
                                                  sigma_star: np.matrix,
                                                  delta_es: np.matrix,
                                                  tolerance=1e-12,
                                                  **kwargs) -> Power:
    """
    This function calculates power for Hotelling-Lawley trace
    based on the Pillai F approximation. HLT is the "population value"
    Hotelling Lawley trace. F1 and DF2 are the hypothesis and
    error degrees of freedom, OMEGA is the non-centrality parameter, and
    FCRIT is the critical value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 4  McKeon (1974) two moment null approx+ OS noncen mult
    df2 = _hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    # df2 need to > 0 and eigenvalues not missing
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        return _multi_power(alpha, df1, df2, omega, total_N, **kwargs)
    else:
        return _undefined_power()


def pbt_one_moment_null_approx(rank_C: float,
                               rank_X: float,
                               relative_group_sizes,
                               rep_N: float,
                               alpha: float,
                               sigma_star: np.matrix,
                               delta_es: np.matrix,
                               tolerance=1e-12,
                               **kwargs) -> Power:
    """
    This function calculates power for Pillai-Bartlett trace based on the F approx. method.
    V is the "population value" of PBT.
    DF1 and DF2 are the hypothesis and error degrees of freedom.
    OMEGA is the noncentrality parameter.
    FCRIT is the critical value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        a power object
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)
        if (min_rank_C_U - v) <= tolerance:
            warnings.warn('Power is missing because because the min_rank_C_U - v  <= 0.')
        else:
            if min(rank_U, rank_C) == 1:
                omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            else:
                omega = df2 * v / (min_rank_C_U - v)
            power = _multi_power(alpha, df1, df2, omega, total_N)
            return power
    else:
        return _undefined_power()


def pbt_two_moment_null_approx(rank_C: float,
                               rank_X: float,
                               relative_group_sizes,
                               rep_N: float,
                               alpha: float,
                               sigma_star: np.matrix,
                               delta_es: np.matrix,
                               tolerance=1e-12,
                               **kwargs) -> Power:
    """
        This function calculates power for Pillai-Bartlett trace based on the F approx. method.
        V is the "population value" of PBT.
        DF1 and DF2 are the hypothesis and error degrees of freedom.
        OMEGA is the noncentrality parameter.
        FCRIT is the critical value from the F distribution.

        Parameters
        ----------
        rank_C
            rank of C matrix
        rank_U
            rank of U matrix
        rank_X
            rank of X matrix
        total_N
            total N
        eval_HINVE
            eigenvalues for H*INV(E)
        alpha
            Significance level for target GLUM test
        tolerance
            value below which a number is considered zero. defaults to 1e-12

        Returns
        -------
        power
            a power object
        """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1, df2 = _pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)
        if (min_rank_C_U - v) <= tolerance:
            warnings.warn('Power is missing because because the min_rank_C_U - v  <= 0.')
        else:
            if min(rank_U, rank_C) == 1:
                omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            else:
                omega = df2 * v / (min_rank_C_U - v)

            power = _multi_power(alpha, df1, df2, omega, total_N)
            return power

    return _undefined_power()


def pbt_one_moment_null_approx_obrien_shieh(rank_C: float,
                                            rank_X: float,
                                            relative_group_sizes,
                                            rep_N: float,
                                            alpha: float,
                                            sigma_star: np.matrix,
                                            delta_es: np.matrix,
                                            tolerance=1e-12,
                                            **kwargs) -> Power:
    """
        This function calculates power for Pillai-Bartlett trace based on the F approx. method.
        V is the "population value" of PBT.
        DF1 and DF2 are the hypothesis and error degrees of freedom.
        OMEGA is the noncentrality parameter.
        FCRIT is the critical value from the F distribution.

        Parameters
        ----------
        rank_C
            rank of C matrix
        rank_U
            rank of U matrix
        rank_X
            rank of X matrix
        total_N
            total N
        eval_HINVE
            eigenvalues for H*INV(E)
        alpha
            Significance level for target GLUM test
        tolerance
            value below which a number is considered zero. defaults to 1e-12

        Returns
        -------
        power
            a power object
        """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _trace(eval_HINVE, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)

        if (min_rank_C_U - v) <= tolerance:
            warnings.warn('Power is missing because because the min_rank_C_U - v  <= 0.')
        else:
            omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            power = _multi_power(alpha, df1, df2, omega, total_N)
            return power
    return _undefined_power()


def pbt_two_moment_null_approx_obrien_shieh(rank_C: float,
                                            rank_X: float,
                                            relative_group_sizes,
                                            rep_N: float,
                                            alpha: float,
                                            sigma_star: np.matrix,
                                            delta_es: np.matrix,
                                            tolerance=1e-12,
                                            **kwargs) -> Power:
    """
    This function calculates power for Pillai-Bartlett trace based on the F approx. method.
    V is the "population value" of PBT.
    DF1 and DF2 are the hypothesis and error degrees of freedom.
    OMEGA is the noncentrality parameter.
    FCRIT is the critical value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test
    tolerance
        value below which a number is considered zero. defaults to 1e-12

    Returns
    -------
    power
        a power object
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1, df2 = _pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _trace(eval_HINVE, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)
        if (min_rank_C_U - v) <= tolerance:
            warning_message_min_rank_C_U = 'Power is missing because because the min_rank_C_U - v  <= 0.'
            warnings.warn(warning_message_min_rank_C_U)
            return _undefined_power(warning_message_min_rank_C_U)
        else:
            omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            power = _multi_power(alpha, df1, df2, omega, total_N)
            return power
    warning_message_df2_eval_HINVE = 'Power is missing because df2 or eval_HINVE is not valid.'
    warnings.warn(warning_message_df2_eval_HINVE)
    return _undefined_power(warning_message_df2_eval_HINVE)



def wlk_two_moment_null_approx(rank_C: float,
                               rank_X: float,
                               relative_group_sizes,
                               rep_N: float,
                               alpha: float,
                               sigma_star: np.matrix,
                               delta_es: np.matrix,
                               tolerance=1e-12,
                               **kwargs) -> Power:
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if _valid_df2_eigenvalues(eval_HINVE):
        evalt = eval_HINVE * (total_N - rank_X) / total_N
        w = np.exp(np.sum(-np.log(np.ones((1, min_rank_C_U)) + evalt)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X - rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1) / 2
        rs = np.sqrt((rank_C * rank_C * rank_U * rank_U - 4) / (rank_C * rank_C + rank_U * rank_U - 5))
        r1 = (rank_U * rank_C - 2) / 4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1 / rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        omega = total_N * rs * (1 - tempw) / tempw

    if df2 <= tolerance or np.isnan(w) or np.isnan(omega):
        warning_message = 'Power is missing because because the noncentrality could not be computed.'
        warnings.warn(warning_message)
        return _undefined_power(warning_message)
    else:
        return _multi_power(alpha, df1, df2, omega, total_N)


def wlk_two_moment_null_approx_obrien_shieh(rank_C: float,
                                            rank_X: float,
                                            relative_group_sizes,
                                            rep_N: float,
                                            alpha: float,
                                            sigma_star: np.matrix,
                                            delta_es: np.matrix,
                                            tolerance=1e-12,
                                            **kwargs) -> Power:
    """
    This function calculates power for Wilk's Lambda based on
    the F approx. method.  W is the "population value" of Wilks` Lambda,
    DF1 and DF2 are the hypothesis and error degrees of freedom, OMEGA
    is the noncentrality parameter, and FCRIT is the critical value
    from the F distribution. RM, RS, R1, and TEMP are intermediate
    variables.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if _valid_df2_eigenvalues(eval_HINVE):
        w = np.exp(np.sum(-np.log(np.ones((1, min_rank_C_U)) + eval_HINVE * (total_N - rank_X) / total_N)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X - rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1) / 2
        rs = np.sqrt((rank_C * rank_C * rank_U * rank_U - 4) / (rank_C * rank_C + rank_U * rank_U - 5))
        r1 = (rank_U * rank_C - 2) / 4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1 / rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        omega = (total_N * rs) * (1 - tempw) / tempw

    if df2 <= tolerance or np.isnan(w) or np.isnan(omega):
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        return _multi_power(alpha, df1, df2, omega, total_N)
    return _undefined_power()


def special(rank_C: float,
            rank_X: float,
            relative_group_sizes,
            rep_N: float,
            alpha: float,
            sigma_star: np.matrix,
            delta_es: np.matrix,
            tolerance=1e-12,
            **kwargs) -> Power:
    """
    This function performs two disparate tasks. For B=1 (UNIVARIATE
    TEST), the powers are calculated more efficiently. For A=1 (SPECIAL
    MULTIVARIATE CASE), exact multivariate powers are calculated.
    Powers for the univariate tests require separate treatment.
    DF1 & DF2 are the hypothesis and error degrees of freedom,
    OMEGA is the noncentrality parameter, and FCRIT is the critical
    value from the F distribution.

    Parameters
    ----------
    rank_C
        rank of C matrix
    rank_U
        rank of U matrix
    rank_X
        rank of X matrix
    total_N
        total N
    eval_HINVE
        eigenvalues for H*INV(E)
    alpha
        Significance level for target GLUM test

    Returns
    -------
    power
        power for Hotelling-Lawley trace & CL if requested
    """
    error_sum_square, hypothesis_sum_square, rank_U, total_N = calc_properties(delta_es=delta_es,
                                                                               rank_X=rank_X,
                                                                               relative_group_sizes=relative_group_sizes,
                                                                               rep_N=rep_N,
                                                                               sigma_star=sigma_star)
    try:
        np.linalg.cholesky(error_sum_square)
    except:
        return _undefined_power()
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = total_N - rank_X - rank_U + 1
    eval_HINVE = _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = eval_HINVE[0] * (total_N - rank_X)
        return _multi_power(alpha, df1, df2, omega, total_N, **kwargs)
    return _undefined_power()


def _df1_rank_c_u(rank_C: float, rank_U: float) -> float:
    """Calculate df1 from the rank of the C and U matrices"""
    df1 = rank_C * rank_U
    return df1


def _multi_power(alpha: float,
                 df1: float,
                 df2: float,
                 omega: float,
                 total_N: float,
                 **kwargs) -> Power:
    """ The common part for these four multirep methods computing power"""
    noncentrality_dist = None
    quantile = None
    confidence_interval = None
    for key, value in kwargs.items():
        if key == 'noncentrality_distribution':
            noncentrality_dist = value
        if key == 'quantile':
            quantile = value
        if key == 'confidence_interval':
            confidence_interval = value

    fcrit = finv(1 - alpha, df1, df2)
    if noncentrality_dist and quantile:
        omega = __calc_quantile_omega(noncentrality_dist, quantile)
        prob, fmethod = probf(fcrit, df1, df2, omega)
    elif noncentrality_dist and not quantile:
        prob, fmethod = noncentrality_dist.unconditional_power_simpson(fcrit=fcrit, df1=df1, df2=df2)
    else:
        prob, fmethod = probf(fcrit, df1, df2, omega)

    if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
        powerval = alpha
    else:
        powerval = 1 - prob
    powerval = float(powerval)
    power = Power(powerval, omega, fmethod)
    if confidence_interval:
        cl_type = _get_cl_type(confidence_interval)

        power.glmmpcl(is_multirep=True,
                      alphatest=alpha,
                      dfh=df1,
                      n2=total_N,
                      dfe2=df2,
                      cl_type=cl_type,
                      n_est=confidence_interval.n_est,
                      rank_est=confidence_interval.rank_est,
                      alpha_cl=confidence_interval.lower_tail,
                      alpha_cu=confidence_interval.upper_tail,
                      fcrit=fcrit,
                      tolerance=1e-12,
                      omega=omega)
    return power


def _get_cl_type(confidence_interval):
    cl_type = Constants.CLTYPE_DESIRED_KNOWN
    if not confidence_interval.beta_known:
        cl_type = Constants.CLTYPE_DESIRED_ESTIMATE
    return cl_type


def __calc_quantile_omega(noncentrality_distribution, quantile):
    omega = noncentrality_distribution.inverseCDF(quantile)
    return omega


def _trace(eval_HINVE, rank_X, total_N):
    """Calculate the value \'trace\'"""
    trace = eval_HINVE * (total_N - rank_X) / total_N
    return trace


def _calc_omega(min_rank_C_U: float, eval_HINVE: [], rank_X: float, total_N: float) -> float:
    """calculate the noncentrality parameter, omega"""
    hlt = _trace(np.sum(eval_HINVE), rank_X, total_N)
    omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
    return omega


def _calc_hlt_omega(min_rank_C_U: float, eval_HINVE: [], rank_X: float, total_N: float, df2: float):
    """calculate the noncentrality parameter, omega, for a hotelling lawley trace."""
    if min_rank_C_U == 1:
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
    else:
        hlt = np.sum(eval_HINVE)
        omega = df2 * (hlt / min_rank_C_U)
    return omega


def _hlt_one_moment_df2(min_rank_C_U: float, rank_U: float, rank_X: float, total_N: float) -> float:
    """Calculate df2 for a hlt which is using an approximator which matches one moment"""
    df2 = min_rank_C_U * (total_N - rank_X - rank_U - 1) + 2
    return df2


def _hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N):
    """Calculate df2 for a hlt which is using an approximator which matches two moments"""
    df2 = (total_N - rank_X) * (total_N - rank_X) - (total_N - rank_X) * (2 * rank_U + 3) + rank_U * (rank_U + 3);
    df2 = df2 / ((total_N - rank_X) * (rank_C + rank_U + 1) - (rank_C + 2 * rank_U + rank_U * rank_U - 1));
    df2 = 4 + (rank_C * rank_U + 2) * df2;
    return df2


def _valid_df2_eigenvalues(eval_HINVE: [], df2=1, tolerance=1e-12) -> bool:
    """check that df2 is positive and thath the eigenvalues have been calculated"""
    # df2 need to be > 0 and eigenvalues not missing
    if df2 <= tolerance or np.isnan(df2) or np.isnan(eval_HINVE[0]):
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
        return False
    else:
        return True


def _pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N):
    """Calculate df2 for a pbt which is using an approximator which matches one moment"""
    min_rank_C_U = min(rank_C, rank_U)
    df2 = min_rank_C_U * (total_N - rank_X + min_rank_C_U - rank_U)
    return df2


def _pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N):
    """ calculate the degrees of freedom df1, df2 for a pbt which is using an approximator which matches two moments"""
    min_rank_C_U = min(rank_C, rank_U)
    mu1 = rank_C * rank_U / (total_N - rank_X + rank_C)
    factor1 = (total_N - rank_X + rank_C - rank_U) / (total_N - rank_X + rank_C - 1)
    factor2 = (total_N - rank_X) / (total_N - rank_X + rank_C + 2)
    variance = 2 * rank_C * rank_U * factor1 * factor2 / (total_N - rank_X + rank_C) ** 2
    mu2 = variance + mu1 ** 2
    m1 = mu1 / min_rank_C_U
    m2 = mu2 / (min_rank_C_U * min_rank_C_U)
    denom = m2 - m1 * m1
    df1 = 2 * m1 * (m1 - m2) / denom
    df2 = 2 * (m1 - m2) * (1 - m1) / denom
    return df1, df2


def _pbt_population_value(evalt, min_rank_C_U):
    """ calculate the populations value for a pbt"""
    v = np.sum(evalt / (np.ones((1, min_rank_C_U)) + evalt))
    return v


def _pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N):
    """ calculate evalt for pbt"""
    if min(rank_U, rank_C) == 1:
        evalt = _trace(eval_HINVE, rank_X, total_N)
    else:
        evalt = eval_HINVE
    return evalt


def _undefined_power(error_message=None):
    """ Returns a Power object with NaN power and noncentralith and missing fmethod"""
    return Power(float('nan'), float('nan'), Constants.FMETHOD_MISSING, error_message)


def calc_properties(delta_es, rank_X, relative_group_sizes, rep_N, sigma_star):
    """
    Calculate properties of this design for a given samplesize.

    :param delta_es:
    :param rank_X:
    :param relative_group_sizes:
    :param rep_N:
    :param sigma_star:
    :return:
    """
    rank_U = np.shape(sigma_star)[0]
    total_N = rep_N * sum(relative_group_sizes) * 1.0
    error_sum_square = calc_error_sum_square(total_n=total_N,
                                             rank_x=rank_X,
                                             sigma_star=sigma_star)
    hypothesis_sum_square = calc_hypothesis_sum_square(repeated_rows_in_design_matrix=rep_N,
                                                       delta=delta_es)
    return error_sum_square, hypothesis_sum_square, rank_U, total_N


def _calc_eval(min_rank_C_U, error_sum_square, hypothesis_sum_square):
    """ Calculate eigenvalues for H*INV(E) for Multi-rep"""
    # inverse_error_sum = np.linalg.inv(np.linalg.cholesky(error_sum_square))
    # hei_orth = inverse_error_sum * hypothesis_sum_square * inverse_error_sum.T
    #
    #
    # # we use a singular value decomposition here rather than calvulating eigenvalues
    # # for numerical stability. We can do this because hei_orth is a square
    # # symmetric non-negative definite matrix.
    # eval = np.linalg.svd(hei_orth, full_matrices=True, compute_uv=False)

    inverse_error_sum = np.linalg.inv(np.linalg.cholesky(error_sum_square))
    hei_orth = inverse_error_sum * hypothesis_sum_square * inverse_error_sum.T
    hei_orth_symm = (hei_orth + hei_orth.T) / 2
    # get the eigenvalues of hei_orth_symm using a singular value decomposition
    # eigenvalues is an array of dimension 1 x b
    eigenvalues = np.linalg.svd(hei_orth_symm, full_matrices=False, compute_uv=False, hermitian=True)
    eval = eigenvalues[0:min_rank_C_U]
    return eval

def calc_error_sum_square(total_n, rank_x, sigma_star):
    """
    Calculate error sum of squares matrix = nu_e * sigma star

    :param total_n: total samplesize
    :param rank_x: rank of the design matrix X
    :param sigma_star: sigma star
    :return: error sums of squares matrix
    """
    nu_e = total_n - rank_x
    return float(nu_e) * sigma_star

def calc_hypothesis_sum_square(repeated_rows_in_design_matrix, delta):
    return float(repeated_rows_in_design_matrix) * delta
