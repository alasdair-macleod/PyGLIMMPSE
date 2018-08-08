import warnings

import numpy as np

from pyglimmpse.constants import Constants
from pyglimmpse.finv import finv
from pyglimmpse.model.power import Power
from pyglimmpse.probf import probf


def hlt_one_moment_null_approximator(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _hlt_one_moment_df2(min_rank_C_U, rank_U, rank_X, total_N)
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_hlt_omega(min_rank_C_U, eval_HINVE, rank_X, total_N, df2)
        return _multi_power(alpha, df1, df2, omega)
    return _undefined_power()


def hlt_two_moment_null_approximator(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    """    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 2  McKeon (1974) two moment null approx
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_hlt_omega(min_rank_C_U, eval_HINVE, rank_X, total_N, df2)
        return _multi_power(alpha, df1, df2, omega)
    else:
        return _undefined_power()


def hlt_one_moment_null_approximator_obrien_shieh(rank_C: float,
                                                  rank_U: float,
                                                  rank_X: float,
                                                  total_N: float,
                                                  eval_HINVE: [],
                                                  alpha: float,
                                                  tolerance=1e-12 ) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 3  Pillai (1959) one moment null approx+ OS noncen mult
    df2 = _hlt_one_moment_df2(min_rank_C_U, rank_U, rank_X, total_N)

    # df2 need to > 0 and eigenvalues not missing
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        return _multi_power(alpha, df1, df2, omega)
    else:
        return _undefined_power()


def hlt_two_moment_null_approximator_obrien_shieh(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12 ) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 4  McKeon (1974) two moment null approx+ OS noncen mult
    df2 = _hlt_two_moment_df2(rank_C, rank_U, rank_X, total_N)

    # df2 need to > 0 and eigenvalues not missing
    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
        return _multi_power(alpha, df1, df2, omega)
    else:
        return _undefined_power()


def pbt_one_moment_null_approx(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12 ) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N)

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
            power = _multi_power(alpha, df1, df2, omega)
            return power
    else:
        return _undefined_power()


def pbt_two_moment_null_approx(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1, df2 = _pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)

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

            power = _multi_power(alpha, df1, df2, omega)
            return power

    return _undefined_power()


def pbt_one_moment_null_approx_obrien_shieh(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = _pbt_one_moment_df2(rank_C, rank_U, rank_X, total_N)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _trace(eval_HINVE, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)

        if (min_rank_C_U - v) <= tolerance:
            warnings.warn('Power is missing because because the min_rank_C_U - v  <= 0.')
        else:
            omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            power = _multi_power(alpha, df1, df2, omega)
            return power
    return _undefined_power()


def pbt_two_moment_null_approx_obrien_shieh(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1, df2 = _pbt_two_moment_df1_df2(rank_C, rank_U, rank_X, total_N)

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        evalt = _trace(eval_HINVE, rank_X, total_N)
        v = _pbt_population_value(evalt, min_rank_C_U)
        if (min_rank_C_U - v) <= tolerance:
            warnings.warn('Power is missing because because the min_rank_C_U - v  <= 0.')
        else:
            omega = total_N * min_rank_C_U * v / (min_rank_C_U - v)
            power = _multi_power(alpha, df1, df2, omega)
            return power
    return _undefined_power()


def wlk_two_moment_null_approx(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if _valid_df2_eigenvalues(eval_HINVE):
        w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X -rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1)/2
        rs = np.sqrt(rank_C*rank_C*rank_U*rank_U - 4) / (rank_C*rank_C + rank_U*rank_U - 5)
        r1 = (rank_U - rank_C - 2)/4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1/rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        omega = df2 * (1 - tempw) / tempw

    if df2 <= tolerance or np.isnan(w) or np.isnan(omega):
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        return _multi_power(alpha, df1, df2, omega)
    return _undefined_power()


def wlk_two_moment_null_approx_obrien_shieh(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_c_u(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if _valid_df2_eigenvalues(eval_HINVE):
        w = np.exp(np.sum(-np.log(np.ones((min_rank_C_U, 1)) + eval_HINVE * (total_N - rank_X)/total_N)))

    if min_rank_C_U == 1:
        df2 = total_N - rank_X -rank_U + 1
        rs = 1
        tempw = w
    else:
        rm = total_N - rank_X - (rank_U - rank_C + 1)/2
        rs = np.sqrt(rank_C*rank_C*rank_U*rank_U - 4) / (rank_C*rank_C + rank_U*rank_U - 5)
        r1 = (rank_U - rank_C - 2)/4
        if np.isnan(w):
            tempw = float('nan')
        else:
            tempw = np.power(w, 1/rs)
        df2 = (rm * rs) - 2 * r1

    if np.isnan(tempw):
        omega = float('nan')
    else:
        omega = (total_N * rs) * (1 - tempw) /tempw

    if df2 <= tolerance or np.isnan(w) or np.isnan(omega):
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        return _multi_power(alpha, df1, df2, omega)
    return _undefined_power()


def special(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
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
    df1 = _df1_rank_c_u(rank_C, rank_U)
    df2 = total_N - rank_X - rank_U + 1

    if _valid_df2_eigenvalues(eval_HINVE, df2, tolerance):
        omega = eval_HINVE[0] * (total_N - rank_X)
        return _multi_power(alpha, df1, df2, omega)
    return _undefined_power()


def _df1_rank_c_u(rank_C: float, rank_U: float) -> float:
    """Calculate df1 from the rank of the C and U matrices"""
    df1 = rank_C * rank_U
    return df1


def _multi_power(alpha: float, df1: float, df2: float, omega: float) -> Power:
    """ The common part for these four multirep methods computing power"""
    fcrit = finv(1 - alpha, df1, df2)
    prob, fmethod = probf(fcrit, df1, df2, omega)
    if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
        powerval = alpha
    else:
        powerval = 1 - prob
    powerval = float(powerval)
    power = Power(powerval, omega, fmethod)
    return power


def _trace(eval_HINVE, rank_X, total_N):
    """Calculate the value \'trace\'"""
    trace = eval_HINVE * (total_N - rank_X) / total_N
    return trace


def _calc_omega(min_rank_C_U: float, eval_HINVE: [], rank_X: float, total_N: float) -> float:
    """calculate the noncentrality parameter, omega"""
    hlt = _trace(eval_HINVE, rank_X, total_N)
    omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
    return omega


def _calc_hlt_omega(min_rank_C_U: float, eval_HINVE: [], rank_X: float, total_N: float, df2:float):
    """calculate the noncentrality parameter, omega, for a hotelling lawley trace."""
    if min_rank_C_U == 1:
        omega = _calc_omega(min_rank_C_U, eval_HINVE, rank_X, total_N)
    else:
        hlt = eval_HINVE
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


def _valid_df2_eigenvalues(eval_HINVE: [],df2=1, tolerance=1e-12) -> bool:
    """check that df2 is positive and thath the eigenvalues have been calculates"""
    # df2 need to be > 0 and eigenvalues not missing
    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
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
    v = sum(evalt / (np.ones((min_rank_C_U, 1)) + evalt))
    return v


def _pbt_uncorrected_evalt(eval_HINVE, rank_C, rank_U, rank_X, total_N):
    """ calculate evalt for pbt"""
    if min(rank_U, rank_C) == 1:
        evalt = _trace(eval_HINVE, rank_X, total_N)
    else:
        evalt = eval_HINVE
    return evalt


def _undefined_power():
    """ Returns a Power object with NaN power and noncentralith and missing fmethod"""
    return Power(float('nan'), float('nan'), Constants.FMETHOD_MISSING)

