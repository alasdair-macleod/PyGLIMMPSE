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
    df1 = _df1_rank_C_U(rank_C, rank_U)
    df2 = _one_moment_df2(min(rank_C, rank_U), rank_U, rank_X, total_N)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= 0 or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        hlt, omega = _calc_hlt_omega(min(rank_C, rank_U), eval_HINVE, rank_X, total_N, df2)
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)
    power = Power(powerval, omega, fmethod)

    return power

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
    """
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_C_U(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 2  McKeon (1974) two moment null approx
    nu_df2 = (total_N - rank_X)*(total_N - rank_X) - (total_N - rank_X)*(2*rank_U + 3) + rank_U*(rank_U + 3)
    de_df2 = (total_N - rank_X)*(rank_C + rank_U + 1) - (rank_C + 2*rank_U + rank_U*rank_U - 1)
    df2 = 4 + (rank_C*rank_U + 2) * (nu_df2/de_df2)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        hlt, omega = _calc_hlt_omega(min(rank_C, rank_U), eval_HINVE, rank_X, total_N, df2)
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)
    power = Power(powerval, omega, fmethod)

    return power

def hlt_one_moment_null_approximator_obrien_shieh(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12 ) -> Power:
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
    df1 = _df1_rank_C_U(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 3  Pillai (1959) one moment null approx+ OS noncen mult
    df2 = min_rank_C_U * (total_N - rank_X - rank_U - 1) + 2

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        omega = _calc_omega(eval_HINVE, min_rank_C_U, rank_X, total_N)
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)
    power = Power(powerval, omega, fmethod)

    return power

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
    df1 = _df1_rank_C_U(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MultiHLT  Choices for Hotelling-Lawley Trace
    #       = 4  McKeon (1974) two moment null approx+ OS noncen mult
    nu_df2 = (total_N - rank_X)*(total_N - rank_X) - (total_N - rank_X)*(2*rank_U + 3) + rank_U*(rank_U + 3)
    de_df2 = (total_N - rank_X)*(rank_C + rank_U + 1) - (rank_C + 2*rank_U + rank_U*rank_U - 1)
    df2 = 4 + (rank_C*rank_U + 2) * (nu_df2/de_df2)

    # df2 need to > 0 and eigenvalues not missing
    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        omega = _calc_omega(eval_HINVE, min_rank_C_U, rank_X, total_N)
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)
    power = Power(powerval, omega, fmethod)

    return power

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
    df1 = _df1_rank_C_U(rank_C, rank_U)
    df2 = min(rank_C, rank_U) * (total_N - rank_X + min(rank_C, rank_U) - rank_U)

    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        if min(rank_U, rank_C) == 1:
            evalt = eval_HINVE * (total_N - rank_X) / total_N
        else:
            evalt = eval_HINVE

        v = sum(evalt / (np.ones((min(rank_C, rank_U), 1)) + evalt))
        if (min(rank_C, rank_U) - v) <= tolerance:
            powerval = float('nan')
        else:
            if min(rank_U, rank_C) == 1:
                omega = total_N * min(rank_C, rank_U) * v / (min(rank_C, rank_U) - v)
            else:
                omega = df2 * v / (min(rank_C, rank_U) - v)
            powerval, fmethod = _multi_power(alpha, df1, df2, omega)
    power = Power(powerval, omega, fmethod)
    return power

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
    mu1 = rank_C * rank_U / (total_N - rank_X + rank_C)
    factor1 = (total_N - rank_X + rank_C - rank_U) / (total_N - rank_X + rank_C - 1)
    factor2 = (total_N - rank_X) / (total_N - rank_X + rank_C + 2)
    variance = 2 * rank_C * rank_U * factor1 * factor2 / (total_N - rank_X + rank_C) ** 2
    mu2 = variance + mu1 ** 2
    m1 = mu1 / min(rank_C, rank_U)
    m2 = mu2 / (min(rank_C, rank_U) * min(rank_C, rank_U))
    denom = m2 - m1 * m1
    df1 = 2 * m1 * (m1 - m2) / denom
    df2 = 2 * (m1 - m2) * (1 - m1) / denom

    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        power = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        if min(rank_U, rank_C) == 1:
            evalt = eval_HINVE * (total_N - rank_X) / total_N
        else:
            evalt = eval_HINVE

        v = sum(evalt / (np.ones((min(rank_C, rank_U), 1)) + evalt))
        if (min(rank_C, rank_U) - v) <= tolerance:
            powerval = float('nan')
        else:
            if min(rank_U, rank_C) == 1:
                omega = total_N * min(rank_C, rank_U) * v / (min(rank_C, rank_U) - v)
            else:
                omega = df2 * v / (min(rank_C, rank_U) - v)

            powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)
    return power

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
    df1 = _df1_rank_C_U(rank_C, rank_U)
    df2 = min(rank_C, rank_U) * (total_N - rank_X + min(rank_C, rank_U) - rank_U)

    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        power = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        evalt = eval_HINVE * (total_N - rank_X) / total_N
        v = sum(evalt / (np.ones((min(rank_C, rank_U), 1)) + evalt))

        if (min(rank_C, rank_U) - v) <= tolerance:
            power = float('nan')
        else:
            omega = total_N * min(rank_C, rank_U) * v / (min(rank_C, rank_U) - v)
            powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)
    return power

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
    mu1 = rank_C * rank_U / (total_N - rank_X + rank_C)
    factor1 = (total_N - rank_X + rank_C -rank_U) / (total_N - rank_X + rank_C - 1)
    factor2 = (total_N - rank_X) / (total_N - rank_X + rank_C + 2)
    variance = 2 * rank_C * rank_U * factor1 * factor2 / (total_N - rank_X + rank_C)**2
    mu2 = variance + mu1**2
    m1 = mu1 / min(rank_C, rank_U)
    m2 = mu2 / (min(rank_C, rank_U) * min(rank_C, rank_U))
    denom = m2 - m1 * m1
    df1 = 2 * m1 * (m1 - m2) / denom
    df2 = 2 * (m1 - m2) * (1 - m1) /denom

    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        power = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        evalt = eval_HINVE * (total_N - rank_X) / total_N
        v = sum(evalt / (np.ones((min(rank_C, rank_U), 1)) + evalt))
        if (min(rank_C, rank_U) - v) <= tolerance:
            power = float('nan')
        else:
            omega = total_N * min(rank_C, rank_U) * v / (min(rank_C, rank_U) - v)
            powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)
    return power

def wlk_two_moment_null_approx(rank_C: float, rank_U: float, rank_X: float, total_N: float, eval_HINVE: [], alpha: float, tolerance=1e-12) -> Power:
    min_rank_C_U = min(rank_C, rank_U)
    df1 = _df1_rank_C_U(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if np.isnan(eval_HINVE[0]):
        w = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
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
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)
    return power

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
    df1 = _df1_rank_C_U(rank_C, rank_U)

    # MMETHOD default= [4,2,2]
    # MMETHOD[2] Choices for Wilks' Lambda
    #       = 1  Rao (1951) two moment null approx
    #       = 2  Rao (1951) two moment null approx
    #       = 3  Rao (1951) two moment null approx + OS Obrien shieh noncen mult
    #       = 4  Rao (1951) two moment null approx + OS noncen mult
    if np.isnan(eval_HINVE[0]):
        w = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
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
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)

    return power

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
    df1 = _df1_rank_C_U(rank_C, rank_U)
    df2 = total_N - rank_X - rank_U + 1

    if df2 <= tolerance or np.isnan(eval_HINVE[0]):
        powerval = float('nan')
        warnings.warn('Power is missing because because the noncentrality could not be computed.')
    else:
        omega = eval_HINVE[0] * (total_N - rank_X)
        powerval, fmethod = _multi_power(alpha, df1, df2, omega)

    power = Power(powerval, omega, fmethod)

    return power

def _df1_rank_C_U(rank_C: float, rank_U: float) -> float:
    df1 = rank_C * rank_U
    return df1

def _multi_power(alpha: float, df1: float, df2: float, omega: float) -> Power:
    """ The common part for these four multirep methods
        Computing power"""
    fcrit = finv(1 - alpha, df1, df2)
    prob, fmethod = probf(fcrit, df1, df2, omega)
    if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
        power = alpha
    else:
        power = 1 - prob
    power = float(power)
    return power, fmethod

def _calc_omega(eval_HINVE: [], min_rank_C_U: float, rank_X: float, total_N: float) -> float:
    hlt = eval_HINVE * (total_N - rank_X) / total_N
    omega = (total_N * min_rank_C_U) * (hlt / min_rank_C_U)
    return omega

def _calc_hlt_omega(min_rank_C_U: float, eval_HINVE: [], rank_X: float, total_N: float, df2:float):
    if min_rank_C_U == 1:
        omega = _calc_omega(eval_HINVE, min_rank_C_U, rank_X, total_N)
    else:
        hlt = eval_HINVE
        omega = df2 * (hlt / min_rank_C_U)
    return hlt, omega

def _one_moment_df2(min_rank_C_U: float, rank_U: float, rank_X: float, total_N: float) -> float:
    df2 = min_rank_C_U * (total_N - rank_X - rank_U - 1) + 2
    return df2

