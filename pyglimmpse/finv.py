import numpy as np
from scipy import stats

def finv(alpha, df1, df2):
    """
    This function returns the critical value from a central F(DF1, DF2) where df is degree of freedom
    distribution for a given signficance level alpha.  It screens the FINV function for numerator DF
    greater than 1*10^7.6 or denominator DF greater than 1*10^9.4. For large degrees of freedom, it returns
    a missing value.

    :param alpha:
    :param df1:
    :param df2:
    :return: fcrit, Critical value from the probability that a variable
                    distributed F(DF1,DF2) <= FCRIT is equal to (1-ALPHA)
    """

    if df1 > 10**7.6 or df1 < 0 or df2 < 0:
        fcrit = np.NaN
    else:
        if df2 <= 10**9.4:
            fcrit = stats.f.ppf(alpha, df1, df2)
        else:
            fcrit = stats.chi2.ppf(alpha, df1)

    return fcrit