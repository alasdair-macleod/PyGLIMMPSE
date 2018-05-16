import warnings
import numpy as np
from scipy import special
from scipy.stats import chi2

from pyglimmpse.constants import Constants
from pyglimmpse.finv import finv
from pyglimmpse.probf import probf


class Power:
    """
    This object represents power and contains metedata about how it was calculated

    Parameters
    ----------
    power
        the power value
    noncentrality_parameter
        the noncentrality parameter used to calculate this power
    fmethod
        constant string referring to the method used to calculate this power
    lower_bound
        the power value chosen as the lower bound for this power calculation. This is itself a Power object
    upper_bound
        the power value chosen as the upper bound for this power calculation. This is itself a Power object


    Methods
    -------
    glmmpcl(self,alphatest,dfh,n2,dfe2,cl_type,n_est,rank_est,alpha_cl,alpha_cu,tolerance,power,omega)

    Notes
    -----
    lower_bound
        power
            confidence interval lower bound
        noncentrality_parameter
            noncentrality confidence interval lower bound
        fmethod
            Method used to calculate probability from F CDF used in lower confidence limits power calculation
    upper_bound
        power
            power confidence interval upper bound
        noncentrality_parameter
            noncentrality confidence interval upper bound
        fmethod
            Method used to calculate probability from F CDF used in lower confidence limits power calculation

    Examples
    --------

    """
    def __init__(self, power=0, noncentrality_parameter=0, fmethod="DEFAULT"):
        self.power = power
        self.noncentrality_parameter = noncentrality_parameter
        self.fmethod = fmethod
        self.lower_bound = None
        self.upper_bound = None

    def glmmpcl(self,
                alphatest,
                dfh,   # df1
                n2,    # total_N ??? what is this
                dfe2,  # df2
                cl_type,
                n_est,
                rank_est,
                alpha_cl,
                alpha_cu,
                tolerance,
                omega):
        """
        This function computes confidence intervals for noncentrality and
        power for a General Linear Hypothesis (GLH  Ho:C*beta=theta0) test
        in the General Linear Univariate Model (GLUM: y=X*beta+e, HILE
        GAUSS), based on estimating the effect, error variance, or neither.
        Methods from Taylor and Muller (1995).

        :param f_a:
            MSH/MSE, the F value observed if BETAhat=BETA and Sigmahat=Sigma, under the alternative hypothesis, with:
                MSH=Mean Square Hypothesis (effect variance)
                MSE=Mean Square Error (error variance)
            NOTE:
                F_A = (N2/N1)*F_EST and
                MSH = (N2/N1)*MSH_EST,
                with "_EST" indicating value which was observed
                in sample 1 (source of estimates)
        :param alphatest:
            Significance level for target GLUM test
        :param dfh:
            degrees of freedom for target GLH
        :param n2:
            What is this???
        :param dfe2:
            Error df for target hypothesis
        :param cl_type:
            =1 if Sigma estimated and Beta known
            =2 if Sigma estimated and Beta estimated
        :param n_est:
            (scalar) # of observations in analysis which yielded BETA and SIGMA estimates
        :param rank_est:
            (scalar) design matrix rank in analysis which yielded BETA and SIGMA estimates
        :param alpha_cl:
            Lower tail probability for confidence interval
        :param alpha_cu:
            Upper tail probability for confidence interval
        :param tolerance:
            value below which numbes are declared zero
        :param omega:
            noncentrality parameter

        """
        if cl_type == Constants.CLTYPE_DESIRED_KNOWN or cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
            if np.isnan(self.power):
                warnings.warn('Powerwarn16: Confidence limits are missing because power is missing.')
            else:
                f_a = omega/dfh
                dfe1, fcrit, noncen_e = self._calc_noncentrality(alphatest, dfe2, dfh, f_a, n_est, rank_est)

                self.lower_bound = self._calc_lower_bound(alphatest, alpha_cl, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance)
                self.upper_bound = self._calc_upper_bound(alphatest, alpha_cu, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance)
                self._warn_conservative_ci(alpha_cl, cl_type, n2, n_est)
        else:
            self.lower_bound = None
            self.upper_bound = None

    def _warn_conservative_ci(self, alpha_cl, cl_type, n2, n_est):
        """warning for conservative confidence interval"""
        if (cl_type == Constants.CLTYPE_DESIRED_KNOWN or
                    cl_type == Constants.CLTYPE_DESIRED_ESTIMATE) and n2 != n_est:
            if self.lower_bound and self.lower_bound.noncentrality_parameter and alpha_cl > 0 and self.lower_bound.noncentrality_parameter == 0:
                warnings.warn('The lower confidence limit on power is conservative.')
            if self.upper_bound and self.upper_bound.noncentrality_parameter and alpha_cl == 0 and self.upper_bound.noncentrality_parameter == 0:
                warnings.warn('The upper confidence limit on power is conservative.')

    def _calc_noncentrality(self, alphatest, dfe2, dfh, f_a, n_est, rank_est):
        """Calculate noncentrality"""
        dfe1 = n_est - rank_est
        noncen_e = dfh * f_a
        fcrit = finv(1 - alphatest, dfh, dfe2)
        return dfe1, fcrit, noncen_e

    def _calc_upper_bound(self, alphatest, alpha_cu, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance):
        """Calculate upper bound for noncentrality"""
        lower_tail_prob = 1 - alpha_cu
        upper_tail_prob = alpha_cu
        # default values if alpha < tolerance
        noncentrality = float('Inf')
        prob = 0
        power = self._calc_bound(alphatest, alpha_cu, lower_tail_prob, upper_tail_prob, prob, noncentrality, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance)
        return power

    def _calc_lower_bound(self, alphatest, alpha_cl, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance):
        """Calculate lower bound for noncentrality"""
        lower_tail_prob = alpha_cl
        upper_tail_prob = 1 - alpha_cl
        # default values if alpha < tolerance
        noncentrality = 0
        prob = 1 - alphatest
        power = self._calc_bound(alphatest, alpha_cl, lower_tail_prob, upper_tail_prob, prob, noncentrality, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance)
        return power

    def glmmpcl_variant(self, alpha, df1, total_N,  df2, cl_type, n_est, alpha_cl, alpha_cu, cl1df, fcrit, tolerance, omega):
        power_l = self._calc_lower_bound_v(alpha, alpha_cl, cl1df, df1, df2, fcrit, omega, tolerance)
        power_u = self._calc_upper_bound_v(alpha, alpha_cu, cl1df, df1, df2, fcrit, omega, tolerance)

        power_l = float(power_l)
        power_u = float(power_u)
        return power_l, power_u

    def _calc_upper_bound_v(self, alpha, alpha_cu, cl1df, df1, df2, fcrit, omega, tolerance):
        if alpha_cu <= tolerance:
            prob_u = 0
            fmethod_u = Constants.FMETHOD_MISSING
            noncen_u = float('nan')
        else:
            chi_u = chi2.ppf(1 - alpha_cu, cl1df)
            noncen_u = omega * (chi_u / cl1df)
            prob_u, fmethod_u = probf(fcrit, df1, df2, noncen_u)
        if fmethod_u == Constants.FMETHOD_NORMAL_LR and prob_u == 1:
            power_u = alpha
        else:
            power_u = 1 - prob_u
        return power_u

    def _calc_lower_bound_v(self, alpha, alpha_cl, cl1df, df1, df2, fcrit, noncen_e, tolerance):
        prob_l = 1 - alpha
        cl_type = Constants.CLTYPE_DESIRED_KNOWN
        fmethod_l = Constants.FMETHOD_MISSING
        noncentrality = float('nan')

        chi = chi2.ppf(alpha_cl, cl1df)
        noncentrality = noncen_e * (chi / cl1df)
        prob_l, fmethod_l = probf(fcrit, df1, df2, noncentrality)
        if fmethod_l == Constants.FMETHOD_NORMAL_LR and prob_l == 1:
            power_l = alpha
        else:
            power_l = 1 - prob_l

        return power_l

    def _calc_bound(self, alphatest, alpha, lower_tail_prob, upper_tail_prob, prob, noncentrality, cl_type, dfe1, dfe2, dfh, fcrit, f_a, noncen_e, tolerance):
        """Calculate power bounds """
        fmethod = Constants.FMETHOD_MISSING
        if alpha > tolerance:
            if cl_type == Constants.CLTYPE_DESIRED_KNOWN:
                chi = chi2.ppf(lower_tail_prob, dfe1)
                noncentrality = (chi / dfe1) * noncen_e
            elif cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
                bound = finv(upper_tail_prob, dfh, dfe1)
                if f_a <= bound:
                    noncentrality = 0
                else:
                    noncentrality = special.ncfdtrinc(dfh, dfe1, upper_tail_prob, f_a)
            prob, fmethod = probf(fcrit, dfh, dfe2, noncentrality)
        if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
            power_bound = alphatest
        else:
            power_bound = 1 - prob

        power = Power()
        power.power = power_bound
        power.fmethod = fmethod
        power.noncentrality_parameter = noncentrality
        return power