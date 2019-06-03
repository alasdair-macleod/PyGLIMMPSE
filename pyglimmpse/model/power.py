import warnings
import numpy as np
from scipy import special
from scipy.stats import chi2

from pyglimmpse.constants import Constants
from pyglimmpse.finv import finv
from pyglimmpse.probf import probf


def subtrtact_target_power(a, b):
    return a.power - b

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
    def __init__(self, power=0, noncentrality_parameter=0, fmethod="DEFAULT", error_message = None):
        self.power = power
        self.noncentrality_parameter = noncentrality_parameter
        self.fmethod = fmethod
        self.lower_bound = None
        self.upper_bound = None
        self.error_message = error_message

    def glmmpcl(self,
                is_multirep,
                alphatest,
                dfh,  # df1
                n2,  # total_N ??? what is this
                dfe2,  # df2
                cl_type,
                n_est,
                rank_est,
                alpha_cl,
                alpha_cu,
                fcrit,
                tolerance,
                omega,
                **kwargs):
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
        :param dfe1:
            Error df for target hypothesis
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
        :param fcrit:
            critical valuse for prob F.
        :param tolerance:
            value below which numbes are declared zero
        :param omega:
            noncentrality parameter

        """
        df1_unirep = None
        if 'df1_unirep' in kwargs.keys():
            df1_unirep = kwargs['df1_unirep']
        if cl_type == Constants.CLTYPE_DESIRED_KNOWN or cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
            if np.isnan(self.power):
                warnings.warn('Powerwarn16: Confidence limits are missing because power is missing.')
            else:
                dfe1 = n_est - rank_est
                self.lower_bound = self._calc_lower_bound(is_multirep=is_multirep,
                                                          alphatest=alphatest,
                                                          alpha_cl=alpha_cl,
                                                          cl_type=cl_type,
                                                          dfe1=dfe1,
                                                          dfe2=dfe2,
                                                          dfh=dfh,
                                                          fcrit=fcrit,
                                                          omega=omega,
                                                          tolerance=tolerance,
                                                          df1_unirep=df1_unirep)
                self.upper_bound = self._calc_upper_bound(is_multirep=is_multirep,
                                                          alphatest=alphatest,
                                                          alpha_cu=alpha_cu,
                                                          cl_type=cl_type,
                                                          dfe1=dfe1,
                                                          dfe2=dfe2,
                                                          dfh=dfh,
                                                          fcrit=fcrit,
                                                          noncen_e=omega,
                                                          tolerance=tolerance,
                                                          df1_unirep=df1_unirep)
                self._warn_conservative_ci(alpha_cl, cl_type, n2, n_est)
        else:
            self.lower_bound = None
            self.upper_bound = None

    @staticmethod
    def subtrtact_target_power(a, b):
        return a.power - b

    def _warn_conservative_ci(self, alpha_cl, cl_type, n2, n_est):
        """warning for conservative confidence interval"""
        if (cl_type == Constants.CLTYPE_DESIRED_KNOWN or
                    cl_type == Constants.CLTYPE_DESIRED_ESTIMATE) and n2 != n_est:
            if self.lower_bound and self.lower_bound.noncentrality_parameter and alpha_cl > 0 and self.lower_bound.noncentrality_parameter == 0:
                warnings.warn('The lower confidence limit on power is conservative.')
            if self.upper_bound and self.upper_bound.noncentrality_parameter and alpha_cl == 0 and self.upper_bound.noncentrality_parameter == 0:
                warnings.warn('The upper confidence limit on power is conservative.')

    def _calc_upper_bound(self, is_multirep, alphatest, alpha_cu, cl_type, dfe1, dfe2, dfh, fcrit, noncen_e, tolerance, **kwargs):
        """Calculate upper bound for noncentrality"""
        df1_unirep = None
        if 'df1_unirep' in kwargs.keys():
            df1_unirep = kwargs['df1_unirep']
        lower_tail_prob = 1 - alpha_cu
        upper_tail_prob = alpha_cu
        # default values if alpha < tolerance
        noncentrality = float('Inf')
        prob = 0
        power = self._calc_bound(is_multirep=is_multirep,
                                 alphatest=alphatest,
                                 alpha=alpha_cu,
                                 lower_tail_prob=lower_tail_prob,
                                 upper_tail_prob=upper_tail_prob,
                                 prob=prob,
                                 noncentrality=noncentrality,
                                 cl_type=cl_type,
                                 dfe1=dfe1,
                                 dfe2=dfe2,
                                 dfh=dfh,
                                 fcrit=fcrit,
                                 omega=noncen_e,
                                 tolerance=tolerance,
                                 df1_unirep=df1_unirep)
        return power

    def _calc_lower_bound(self, is_multirep, alphatest, alpha_cl, cl_type, dfe1, dfe2, dfh, fcrit, omega, tolerance, **kwargs):
        """Calculate lower bound for noncentrality"""
        df1_unirep = None
        if 'df1_unirep' in kwargs.keys():
            df1_unirep = kwargs['df1_unirep']
        lower_tail_prob = alpha_cl
        upper_tail_prob = 1 - alpha_cl
        # default values if alpha < tolerance
        noncentrality = 0
        prob = 1 - alphatest
        power = self._calc_bound(is_multirep=is_multirep,
                                 alphatest=alphatest,
                                 alpha=alpha_cl,
                                 lower_tail_prob=lower_tail_prob,
                                 upper_tail_prob=upper_tail_prob,
                                 prob=prob,
                                 noncentrality=noncentrality,
                                 cl_type=cl_type,
                                 dfe1=dfe1,
                                 dfe2=dfe2,
                                 dfh=dfh,
                                 fcrit=fcrit,
                                 omega=omega,
                                 tolerance=tolerance,
                                 df1_unirep=df1_unirep)
        return power

    def _calc_bound(self, is_multirep, alphatest, alpha, lower_tail_prob, upper_tail_prob, prob, noncentrality, cl_type, dfe1, dfe2, dfh, fcrit, omega, tolerance, **kwargs):
        """Calculate power bounds """
        df1_unirep = None
        if 'df1_unirep' in kwargs.keys():
            df1_unirep = kwargs['df1_unirep']
        fmethod = Constants.FMETHOD_MISSING
        if alpha > tolerance:
            if cl_type == Constants.CLTYPE_DESIRED_KNOWN:
                if is_multirep:
                    chi = chi2.ppf(lower_tail_prob, dfe1)
                    noncentrality = (chi / dfe1) * omega
                else:
                    chi = chi2.ppf(lower_tail_prob, dfh)
                    noncentrality = (chi / dfh) * omega
            elif cl_type == Constants.CLTYPE_DESIRED_ESTIMATE:
                f_a = omega / dfh
                bound = finv(upper_tail_prob, dfh, dfe1)
                if f_a <= bound:
                    noncentrality = 0
                else:
                    noncentrality = special.ncfdtrinc(dfh, dfe1, upper_tail_prob, f_a)
            if is_multirep:
                prob, fmethod = probf(fcrit, dfh, dfe2, noncentrality)
            else:
                prob, fmethod = probf(fcrit, df1_unirep, dfe2, noncentrality)
        if fmethod == Constants.FMETHOD_NORMAL_LR and prob == 1:
            power_bound = alphatest
        else:
            power_bound = 1 - prob

        power = Power()
        power.power = power_bound
        power.fmethod = fmethod
        power.noncentrality_parameter = noncentrality
        return power