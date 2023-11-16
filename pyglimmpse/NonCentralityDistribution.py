#!/usr/bin/env python
import math

import numpy as np
from scipy import optimize
from scipy.integrate.quadrature import tupleset
from scipy.stats import f
from scipy import special
import scipy.integrate as integrate

from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import WeightedSumOfNoncentralChiSquaresDistribution
from pyglimmpse.constants import Constants
from pyglimmpse.exceptions.glimmpse_exception import GlimmpseCalculationException, GlimmpseValidationException
from pyglimmpse.probf import probf

""" generated source for module NonCentralityDistribution """
from pyglimmpse.chisquareterm import ChiSquareTerm

class NonCentralityDistribution(object):
    """ generated source for class NonCentralityDistribution """
    NOT_POSITIVE_DEFINITE = "Unfortunately, there is no solution for this combination of input parameters. " + "A matrix that arose during the computation is not positive definite. " + "It may be possible to reduce expected covariate/response correlations " + "and obtain a soluble combination."
    MAX_ITERATIONS = 10000
    ACCURACY = 0.001

    #  intermediate forms
    T1 = None
    FT1 = None
    S = None
    mzSq = None
    H1 = float()
    H0 = 0
    qF = int()
    a = int()
    N = float()
    sEigenValues = []
    sStar = 0

    #  indicates if an "exact" cdf should be calculated via Davie's algorithm or
    #  with the Satterthwaite approximation from Glueck & Muller
    exact = bool()

    # 
    #      * Create a non-centrality distribution for the specified inputs.
    #      * @param params GLMM input parameters
    #      * @param exact if true, Davie's algorithm will be used to compute the cdf,
    #      * otherwise a Satterthwaite style approximation is used.
    #      * @throws IllegalArgumentException
    #      
    def __init__(self, test, FEssence, perGroupN, CFixed, CGaussian, thetaDiff, sigmaStar, stddevG, exact):
        """ generated source for method __init__ """
        self.initialize(
            test=test,
            FEssence=FEssence,
            perGroupN=perGroupN,
            Cfixed=CFixed,
            CGaussian=CGaussian,
            thetaDiff=thetaDiff,
            sigmaStar=sigmaStar,
            stddevG = stddevG,
            exact=exact)

    # 
    #      * Pre-calculate intermediate matrices, perform setup, etc.
    #      
    def initialize(self, test, FEssence, perGroupN, Cfixed, CGaussian, thetaDiff, sigmaStar, stddevG, exact):
        """ generated source for method initialize """
        #  reset member variables
        self.T1 = None
        self.FT1 = None
        self.S = None
        self.mzSq = None
        self.H0 = 0
        self.sStar = 0
        self.N = float(FEssence.shape[0]) * perGroupN
        self.exact = exact
        self.errors = []
        try:
            #  TODO: need to calculate H0, need to adjust H1 for Unirep
            #  get design matrix for fixed parameters only
            #self.qF = FEssence.getColumnDimension()
            self.qF = FEssence.shape[1]
            #  get fixed contrasts

            #  build intermediate terms h1, S
            FtFinverse = np.linalg.inv(FEssence.T * FEssence)
            PPt = Cfixed * FtFinverse * (1 / perGroupN) * Cfixed.T
            self.T1 = self.forceSymmetric(np.linalg.inv(PPt))
            self.FT1 = np.linalg.cholesky(self.T1)
            #calculate theta difference
            # TODO I think CRand should already be an array
            # C = np.concatenate((np.array(CFixed), np.array(CRand)), axis=1)
            #TODO: specific to HLT or UNIREP
            sigmaStarInverse = self.getSigmaStarInverse(sigmaStar, test)
            H1matrix = thetaDiff.T * self.T1 * thetaDiff * sigmaStarInverse
            self.H1 = np.trace(H1matrix)
            if self.H1 > 0:
                # We could use this to truncate the eigenvectors and eigenvalues
                # thus removing 'zero dust' values
                # a = thetaDiff.shape[0]
                # b = thetaDiff.shape[1]
                # s = min(a,b)

                # Matrix which represents the non-centrality parameter as a linear combination of chi-squared r.v.'s.
                self.S = self.FT1.T * thetaDiff * sigmaStarInverse * thetaDiff.T * self.FT1 * (1 / self.H1)
                self.S = self.forceSymmetric(self.S)
                # We use the S matrix to generate the F critical value, numerical df's, and denominator df's
                # for a central F distribution.  The resulting F distribution is used as an approximation
                # for the distribution of the non-centrality parameter.
                # See formulas 18-21 and A8,A10 from Glueck & Muller (2003) for details.

                # get the eigenvalues of self.S using a singular value decomposition
                svecsT, sEigenValuesT, vh = np.linalg.svd(self.S, full_matrices=False, compute_uv=True, hermitian=True)
                self.sEigenValues = np.matrix(sEigenValuesT).T

                self.H0 = self.H1 * (1 - self.sEigenValues.item(0))
                if self.H0 <= 0:
                    self.H0 = 0

                for value in self.sEigenValues[0]:
                    if value > 0:
                        self.sStar += 1
                # TODO: throw error if sStar is <= 0
                # TODO: NO: throw error if sStar != sEigenValues.length instead???
                # create square matrix using these
                self.mzSq = svecsT * self.FT1.T * CGaussian * (1 / stddevG)
                i = 0
                while i < self.mzSq.shape[0]:
                    j = 0
                    #while j < self.mzSq.getColumnDimension():
                    while j <  self.mzSq.shape[1]:
                        entry = self.mzSq[i, j]
                        self.mzSq[i, j] = entry * entry
                        j += 1
                    i += 1
        except Exception as e:
            raise e

    def setPerGroupSampleSize(self, perGroupN):
        """ generated source for method setPerGroupSampleSize """
        self.initialize(self.test, self.FEssence, self.FtFinverse, perGroupN, self.CFixed, self.CRand, self.U, self.thetaNull, self.beta, self.sigmaError, self.sigmaG, self.exact)

    def setBeta(self, beta):
        """ generated source for method setBeta """
        self.initialize(self.test, self.FEssence, self.FtFinverse, self.perGroupN, self.CFixed, self.CRand, self.U, self.thetaNull, beta, self.sigmaError, self.sigmaG, self.exact)

    def cdf(self, w):
        """ generated source for method cdf """
        if self.H1 <= 0 or w <= self.H0:
            return 0
        if self.H1 - w <= 0:
            return 1
        chiSquareTerms = []

        try:
            b0 = 1 - w / self.H1
            m1Positive = 0
            m1Negative = 0
            m2Positive = 0
            m2Negative = 0

            numPositive = 0
            numNegative = 0
            lastPositiveNoncentrality = 0  # for special cases
            lastNegativeNoncentrality = 0

            nu = self.N - self.qF
            lambda_ = b0
            delta = 0
            chiSquareTerms.append(ChiSquareTerm(lambda_, nu, delta))
            # add in the first chi-squared term in the estimate of the non-centrality
            # (expressed as a sum of weighted chi-squared r.v.s)
            # initial chi-square term is central (delta=0) with N-qf df, and lambda = b0
            if lambda_ > 0:
                # positive terms
                numPositive += 1
                lastPositiveNoncentrality = delta
                m1Positive += lambda_ * (nu + delta)
                m2Positive += lambda_ * lambda_ * 2 * (nu + 2 * delta)
            elif lambda_ < 0:
                # negative terms - we take absolute value of lambda where needed
                numNegative += 1
                lastNegativeNoncentrality = delta
                m1Negative += -1 * lambda_ * (nu + delta)
                m2Negative += lambda_ * lambda_ * 2 * (nu + 2 * delta)
            # accumulate the remaining terms
            k = 0
            while k < self.sStar:
                if k < self.sStar:
                    # for k = 1 (well, 0 in java array terms and 1 in the paper) to sStar, chi-square term is
                    # non-central (delta = mz^2), 1 df, lambda = (b0 - kth eigen value of S)
                    nu = 1
                    lambda_ = b0 - self.sEigenValues[k]
                    delta = self.mzSq[k, 0]
                    chiSquareTerms.append(ChiSquareTerm(lambda_, nu, delta))
                else:
                    # for k = sStar+1 to a, chi-sqaure term is non-central (delta = mz^2), 1 df,
                    # lambda = b0
                    nu = 1
                    lambda_ = b0
                    delta = self.mzSq[k, 0]
                    chiSquareTerms.add(ChiSquareTerm(lambda_, nu, delta))
                # accumulate terms
                if lambda_ > 0:
                    # positive terms
                    numPositive += 1
                    lastPositiveNoncentrality = delta
                    m1Positive += lambda_ * (nu + delta)
                    m2Positive += lambda_ * lambda_ * 2 * (nu + 2 * delta)
                elif lambda_ < 0:
                    # negative terms - we take absolute value of lambda where needed
                    numNegative += 1
                    lastNegativeNoncentrality = delta
                    m1Negative += -1 * lambda_ * (nu + delta)
                    m2Negative += lambda_ * lambda_ * 2 * (nu + 2 * delta)
                k += 1
                # Note, we deliberately ignore terms for which lambda == 0
            # handle special cases
            if numNegative == 0:
                return 0
            if numPositive == 0:
                return 1
            # handle special cases
            if numNegative == 1 and numPositive == 1:
                Nstar = self.N - self.qF + self.a - 1
                Fstar = w / (Nstar * (self.H1 - w))
                if lastPositiveNoncentrality >= 0 and lastNegativeNoncentrality == 0:
                    return probf(fcrit=Fstar,
                                 df1=Nstar,
                                 df2=1,
                                 noncen=lastPositiveNoncentrality)[0]
                elif lastPositiveNoncentrality == 0 and lastNegativeNoncentrality > 0:
                    # print("fcrit", 1 / Fstar)
                    # print("df1", 1)
                    # print("df2", Nstar)
                    # print("omega", lastNegativeNoncentrality)
                    # print("RUN _PROBF(PROB, FMETHOD, ", 1 / Fstar,", ",1,", ",Nstar,", ",lastNegativeNoncentrality,");")
                    prob, method = probf(fcrit=1 / Fstar,
                          df1=1,
                          df2=Nstar,
                          noncen=lastNegativeNoncentrality)
                    return 1 - prob
            if self.exact:
                dist = WeightedSumOfNoncentralChiSquaresDistribution(chiSquareTerms, 0.0, 0.001)
                return dist.cdf(0)
            else:
                # handle general case - Satterthwaite approximation
                nuStarPositive = 2 * (m1Positive * m1Positive) / m2Positive
                nuStarNegative = 2 * (m1Negative * m1Negative) / m2Negative
                lambdaStarPositive = m2Positive / (2 * m1Positive)
                lambdaStarNegative = m2Negative / (2 * m1Negative)

                # create a central F to approximate the distribution of the non-centrality parameter
                # return power based on the non-central F
                if isinstance(nuStarPositive, complex):
                    nuStarPositive = nuStarPositive.real
                if isinstance(nuStarNegative, complex):
                    nuStarNegative = nuStarNegative.real
                if isinstance(lambdaStarPositive, complex):
                    lambdaStarPositive = lambdaStarPositive.real
                if isinstance(lambdaStarNegative, complex):
                    lambdaStarNegative = lambdaStarNegative.real
                x = (nuStarNegative * lambdaStarNegative) / (nuStarPositive * lambdaStarPositive)
                return f.cdf(x, nuStarPositive, nuStarNegative)
        except GlimmpseCalculationException as e:
            print("exiting cdf abnormally", e)
            raise GlimmpseCalculationException(e)

    def inverseCDF(self, quantile):
        """ generated source for method inverseCDF """
        if self.H1 <= 0:
            return 0
        quantFunc = lambda n: quantile - self.cdf(n)
        try:
            return optimize.bisect(quantFunc, self.H0, self.H1)
        except GlimmpseCalculationException as e:
            raise GlimmpseCalculationException("Failed to determine non-centrality quantile: " + e.args[0])

    def NonCentralityQuantileFunction(self, quantile):
        """ generated source for class NonCentralityQuantileFunction """
        try:
            return self.cdf(n) - quantile
        except GlimmpseCalculationException as pe:
            raise GlimmpseCalculationException(pe, pe)

    def getSigmaStarInverse(self, sigma_star, test):
        """ generated source for method getSigmaStarInverse """
        if not self.isPositiveDefinite(sigma_star):
            self.errors.append(Constants.ERR_NOT_POSITIVE_DEFINITE)
        if test == Constants.HLT or test == Constants.HLT.value or test.value == Constants.HLT.value:
            return np.linalg.inv(sigma_star)
        else:
            # stat should only be UNIREP (uncorrected, box, GG, or HF) at this point
            # (exception is thrown by valdiateParams otherwise)
            b = sigma_star.shape[1]
            # get discrepancy from sphericity for unirep test
            sigmaStarTrace = np.trace(sigma_star)
            sigmaStarSquaredTrace = np.trace(sigma_star * sigma_star)
            epsilon = (sigmaStarTrace * sigmaStarTrace) / (b * sigmaStarSquaredTrace)
            identity = np.identity(b)
            return identity * float(b) * epsilon / sigmaStarTrace

    def getH1(self):
        """ generated source for method getH1 """
        return self.H1

    def getH0(self):
        """ generated source for method getH0 """
        return self.H0

    def isPositiveDefinite(self, m: np.matrix):
        """generated source for method isPositiveDefinite"""
        if m.shape[0] != m.shape[1]:
            raise GlimmpseValidationException("Matrix must be non-null, square")
        # get the eigenvalues of m using a singular value decomposition
        # m is an array of dimension 1 x b
        eigenvalues = np.linalg.svd(m, full_matrices=False, compute_uv=False, hermitian=True)
        test = [val > 0.0 for val in eigenvalues]
        return all(test)

    def forceSymmetric(self, m: np.matrix):
        """generated source for method forceSymmetric"""
        return np.tril(m) + np.triu(m.T, 1)

    def __unconditional_power_simpson_term(self, fcrit, df1, df2, t):
        """
        calculate the integration performed in glueck and muller 200?? eq ??
        """
        # check bounds H0 ,H1
        if self.H1 < self.H0:
            raise GlimmpseValidationException("H0 is greater than H1")
        elif round(self.H1, 12) == round(self.H0, 12):
            return 0
        else:
            t1 = special.ncfdtr(df1,df2,t,fcrit)
            t2_fcrit = (fcrit * df1)/(df1+2)
            t2 = special.ncfdtr(df1+2,df2,t,t2_fcrit)
            return self.cdf(t) * (t1 - t2)

    def unconditional_power_simpson(self, fcrit, df1, df2):
        """
        Calculates unconditional power using integration by simpsons rule.
        """
        y = lambda x: self.__unconditional_power_simpson_term(fcrit=fcrit, df1=df1, df2=df2, t=x)
        bounds = [self.H0, self.H1]

        t1 = special.ncfdtr(df1, df2, self.H1, fcrit)

        # set up properties for integration by Simpson's rule
        n = 2
        old_prob = 1
        max_iterations = math.pow(64, 2)

        # start iteration
        end_condition = False
        while not end_condition:
            h = (self.H1 - self.H0) / n
            x = []
            fx = []
            for i in range(n):
                x.append(self.H0 + i * h)
                fx.append(y(x[i]))

            res = 0
            for i in range(n):
                if i == 0 or i == n:
                    res += fx[i]
                elif i % 2 != 0:
                    res += 4 * fx[i]
                else:
                    res += 2 * fx[i]

            res = res * (h/3)
            t2 = res/2
            prob = t1 + t2

            #check delta
            if n >= max_iterations:
                end_condition = True
            delta = math.fabs(prob - old_prob)
            r_limit = math.pow(10, -6) * (math.fabs(prob) + math.fabs(old_prob)) * 0.5
            if (delta <= r_limit) or (delta <= math.pow(10, -6)):
                end_condition = True
            old_prob = prob
            n = n * 2


        return prob, None


