#!/usr/bin/env python

import numpy as np
from scipy.optimize import optimize
from scipy.stats import f

from pyglimmpse import constants
from pyglimmpse.WeightedSumOfNoncentralChiSquaresDistribution import WeightedSumOfNoncentralChiSquaresDistribution
from pyglimmpse.constants import Constants
from pyglimmpse.probf import probf

""" generated source for module NonCentralityDistribution """
# 
#  * Java Statistics.  A java library providing power/sample size estimation for
#  * the general linear model.
#  *
#  * Copyright (C) 2010 Regents of the University of Colorado.
#  *
#  * This program is free software; you can redistribute it and/or
#  * modify it under the terms of the GNU General Public License
#  * as published by the Free Software Foundation; either version 2
#  * of the License, or (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# 
#  * Class representing the distribution of the non-centrality parameter in
#  * the general linear multivariate model.  Used by the GLMMPowerCalculator class
#  * for computing unconditional and quantile power.
#  *
#  * @see edu.cudenver.bios.power.GLMMPowerCalculator
#  * @author Sarah Kreidler
#
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

    #  cache input parameters - needed for dynamic reset of sample size and beta matrix
    #test = Test()
    # FEssence = np.matrix()
    # FtFinverse = np.matrix()
    # perGroupN = int()
    # CFixed = np.matrix()
    # CRand = np.matrix()
    # U = np.matrix()
    # thetaNull = np.matrix()
    # beta = np.matrix()
    # sigmaError = np.matrix()
    # sigmaG = np.matrix()

    # 
    #      * Function calculating the difference between the probability of a target quantile
    #      * and the  (used by the bisection solver from Apache Commons Math)
    #      * @see org.apache.commons.math.analysis.UnivariateRealFunction
    #      
    class NonCentralityQuantileFunction():
        """ generated source for class NonCentralityQuantileFunction """
        quantile = float()

        def __init__(self, quantile):
            """ generated source for method __init__ """
            self.quantile = quantile

        def value(self, n):
            """ generated source for method value """
            try:
                return self.cdf(n) - self.quantile
            except Exception as pe:
                raise Exception(pe.getMessage(), pe)

    # 
    #      * Create a non-centrality distribution for the specified inputs.
    #      * @param params GLMM input parameters
    #      * @param exact if true, Davie's algorithm will be used to compute the cdf,
    #      * otherwise a Satterthwaite style approximation is used.
    #      * @throws IllegalArgumentException
    #      
    def __init__(self, test, FEssence, FtFinverse, perGroupN, CFixed, CRand, U, thetaNull, beta, sigmaError, sigmaG, exact):
        """ generated source for method __init__ """
        print("CREATING NonCentralityDistribution")
        print("begin parameters")
        print("test: " + test)
        print("FEssence:", FEssence)
        print("FtFinverse:", FtFinverse)
        print("perGroupN: ", perGroupN)
        print("CFixedRand:", CFixed + CRand)
        print("U:", U)
        print("thetaNull:", thetaNull)
        print("beta:", beta)
        print("sigmaError:", sigmaError)
        print("sigmaG:", sigmaG)
        print("exact: ", exact)
        print("end parameters")
        self.initialize(
            test=test,
            FEssence=FEssence,
            FtFinverse=FtFinverse,
            perGroupN=perGroupN,
            CFixed=CFixed,
            CRand=CRand,
            U=U,
            thetaNull=thetaNull,
            beta=beta,
            sigmaError=sigmaError,
            sigmaG=sigmaG,
            exact=exact)

    # 
    #      * Pre-calculate intermediate matrices, perform setup, etc.
    #      
    def initialize(self, test, FEssence, FtFinverse, perGroupN, CFixed, CRand, U, thetaNull, beta, sigmaError, sigmaG, exact):
        """ generated source for method initialize """
        print("entering initialize")
        #  reset member variables
        self.T1 = None
        self.FT1 = None
        self.S = None
        self.mzSq = None
        self.H0 = 0
        self.sStar = 0
        #  cache inputs
        self.test = test
        self.FEssence = FEssence
        self.FtFinverse = FtFinverse
        self.perGroupN = perGroupN
        self.CFixed = CFixed
        self.CRand = CRand
        self.U = U
        self.thetaNull = thetaNull
        self.beta = beta
        self.sigmaError = sigmaError
        self.sigmaG = sigmaG
        #  calculate intermediate matrices
        #         RealMatrix FEssence = params.getDesignEssence().getFullDesignMatrixFixed();
        #  TODO: do we ever get here with values that can cause integer overflow,
        #        and if so, does it matter?
        self.N = float(FEssence.shape[0]) * perGroupN
        self.exact = exact
        try:
            #  TODO: need to calculate H0, need to adjust H1 for Unirep
            #  get design matrix for fixed parameters only
            #self.qF = FEssence.getColumnDimension()
            self.qF = FEssence.shape[1]
            #  a = CFixedRand.getCombinedMatrix().getRowDimension();
            #  get fixed contrasts
            Cfixed = self.CFixed
            CGaussian = CRand

            #  build intermediate terms h1, S
            if FtFinverse == None:
                FtFinverse = np.linalg.inv(FEssence.transpose().multiply(FEssence))
                print("FEssence", FEssence)
                print("FtFinverse = (FEssence transpose * FEssence) inverse", FtFinverse)
            else:
                print("FtFinverse", FtFinverse)
            PPt = Cfixed * self.FtFinverse * (1 / self.perGroupN) * Cfixed.T
            print("Cfixed", Cfixed)
            print("n = ", self.perGroupN)
            print("PPt = Cfixed * FtF inverse * (1/n) * Cfixed transpose", PPt)

            self.T1 = self.forceSymmetric(np.linalg.inv(PPt))
            print("T1 = PPt inverse", self.T1)

            self.FT1 = np.linalg.cholesky(self.T1)
            print("FT1 = Cholesky decomposition (L) of T1", self.FT1)
            #calculate theta difference
            C = CFixed + CRand
            thetaHat = C * beta * U
            print("C", C)
            print("beta", beta)
            print("U", U)
            print("thetaHat = C * beta * U", thetaHat)

            thetaDiff = thetaHat - thetaNull
            print("thetaNull", thetaNull)
            print("thetaDiff = thetaHat - thetaNull", thetaDiff)

            #TODO: specific to HLT or UNIREP
            sigmaStarInverse = self.getSigmaStarInverse(sigmaError, test)
            print("sigmaStarInverse", sigmaStarInverse)

            H1matrix = thetaDiff.T * self.T1 * thetaDiff * sigmaStarInverse
            print("H1matrix = thetaDiff transpose * T1 * thetaDiff * sigmaStarInverse", H1matrix)
            self.H1 = np.trace(H1matrix)
            print("H1 = ", self.H1)

            # Matrix which represents the non-centrality parameter as a linear combination of chi-squared r.v.'s.
            self.S = self.FT1.T * thetaDiff * sigmaStarInverse * thetaDiff.T * self.FT1 * (1 / self.H1)
            print("S = FT1 transpose * thetaDiff * sigmaStar inverse * thetaDiff transpose * FT1 * (1/H1)", self.S)

            # We use the S matrix to generate the F-critical, numerical df's, and denominator df's
            # for a central F distribution.  The resulting F distribution is used as an approximation
            # for the distribution of the non-centrality parameter.
            # See formulas 18-21 and A8,A10 from Glueck & Muller (2003) for details.
            # sEigenDecomp = EigenDecomposition(self.S)
            # self.sEigenValues = sEigenDecomp.getRealEigenvalues()
            self.sEigenValues, svecs = np.linalg.eig(self.S)
            svec = np.matrix(svecs).T

            if len(self.sEigenValues) > 0:
                self.H0 = self.H1 * (1 - self.sEigenValues[0])
            if self.H0 <= 0:
                self.H0 = 0

            for value in self.sEigenValues:
                if value > 0:
                    self.sStar += 1
            # TODO: throw error if sStar is <= 0
            # TODO: NO: throw error if sStar != sEigenValues.length instead???

            stddevG = np.sqrt(sigmaG[0, 0])
            # svec = sEigenDecomp.getVT()
            # get eigenvectors
            # create square matrix using these

            self.mzSq = svec * self.FT1.T * CGaussian * (1 / stddevG)
            i = 0
            while i < self.mzSq.shape[0]:
                j = 0
                #while j < self.mzSq.getColumnDimension():
                while j < self.mzSq.shape[1]:
                    entry = self.mzSq[i, j]
                    self.mzSq[i, j] = entry * entry
                    j += 1
                i += 1
                print("exiting initialize normally")
        except Exception as e:
            #self.LOGGER.warn("exiting initialize abnormally", e)
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
            chiSquareTerms.add(ChiSquareTerm(lambda_, nu, delta))
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
                    delta = self.mzSq.getEntry(k, 0)
                    chiSquareTerms.add(ChiSquareTerm(lambda_, nu, delta))
                else:
                    # for k = sStar+1 to a, chi-sqaure term is non-central (delta = mz^2), 1 df,
                    # lambda = b0
                    nu = 1
                    lambda_ = b0
                    delta = self.mzSq.getEntry(k, 0)
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
                                 noncen=lastPositiveNoncentrality)
                elif lastPositiveNoncentrality == 0 and lastNegativeNoncentrality > 0:
                    return 1 - probf(fcrit=1 / Fstar,
                                     df1=1,
                                     df2=Nstar,
                                     noncen=lastNegativeNoncentrality)
            if self.exact:
                dist = WeightedSumOfNoncentralChiSquaresDistribution(chiSquareTerms, 0.0, 0.001)
                return dist.cdf(0)
            else:
                # handle general case - Satterthwaite approximation
                nuStarPositive = 2 * (m1Positive * m1Positive) / m2Positive;
                nuStarNegative = 2 * (m1Negative * m1Negative) / m2Negative;
                lambdaStarPositive = m2Positive / (2 * m1Positive);
                lambdaStarNegative = m2Negative / (2 * m1Negative);

                # create a central F to approximate the distribution of the non-centrality parameter
                # return power based on the non-central F
                x = (nuStarNegative * lambdaStarNegative) / (nuStarPositive * lambdaStarPositive)
                return f(x, nuStarPositive, nuStarNegative)
        except Exception as e:
            self.LOGGER.warn("exiting cdf abnormally", e)
            raise Exception(e.getMessage(), constants.DISTRIBUTION_NONCENTRALITY_PARAMETER_CDF_FAILED)

    def inverseCDF(self, probability):
        """ generated source for method inverseCDF """
        if self.H1 <= 0:
            return 0
        quantFunc = self.NonCentralityQuantileFunction(probability)
        try:
            return optimize.bisect(f=quantFunc,
                                   a=self.H0,
                                   b=self.H1,
                                   maxiter=self.MAX_ITERATIONS)
        except Exception as e:
            raise Exception("Failed to determine non-centrality quantile: " + e.getMessage())

    def getSigmaStarInverse(self, sigma_star, test):
        """ generated source for method getSigmaStarInverse """
        # sigmaStar = forceSymmetric(U.transpose().multiply(sigmaError).multiply(U))
        # print("U", U)
        # print("sigmaError", sigmaError)
        # print("sigmaStar = U transpose * sigmaError * U", sigmaStar)
        if not self.isPositiveDefinite(sigma_star):
            raise Exception("Sigma star is not positive definite.")
        if test == Constants.HLT:
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
            raise Exception("Matrix must be non-null, square")
        eigenvalues = np.linalg.eigvals(m)
        test = [val > 0.0 for val in eigenvalues]
        return all(test)

    def forceSymmetric(self, m: np.matrix):
        """generated source for method forceSymmetric"""
        return np.tril(m) + np.triu(m.T, 1)


