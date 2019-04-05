#!/usr/bin/env python
import numpy as np

from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException, GlimmpseCalculationException

""" generated source for module WeightedSumOfNoncentralChiSquaresDistribution """
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
#  * Distribution of a weighted sum of noncentral chi square variables
#  * using Davie's algorithm,"Distribution of a linear combination of 
#  * non-central chi-squared random variables" (1980), Applied Statistics
#  * 
#  * @author Sarah Kreidler
#  *
#
#  simple counter class to maintain threadsafety for the iteration counter
#  sorting function for the lambdas
class MinMaxComparator(object):
    """ generated source for class MinMaxComparator """

    def compare(self, cs1, cs2):
        """ generated source for method compare """
        if cs1.getLambda() < cs2.getLambda():
            return -1
        elif cs1.getLambda() > cs2.getLambda():
            return 1
        else:
            return 0


#  rank sort for the lambdas
class RankOrderComparator(object):
    """ generated source for class RankOrderComparator """

    def compare(self, cs1, cs2):
        """ generated source for method compare """
        absLambda1 = np.abs(cs1)
        absLambda2 = np.abs(cs2)
        if absLambda1 < absLambda2:
            return -1
        elif absLambda1 > absLambda2:
            return 1
        else:
            return 0


#  container class for probability information
class TailProbabilityBound(object):
    """ generated source for class TailProbabilityBound """
    bound = float()
    cutoff = float()

    def __init__(self, bound, cutoff):
        """ generated source for method __init__ """
        self.bound = bound
        self.cutoff = cutoff

class Counter(object):
    """ generated source for class Counter """
    count = int()
    maxSteps = int()

    def __init__(self, maxSteps):
        """ generated source for method __init__ """
        self.count = 0
        self.maxSteps = maxSteps

    def increment(self):
        """ generated source for method increment """
        self.count += 1
        if self.count > self.maxSteps:
            raise GlimmpseCalculationException("Exceeded max iterations")

    def getCount(self):
        """ generated source for method getCount """
        return self.count

class WeightedSumOfNoncentralChiSquaresDistribution(object):
    """ generated source for class WeightedSumOfNoncentralChiSquaresDistribution """
    #  lambda coefficient on the normal term added onto the sum of chi-squares
    DEFAULT_GAUSSIAN_COEFFICIENT = 0.0

    #  default tau-sqaured term
    DEFAULT_TAU_SQUARED = 0.0

    #  maximum number of iterations allowed
    MAX_STEPS = 200000

    #  log(2) / 8 - no idea where this comes from, ask Davies
    LN_2_DIV_8 = np.log10(2.0) / 8.0

    #  parameters describing the weights (lambda), degrees of freedom (nu), 
    #  and non-centralities (omega) of the chi square variables
    chiSquareTerms = []

    #  map containing the ranks of the chi square terms (ranked by absolute value of lambda)
    chiSquareRankMap = {}

    #  min and max lambda values
    maxLambda = float()
    maxLambdaAbsValue = float()
    minLambda = float()

    #  maximum error allowed for the computed probability
    accuracy = float()

    #  coefficient of the normal term
    normalCoefficient = DEFAULT_GAUSSIAN_COEFFICIENT

    # 
    # 	 * Create a distribution for the specified weighted sum of non-central chi squares
    # 	 * 
    # 	 * @param chiSquareTerms list of weighted chi-square terms
    # 	 * @param normalCoefficient coefficient on the normal term
    # 	 * @param accuracy maximum error allowed in probability
    # 	 * @throws IllegalArgumentException
    #
    def __init__(self, chiSquareTerms, normalCoefficient, accuracy):
        """ generated source for method __init__ """
        if chiSquareTerms == None or len(chiSquareTerms) == 0:
            raise GlimmpseValidationException("No chi-square terms specified")
        if np.isnan(normalCoefficient):
            raise GlimmpseValidationException("Invalid coefficient for the normal term")
        if np.isnan(accuracy) or accuracy <= 0:
            raise GlimmpseValidationException("Accuracy must be greater than 0")
        self.chiSquareTerms = chiSquareTerms
        self.accuracy = accuracy
        self.normalCoefficient = normalCoefficient
        #  find the min/max lambda  (truncate min at 0)
        self.maxLambda = max(chiSquareTerms, key=lambda x: x.getLambda()).getLambda()
        self.minLambda = min(chiSquareTerms, key=lambda x: x.getLambda()).getLambda()
        self.maxLambdaAbsValue = (-self.minLambda if self.maxLambda < -self.minLambda else self.maxLambda)
        if self.maxLambda == 0 and self.minLambda == 0 and normalCoefficient == 0:
            raise GlimmpseValidationException("At least one of min/max lambda values or coefficient of the normal term must be non-zero")
        #  Build a list of ranks for chi squares based on the absolute value of the lambdas
        sortedList = []
        for chiSquare in self.chiSquareTerms:
            sortedList.append(chiSquare.getLambda())
        sortedList.sort(key=lambda absLambda: np.abs(absLambda))
        rank = 0
        for val in sortedList:
            self.chiSquareRankMap[val] = rank
            rank += 1

    # 
    # 	 * Create a distribution for the specified weighted sum of non-central chi squares
    # 	 * assuming a 0 as the coefficient on the normal term
    # 	 * 
    # 	 * @param chiSquareTerms list of weighted chi-square terms
    # 	 * @param accuracy maximum error allowed in probability
    # 	 * @throws IllegalArgumentException
    # 	 
    #@__init__.register(object, List, float)
    #def __init___0(self, chiSquareTerms, accuracy):
    #    """ generated source for method __init___0 """
    #    self.__init__(chiSquareTerms, self.DEFAULT_GAUSSIAN_COEFFICIENT, accuracy)

    # 
    # 	 * Returns the cdf of the specified weighted sum of chi-squares distribution 
    # 	 * at the given quantile.
    # 	 * 
    # 	 * @param quantile point at which the cdf is evaluated: Pr(Q &lt;= quantile)
    # 	 * @return probability Pr(Q &lt;= quantile)
    # 	 
    def cdf(self, quantile):
        """ generated source for method cdf """
        prob = 0
        #  expected value of ???
        mean = 0
        #  initialize sigma
        sigmaSquared = self.normalCoefficient * self.normalCoefficient
        #  initialize the standard deviation of the normal term
        sd = sigmaSquared
        #  convergence factor
        tauSquared = float()
        #  
        #  sum over something TODO: what?
        for chiSquare in self.chiSquareTerms:
            lambda_ = chiSquare.getLambda()
            df = chiSquare.getDegreesOfFreedom()
            noncentrality = chiSquare.getNoncentrality()
            if not (df < 0 or noncentrality < 0):
                sd += lambda_ * lambda_ * (2 * df + 4 * noncentrality)
                mean += lambda_ * (df + noncentrality)
            else:
                pass
            #  TODO: throw runtime exception??
        #  if sd term is 0, all probability density is piled on 0.
        #  thus, we return prob 1 for quantile=0, 0 otherwise
        if sd == 0:
            if quantile != 0:
                return 0
            else:
                return 1
        sd = np.sqrt(sd)
        #  initialize an interation counter  
        counter = Counter(self.MAX_STEPS)
        halfAccuracy = 0.5 * self.accuracy
        U = self.findTruncationPoint(16 / sd, sigmaSquared, halfAccuracy, counter)
        if quantile != 0 or self.maxLambdaAbsValue > 0.07 * sd:
            #  check if a covergence factor is helpful
            convergenceFactor = self.calculateConvergenceFactor(quantile, counter)
            tauSquared = 0.25 * self.accuracy/convergenceFactor
            if self.calculateIntegrationError(U, sigmaSquared, tauSquared, counter) < 0.2 * self.accuracy:
                sigmaSquared += tauSquared
                U = self.findTruncationPoint(16 / sd, sigmaSquared, halfAccuracy, counter)
        #  Auxiliary integration loop

        print('sd:' + str(sd))
        print('mean:' + str(mean))
        print('U:' + str(U))
        numTermsMain = 0
        numTermsAux = 0
        integralSum = 0
        integrationLimit = 0
        integrationInterval = 0
        while True:
            #  find the range of the distribution, determine if the quantile falls outside 
            #  of the range
            cutoff = self.findCutoffPoint(4.5 / sd, mean, sigmaSquared, halfAccuracy, counter)
            cutoffDiffUpper = cutoff - quantile;
            print('cutoffdiffupper:' + str(cutoffDiffUpper))

            if cutoffDiffUpper < 0:
                return 1
            #  requested quantile past range
            #  get the lower cutoff
            cutoff = self.findCutoffPoint(-4.5 / sd, mean, sigmaSquared, halfAccuracy, counter)
            cutoffDiffLower = quantile - cutoff
            print('cutoffdifflower:' + str(cutoffDiffLower))
            if cutoffDiffLower < 0:
                return 0
            #  pick the larger potential integration interval
            integrationInterval = (cutoffDiffUpper if (cutoffDiffUpper > cutoffDiffLower) else cutoffDiffLower)
            integrationInterval = 2 * np.pi / integrationInterval
            #  calculate the #terms required for main and auxilliary integrations
            numTermsMain = U / integrationInterval
            numTermsAux = 3.0 / np.sqrt(halfAccuracy)
            integralSum = 0
            if numTermsMain > 1.5 * numTermsAux:
                #  perform the auxilliary integration, provided we have enough iterations left
                if numTermsAux > self.MAX_STEPS - counter.getCount():
                    raise GlimmpseCalculationException("Number of auxiliary integration terms exceeds max number of iteration steps allowed")
                integrationLimit = 2 * np.pi / self.integrationIntervalAux
                if integrationLimit <= np.abs(quantile):
                    tauSquared = self.lowerConvergenceFactor + self.upperConvergenceFactor
                    tauSquared = (self.accuracy / 3) / (1.1 * tauSquared)
                    self.accuracy *= 0.67
                    integralSum += self.integrate(int(np.round(numTermsAux)), self.integrationIntervalAux, quantile, tauSquared)
                    sigmaSquared += tauSquared
                    U = self.findTruncationPoint(U, sigmaSquared, 0.25 * self.accuracy, counter)
                    self.accuracy *= 0.75
            if not numTermsMain > 1.5 * numTermsAux and integrationLimit <= np.abs(quantile):
                break
        #  perform main integration
        if numTermsMain > self.MAX_STEPS - counter.getCount():
            raise GlimmpseCalculationException("Number of main integration terms exceeds max number of iteration steps allowed")
        integralSum += self.integrate(int(np.round(numTermsMain)), integrationInterval, quantile, np.NaN)
        prob = 0.5 - integralSum
        return prob

    # 
    # 	 * Integrate
    # 	 
    def integrate(self, numTerms, integrationInterval, quantile, TauSquared):
        """ generated source for method integrate """
        value = 0
        k = numTerms
        while k >= 0:
            U = (k + 0.5) * integrationInterval
            sum1 = -2 * U * quantile
            sum2 = np.abs(sum1)
            sum3 = -0.5 * self.normalCoefficient * self.normalCoefficient * U * U

            for chiSquare in self.chiSquareTerms:
                lambda_ = chiSquare.getLambda()
                df = chiSquare.getDegreesOfFreedom()
                noncentrality = chiSquare.getNoncentrality()
                X = 2 * lambda_ * U
                Y = X * X

                sum3 += -0.25 * df * self.computeLog(Y, True)
                Y = noncentrality * X / (1 + Y)
                Z = df * np.arctan(X) + Y
                sum1 += Z
                sum2 += np.abs(Z)
                sum3 += -0.5 * X * Y
            partialValue = (integrationInterval / np.pi) * np.exp(sum3) / U
            if not np.isnan(TauSquared):
                partialValue *= (1 - np.exp(-0.5 * TauSquared * U * U))
            sum1 = np.sin(0.5 * sum1) * partialValue
            sum2 = 0.5 * sum2 * partialValue
            #  TODO: return sum1 from auxiliary integration
            value += sum1
            k -= 1
        return value

    # 
    # 	 * Find the upper cutoff point for the integral such that P(Q > c) &lt; accuracy
    # 	 * for U > 0, and 
    # 	 * @param startCutoff
    # 	 * @param mean
    # 	 * @param sigmaSquared
    #      * @param acc
    # 	 * @param counter
    # 	 * @return
    # 	 
    def findCutoffPoint(self, startCutoff, mean, sigmaSquared, acc, counter):
        """ generated source for method findCutoffPoint """
        u2 = startCutoff
        u1 = 0
        c1 = mean
        c2 = float()
        rb = self.minLambda
        if u2 > 0:
            rb = self.maxLambda
        rb *= 2
        u = u2 / (1 + u2 * rb)
        bound = self.findTailProbabilityBound(u, sigmaSquared, counter)
        c2 = bound.cutoff
        while bound.bound > acc:
            u1 = u2
            c2 = bound.cutoff
            c1 = c2
            u2 *= 2
            u = u2 / (1 + u2 * rb)
            bound = self.findTailProbabilityBound(u, sigmaSquared, counter)
        u = (c1 - mean) / (bound.cutoff - mean)
        while u < 0.9:
            u = (u1 + u2) / 2
            bound = self.findTailProbabilityBound(u / (1 + u * rb), sigmaSquared, counter)
            if bound.bound > acc:
                u1 = u
                c1 = bound.cutoff
            else:
                u2 = u
                c2 = bound.cutoff
            u = (c1 - mean) / (c2 - mean)
        return c2

    # 
    # 	 * Bound on the tail probability for a specified cutoff
    # 	 
    def findTailProbabilityBound(self, startU, sigmaSquared, counter):
        """ generated source for method findTailProbabilityBound """
        counter.increment()
        U = startU
        cutoff = U * sigmaSquared
        sum = U * cutoff
        U *= 2
        for chiSquare in self.chiSquareTerms:

            lambda_ = chiSquare.getLambda()
            df = chiSquare.getDegreesOfFreedom()
            noncentrality = chiSquare.getNoncentrality()
            X = U * lambda_
            Y = 1 - X

            cutoff += chiSquare.lambda_ * (noncentrality / Y + df) / Y
            sum += noncentrality * ((X / Y) * (X / Y)) +df * ((X * X) / Y + self.computeLog(-X, False))
        bound = TailProbabilityBound(np.exp(-0.5 * sum), cutoff)
        return bound

    # 
    # 	 * Find truncation point (U) such that the integration error is within the 
    # 	 * user specified accuracy, and that the integration error of U/1.2 is 
    # 	 * greater than the accuracy
    # 	 * 
    # 	 * @param startU starting value for the truncation point
    # 	 
    def findTruncationPoint(self, startU, sigmaSquared, acc, counter):
        """ generated source for method findTruncationPoint """
        U = startU / 4
        divisors = [2.0, 1.4, 1.2, 1.1]
        #  find the minimum U such that integration error will be below
        #  the specified accuracy
        if self.calculateIntegrationError(U, sigmaSquared, self.DEFAULT_TAU_SQUARED, counter) <= acc:
            #  our first guess was too high, so decrease U until we hit 
            #  the minimum value with appropriate accuracy
            U /= 4
            while self.calculateIntegrationError(U, sigmaSquared, self.DEFAULT_TAU_SQUARED, counter) <= acc:
                Utemp = U
                U /= 4
            U = Utemp
        else:
            #  our first guess was too low, so increase U until we get
            #  good enough accuracy
            while True:
                U *= 4
                if not self.calculateIntegrationError(U, sigmaSquared, self.DEFAULT_TAU_SQUARED, counter) > acc:
                    break
        #  now ensure that the truncating error of U/1.2 > accuracy
        Utemp = U
        for divisor in divisors:
            Utemp /= divisor
            if self.calculateIntegrationError(Utemp, sigmaSquared, self.DEFAULT_TAU_SQUARED, counter) > acc:
                break
            U = Utemp
        return U

    # 
    # 	 * Calculate the tau convergence factor
    # 	 * @return convergence factor
    # 	 
    def calculateConvergenceFactor(self, quantile, counter):
        """ generated source for method calculateConvergenceFactor """
        counter.increment()
        #  sign of the quantile (1, 0, or -1)
        quantileSign = 0
        if quantile > 0:
            quantileSign = 1
        elif quantile < 0:
            quantileSign = -1
        sum = 0
        axl = np.abs(quantile)
        j = len(self.chiSquareTerms) - 1
        while j >= 0:
            chiSquare = self.chiSquareTerms[j]
            rankIndex = self.chiSquareRankMap[chiSquare.getLambda()]
            rankChiSquare = self.chiSquareTerms[rankIndex]

            lambda_ = rankChiSquare.getLambda()
            lambdaAbsoluteValue = np.abs(lambda_)
            df = rankChiSquare.getDegreesOfFreedom()
            nonCentrality = rankChiSquare.getNoncentrality()

            if lambda_ * quantileSign > 0:
                axl1 = axl - lambdaAbsoluteValue * (df + nonCentrality)
                axl2 = lambda_ / (np.log(2)/8)
                if axl1 > axl2:
                    axl = axl1
                else:
                    if axl > axl2:
                        axl = axl2
                    sum = (axl - axl1) / lambda_
                    k = j - 1
                    while k >= 0:
                        chiSquareK = self.chiSquareTerms[k]
                        rankIndexK = self.chiSquareRankMap[chiSquareK.getLambda()]
                        rankChiSquareK = self.chiSquareTerms[rankIndexK]
                        sum += rankChiSquareK.getDegreesOfFreedom() + rankChiSquareK.getNoncentrality()
                        k -= 1
                    break
            j -= 1
        # 		if (sum > 100)
        # 			throw new RuntimeException("Summation to big, bad roundoff error"); // TODO: do we need this check, another precision thing?
        # 		
        return np.power(2, sum / 4) / (np.pi * axl * axl)

    # 
    # 	 * Calculates the integration error associated with the given
    # 	 * truncation point, U.
    # 	 * 
    # 	 * @return integration error
    # 	 
    def calculateIntegrationError(self, U, sigmaSquared, tauSquared, counter):
        """ generated source for method calculateIntegrationError """
        counter.increment()
        sum1 = 0
        sum2 = (sigmaSquared + tauSquared) * U * U
        prod1 = 2 * sum2
        prod2 = 0
        prod3 = 0
        ns = 0
        X = 0
        U *= 2
        for chiSquare in self.chiSquareTerms:
            X = (U * chiSquare.lambda_) * (U * chiSquare.lambda_)
            sum1 += chiSquare.noncentrality * X / (1 + X)
            if X <= 1:
                prod1 += chiSquare.df * self.computeLog(X, True)
            else:
                prod2 += chiSquare.df * np.log(X)
                prod3 += chiSquare.df * self.computeLog(X, True)
                ns += chiSquare.df
        sum1 *= 0.5
        prod2 += prod1
        prod3 += prod1
        X = np.exp(-sum1 - 0.25 * prod2) / np.pi
        Y = np.exp(-sum1 - 0.25 * prod3) / np.pi
        #  now that we've accumulated terms, build the three forms of
        #  the truncation error
        error1 = float()
        error2 = float()
        if ns != 0:
            error1 = X * 2 / ns
        else:
            error1 = 1
        if prod3 > 1:
            error2 = 2.5 * Y
        else:
            error2 = 1
        #  take the smaller of the two errors
        if error2 < error1:
            error1 = error2
        X = 0.5 * sum2
        if X <= Y:
            error2 = 1
        else:
            error2 = Y / X
        return (error1 if error1 < error2 else error2)

    # 
    # 	 * convenience function for log
    # 	 
    def computeLog(self, x, first):
        """ generated source for method computeLog """
        if np.abs(x) > 0.1:
            return (np.log(1 + x) if first else (np.log(1 + x) - x))
        else:
            #  deal with values very close to 0
            #  2 * Y^3
            y = x / (2 + x)
            term = 2 * y * y * y
            ak = 3.0
            s = 2

            if not first:
                s = -x
            s *= y
            y *= y
            s1 = s + term / ak
            while s1 != s:
                ak = ak + 2
                term *= y
                s = s1
                s1 = s + term / ak
            return s

