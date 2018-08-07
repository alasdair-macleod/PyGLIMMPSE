import numpy as np
import sys

from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power
from scipy import optimize


def samplesize(test, rank_C, rank_U, alpha, sigmaScale, sigma,  betaScale, beta, targetPower, rank_X, eval_HINVE):
    """
    Gets samplesize required for the requested target power.
    :param test:
    :param alpha:
    :param sigmaScale:
    :param betaScale:
    :param targetPower:
    :return:
    """

    # scale beta and sigma matrices
    # TODO: how to include this in the finished product?
    scaled_beta = beta * betaScale
    scaled_sigma = sigma * sigmaScale

    # calculate max valid per group N
    max_n = min(sys.maxsize/rank_X, Constants.MAX_SAMPLE_SIZE)

    # calculate the noncentrality distribution

    # find a samplesize which produces power greater than or equal to the desired power
    upper_bound = Constants.STARTING_SAMPLE_SIZE
    upper_power = Power()
    while (np.isnan(upper_power.power) or upper_power.power <= targetPower) and upper_bound < max_n:
        upper_bound += upper_bound

        if upper_bound >= max_n:
            upper_bound = max_n

        total_N = upper_bound

        # call power for this sample size
        upper_power.power = test(sigma, rank_U, total_N, rank_X)

    # note we are using floor division
    lower_bound = upper_bound//2 + 1
    lower_power = test(sigma, rank_U, total_N, rank_X)

    #
    # At this point we have valid boundaries for searching.
    # There are two possible scenarios
    # 1. The upper bound == lower bound.
    # 2. The upper bound != lower bound and lower bound exceeds required power.
    # In this case we just take the value at the lower bound.
    # 3. The upper bound != lower bound and lower bound is less than the required power.
    # In this case we bisection search
    #
    if lower_power == upper_power.power:
        return lower_bound
    elif lower_power >= targetPower:
        total_N = lower_bound
    else:
        f = lambda samplesize: test(rank_C, rank_U, rank_X, samplesize, eval_HINVE, alpha) - targetPower
        total_N = optimize.bisect(f, lower_bound, upper_bound)

    return total_N
