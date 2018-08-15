import numpy as np
import inspect
import sys

from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power
from scipy import optimize


def samplesize(test, rank_C, rank_U, alpha, sigmaScale, sigma,  betaScale, beta, targetPower, rank_X, eval_HINVE=None, error_sum_square=None, hypothesis_sum_square=None, optional_args=None):
    """
    Gets samplesize required for the requested target power.
    :param test:
    :param rank_C:
    :param rank_U:
    :param alpha:
    :param sigmaScale:
    :param sigma:
    :param betaScale:
    :param beta:
    :param targetPower:
    :param rank_X:
    :param eval_HINVE:
    :param error_sum_square:
    :param hypothesis_sum_square:
    :param optional_args:
    :return:
    """

    # scale beta and sigma matrices
    # TODO: how to include this in the finished product?
    scaled_beta = beta * betaScale
    scaled_sigma = sigma * sigmaScale

    # calculate max valid per group N
    max_n = min(sys.maxsize/rank_X, Constants.MAX_SAMPLE_SIZE.value)

    # calculate the noncentrality distribution

    # find a samplesize which produces power greater than or equal to the desired power
    upper_bound = Constants.STARTING_SAMPLE_SIZE.value
    upper_power = Power()
    while (upper_power.power == np.NaN or upper_power.power <= targetPower) and upper_bound < max_n:
        upper_bound += upper_bound

        if upper_bound >= max_n:
            upper_bound = max_n

        total_N = upper_bound

        # call power for this sample size
        if len(inspect.signature(test).parameters) == 7:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               rank_X=rank_X,
                               total_N=total_N,
                               eval_HINVE=eval_HINVE,
                               alpha=alpha)
        elif len(inspect.signature(test).parameters) == 9:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               total_N=total_N,
                               rank_X=rank_X,
                               error_sum_square=error_sum_square,
                               hypo_sum_square=hypothesis_sum_square,
                               sigma_star=sigma,
                               alpha=alpha,
                               optional_args=optional_args)

    # note we are using floor division
    lower_bound = upper_bound//2 + 1
    if len(inspect.signature(test).parameters) == 7:
        lower_power = test(rank_C=rank_C,
                           rank_U=rank_U,
                           rank_X=rank_X,
                           total_N=total_N,
                           eval_HINVE=eval_HINVE,
                           alpha=alpha)
    elif len(inspect.signature(test).parameters) == 9:
        lower_power = test(rank_C=rank_C,
                           rank_U=rank_U,
                           total_N=total_N,
                           rank_X=rank_X,
                           error_sum_square=error_sum_square,
                           hypo_sum_square=hypothesis_sum_square,
                           sigma_star=sigma,
                           alpha=alpha,
                           optional_args=optional_args)

    #
    # At this point we have valid boundaries for searching.
    # There are two possible scenarios
    # 1. The upper bound == lower bound.
    # 2. The upper bound != lower bound and lower bound exceeds required power.
    # In this case we just take the value at the lower bound.
    # 3. The upper bound != lower bound and lower bound is less than the required power.
    # In this case we bisection search
    #
    if lower_power.power == upper_power.power:
        return lower_bound
    elif lower_power.power >= targetPower:
        total_N = lower_bound
    else:
        f = lambda samplesize: test(rank_C, rank_U, rank_X, samplesize, eval_HINVE, alpha) - targetPower
        total_N = optimize.bisect(f, lower_bound, upper_bound)

    return total_N
