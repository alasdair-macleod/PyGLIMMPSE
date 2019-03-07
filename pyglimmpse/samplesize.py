import math

import numpy as np
import inspect
import sys

from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power, subtrtact_target_power
from scipy import optimize


def samplesize(test,
               rank_C,
               rank_U,
               alpha,
               sigma_star,
               targetPower,
               rank_X,
               delta,
               relative_group_sizes,
               starting_smallest_group_size=Constants.STARTING_SAMPLE_SIZE.value,
               optional_args=None):
    """
    Get the smallest realizable samplesize for the requested target power.
    :param test: The statistical test chosen. This must be pne of the tests available in pyglimmpse.multirep or pyglimmpse.unirep
    :param rank_C: Rank of the within contrast matrix for your study design.
    :param rank_U: Rank of the between contrast matrix for your study design.
    :param alpha: Type one error rate
    :param sigma_star: Sigma star
    :param targetPower: The power you wish to achieve
    :param rank_X: the rank of Es(X). Where X is your design matrix.
    :param delta: (Theta - Theta_0)'M^-1(Theta-Theta_0)
    :param relative_group_sizes: a list of ratios of size of the groups in your design.
    :param starting_smallest_group_size: The starting point for our integration. If this is less than the minimum realizeable smallest group size for your design, this function will return an error.
    :param optional_args:
    :return:
    """

    # calculate max valid per group N
    max_n = min(sys.maxsize/rank_X, Constants.MAX_SAMPLE_SIZE.value)
    # declare variables prior to integration
    error_sum_square = None
    hypothesis_sum_square = None
    upper_power = Power()
    lower_power = Power()
    smallest_group_size = starting_smallest_group_size
    upper_bound_smallest_group_size = starting_smallest_group_size

    # find a samplesize which produces power greater than or equal to the desired power
    while (np.isnan(upper_power.power) or upper_power.power <= targetPower)\
            and upper_bound_smallest_group_size < max_n:
        upper_bound_total_N = sum([smallest_group_size * g for g in relative_group_sizes])

        if upper_bound_total_N >= max_n:
            upper_bound_total_N = max_n

        # recalculate error sum square and hypothesis sum square for upper bound total N
        # this must be recalculated for every samplesize before our power calculation
        error_sum_square = _calc_err_sum_square(upper_bound_total_N, rank_X, sigma_star)
        hypothesis_sum_square = _calc_hypothesis_sum_square(upper_bound_total_N, relative_group_sizes, delta)

        # call power for this sample size
        if len(inspect.signature(test).parameters) == 8:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               rank_X=rank_X,
                               total_N=upper_bound_total_N,
                               alpha=alpha,
                               error_sum_square=error_sum_square,
                               hypothesis_sum_square=hypothesis_sum_square)
        elif len(inspect.signature(test).parameters) == 9:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               total_N=upper_bound_total_N,
                               rank_X=rank_X,
                               error_sum_square=error_sum_square,
                               hypo_sum_square=hypothesis_sum_square,
                               sigma_star=sigma_star,
                               alpha=alpha,
                               optional_args=optional_args)
        if type(upper_power.power) is str:
            raise ValueError('Upper power is not calculable. Check that your design is realisable.'
                             ' Usually the easies way to do this is to increase sample size')
        smallest_group_size += smallest_group_size



    # find a samplesize which produces power greater than or equal to the desired power
    # find a samplesize for the per group n/2 + 1 to define the lower bound of our search.
    #undo last doubling
    smallest_group_size = smallest_group_size / 2
    # note we are using floor division
    lower_bound_total_N = sum([(smallest_group_size//2) * g for g in relative_group_sizes])

    # recalculate error sum square and hypothesis sum square for lower bound total N
    # this must be recalculated for every samplesize before our power calculation
    error_sum_square = _calc_err_sum_square(lower_bound_total_N, rank_X, sigma_star)
    hypothesis_sum_square = _calc_hypothesis_sum_square(lower_bound_total_N, relative_group_sizes, delta)

    if len(inspect.signature(test).parameters) == 8:
        lower_power = test(rank_C=rank_C,
                           rank_U=rank_U,
                           rank_X=rank_X,
                           total_N=lower_bound_total_N,
                           alpha=alpha,
                           error_sum_square=error_sum_square,
                           hypothesis_sum_square=hypothesis_sum_square)
    elif len(inspect.signature(test).parameters) == 9:
        lower_power = test(rank_C=rank_C,
                           rank_U=rank_U,
                           total_N=lower_bound_total_N,
                           rank_X=rank_X,
                           error_sum_square=error_sum_square,
                           hypo_sum_square=hypothesis_sum_square,
                           sigma_star=sigma_star,
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
    if lower_power.power >= targetPower:
        total_N = lower_bound_total_N
        power = lower_power
    else:
        f = None
        if len(inspect.signature(test).parameters) == 8:
            f = lambda n: subtrtact_target_power(test(rank_C=rank_C,
                                                      rank_U=rank_U,
                                                      rank_X=rank_X,
                                                      total_N=sum([n * g for g in relative_group_sizes]),
                                                      alpha=alpha,
                                                      error_sum_square=error_sum_square,
                                                      hypothesis_sum_square=hypothesis_sum_square), targetPower)
        elif len(inspect.signature(test).parameters) == 9:
            f = lambda n: subtrtact_target_power(test(rank_C=rank_C,
                                                      rank_U=rank_U,
                                                      total_N=sum([n * g for g in relative_group_sizes]),
                                                      rank_X=rank_X,
                                                      error_sum_square=error_sum_square,
                                                      hypo_sum_square=hypothesis_sum_square,
                                                      sigma_star=sigma_star,
                                                      alpha=alpha,
                                                      optional_args=optional_args), targetPower)
        total_per_group_n = optimize.bisect(f, smallest_group_size//2, smallest_group_size)
        total_per_group_n = math.ceil(total_per_group_n)
        total_N = sum([total_per_group_n * g for g in relative_group_sizes])

        # recalculate error sum square and hypothesis sum square for result total N
        # this must be recalculated for every samplesize before our power calculation
        error_sum_square = _calc_err_sum_square(total_N, rank_X, sigma_star)
        hypothesis_sum_square = _calc_hypothesis_sum_square(total_N, relative_group_sizes, delta)

        if len(inspect.signature(test).parameters) == 8:
            power = test(rank_C=rank_C,
                         rank_U=rank_U,
                         rank_X=rank_X,
                         total_N=total_N,
                         alpha=alpha,
                         error_sum_square=error_sum_square,
                         hypothesis_sum_square=hypothesis_sum_square)
        elif len(inspect.signature(test).parameters) == 9:
            power =test(rank_C=rank_C,
                        rank_U=rank_U,
                        total_N=total_N,
                        rank_X=rank_X,
                        error_sum_square=error_sum_square,
                        hypo_sum_square=hypothesis_sum_square,
                        sigma_star=sigma_star,
                        alpha=alpha,
                        optional_args=optional_args)

    if power.power < targetPower:
        total_N = sum([(total_per_group_n + 1) * g for g in relative_group_sizes])
        error_sum_square = _calc_err_sum_square(total_N, rank_X, sigma_star)
        hypothesis_sum_square = _calc_hypothesis_sum_square(total_N, relative_group_sizes, delta)

        if len(inspect.signature(test).parameters) == 8:
            power = test(rank_C=rank_C,
                         rank_U=rank_U,
                         rank_X=rank_X,
                         total_N=total_N,
                         alpha=alpha,
                         error_sum_square=error_sum_square,
                         hypothesis_sum_square=hypothesis_sum_square)
        elif len(inspect.signature(test).parameters) == 9:
            power = test(rank_C=rank_C,
                         rank_U=rank_U,
                         total_N=total_N,
                         rank_X=rank_X,
                         error_sum_square=error_sum_square,
                         hypo_sum_square=hypothesis_sum_square,
                         sigma_star=sigma_star,
                         alpha=alpha,
                         optional_args=optional_args)
        if power.power < targetPower:
            raise ValueError('Samplesize cannot be calculated. Please check your design.')
    return total_N, power.power

def _calc_err_sum_square(total_n, rank_x, sigma_star):
    """
    Calculate error sum of squares matrix = nu_e * sigma star

    :param total_n: total samplesize
    :param rank_x: rank of the design matrix X
    :param sigma_star: sigma star
    :return: error sums of squares matrix
    """
    nu_e = total_n - rank_x
    return nu_e * sigma_star

def _calc_hypothesis_sum_square(total_n, relative_group_sizes, delta):
    """
    Calculate hypothesis sum of squares matrix.

    :param total_n: total samplesize
    :param relative_group_sizes: A list of ratios of size of the groups in your design.
    :param delta: (Theta - Theta_0)'M^-1(Theta-Theta_0)
    :return: hypothesis sums of squares matrix
    """
    repeated_rows_in_design_matrix = total_n / sum([g for g in relative_group_sizes])
    return repeated_rows_in_design_matrix * delta