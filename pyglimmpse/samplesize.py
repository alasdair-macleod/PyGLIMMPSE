import math

import numpy as np
import inspect
import sys

from pyglimmpse.constants import Constants
from pyglimmpse.model.power import Power, subtrtact_target_power
from scipy import optimize
from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException


def samplesize(test,
               rank_C: float,
               rank_X: float,
               relative_group_sizes,
               alpha: float,
               sigma_star: np.matrix,
               delta_es: np.matrix,
               targetPower,
               starting_smallest_group_size=Constants.STARTING_SAMPLE_SIZE.value,
               **kwargs):
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
    upper_power = Power()
    lower_power = Power()
    lowest_realizeable_power = Power()
    lowest_realizeable_total_N = 0
    smallest_group_size = starting_smallest_group_size
    upper_bound_smallest_group_size = starting_smallest_group_size
    upper_bound_total_N = upper_bound_smallest_group_size * sum(relative_group_sizes)

    smallest_design_found = False

    # find a samplesize which produces power greater than or equal to the desired power
    while (np.isnan(upper_power.power) or upper_power.power <= targetPower)\
            and upper_bound_total_N < max_n:

        upper_bound_total_N = upper_bound_smallest_group_size * sum(relative_group_sizes)
        if upper_bound_total_N >= max_n:
            upper_bound_smallest_group_size = upper_bound_total_N/max_n

        # call power for this sample size
        try:
            upper_power = test(rank_C=rank_C,
                               rank_X=rank_X,
                               relative_group_sizes=relative_group_sizes,
                               rep_N=upper_bound_smallest_group_size,
                               alpha=alpha,
                               sigma_star=sigma_star,
                               delta_es=delta_es)
            if type(upper_power.power) is str:
                raise ValueError('Upper power is not calculable. Check that your design is realisable.'
                                 ' Usually the easies way to do this is to increase sample size')
        except GlimmpseValidationException as e:
            upper_power.power == np.nan
        upper_bound_smallest_group_size += upper_bound_smallest_group_size
        if not np.isnan(upper_power.power) and not smallest_design_found:
            lowest_realizeable_power = upper_power
            lowest_realizeable_total_N = upper_bound_total_N
            smallest_design_found = True

    if lowest_realizeable_power.power >= targetPower:
        return lowest_realizeable_total_N, lowest_realizeable_power

    # find a samplesize for the per group n/2 + 1 to define the lower bound of our search.
    #undo last doubling
    if upper_power.power is None or math.isnan(upper_power.power):
        raise ValueError('Could not find a samplesize which achieves the target power. Please check your design.')

    upper_bound_smallest_group_size = upper_bound_smallest_group_size / 2
    # note we are using floor division
    lower_bound_smallest_group_size = upper_bound_smallest_group_size//2
    lower_power = test(rank_C=rank_C,
                       rank_X=rank_X,
                       relative_group_sizes=relative_group_sizes,
                       rep_N=upper_bound_smallest_group_size//2,
                       alpha=alpha,
                       sigma_star=sigma_star,
                       delta_es=delta_es)

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
        total_N = lower_bound_smallest_group_size * sum(relative_group_sizes)
        power = lower_power
    else:
        f = lambda n: subtrtact_target_power(test(rank_C=rank_C,
                                                  rank_X=rank_X,
                                                  relative_group_sizes=relative_group_sizes,
                                                  rep_N=n,
                                                  alpha=alpha,
                                                  sigma_star=sigma_star,
                                                  delta_es=delta_es,
                                                  **kwargs),
                                             targetPower)

        total_per_group_n = math.floor(optimize.bisect(f, lower_bound_smallest_group_size, upper_bound_smallest_group_size))
        power = test(rank_C=rank_C,
                     rank_X=rank_X,
                     relative_group_sizes=relative_group_sizes,
                     rep_N=total_per_group_n,
                     alpha=alpha,
                     sigma_star=sigma_star,
                     delta_es=delta_es,
                     **kwargs)

        if (power.power < targetPower) or np.isnan(power.power):
            total_per_group_n = total_per_group_n + 1
            power = test(rank_C=rank_C,
                         rank_X=rank_X,
                         relative_group_sizes=relative_group_sizes,
                         rep_N=total_per_group_n,
                         alpha=alpha,
                         sigma_star=sigma_star,
                         delta_es=delta_es,
                         **kwargs)
            if power.power < targetPower:
                raise ValueError('Samplesize cannot be calculated. Please check your design.')
        total_N = sum([math.ceil(total_per_group_n) * g for g in relative_group_sizes])
    return total_N, power

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