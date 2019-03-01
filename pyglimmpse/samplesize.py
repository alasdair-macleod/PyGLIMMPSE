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
               m,
               t,
               groups,
               starting_smallest_group_size=Constants.STARTING_SAMPLE_SIZE.value,
               optional_args=None):
    """
    Gets samplesize required for the requested target power.
    :param test:
    :param rank_C:
    :param rank_U:
    :param alpha:
    :param sigma_star:
    :param targetPower:
    :param rank_X:
    :param starting_sample_size:
    :param optional_args:
    :return:
    """

    # calculate max valid per group N
    max_n = min(sys.maxsize/rank_X, Constants.MAX_SAMPLE_SIZE.value)

    # declare error_sum_square
    error_sum_square = None
    # declare hypothesis_sum_square
    hypothesis_sum_square = None

    starting_sample_size = 3

    test_smallest_group_size = starting_sample_size
    upper_power = Power()

    # TODO: consider the condition upper bound less than max_n
    upper_bound = starting_sample_size
    while (upper_power.power == np.NaN or upper_power.power <= targetPower) and upper_bound < max_n:
        #upper_bound += upper_bound

        # find a samplesize which produces power greater than or equal to the desired power
        upper_bound = sum([test_smallest_group_size * g for g in groups])

        if upper_bound >= max_n:
            upper_bound = max_n

        total_N = upper_bound

        #correct
        error_sum_square = (total_N - rank_X) * sigma_star
        #incorrect
        repeated_rows_in_design_matrix = total_N/sum([g for g in groups])
        hypothesis_sum_square = repeated_rows_in_design_matrix * np.transpose(t) * np.linalg.inv(m) * t

        # call power for this sample size
        if len(inspect.signature(test).parameters) == 8:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               rank_X=rank_X,
                               total_N=total_N,
                               alpha=alpha,
                               error_sum_square=error_sum_square,
                               hypothesis_sum_square=hypothesis_sum_square)
        elif len(inspect.signature(test).parameters) == 9:
            upper_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               total_N=total_N,
                               rank_X=rank_X,
                               error_sum_square=error_sum_square,
                               hypo_sum_square=hypothesis_sum_square,
                               sigma_star=sigma_star,
                               alpha=alpha,
                               optional_args=optional_args)
        if type(upper_power.power) is str:
            raise ValueError('Upper power is not calculable. Check that your design is realisable.'
                             ' Usually the easies way to do this is to increase sample size')
        test_smallest_group_size += test_smallest_group_size

    # note we are using floor division
    #lower_bound = upper_bound//2 + 1
    test_smallest_group_size = test_smallest_group_size / 2
    lower_bound = sum([(test_smallest_group_size//2) * g for g in groups])
    error_sum_square = (lower_bound - rank_X) * sigma_star
    repeated_rows_in_design_matrix = lower_bound / sum([g for g in groups])
    hypothesis_sum_square = repeated_rows_in_design_matrix * np.transpose(t) * np.linalg.inv(m) * t
    lower_power = Power()
    if len(inspect.signature(test).parameters) == 8:
            lower_power = test(rank_C=rank_C,
                               rank_U=rank_U,
                               rank_X=rank_X,
                               total_N=lower_bound,
                               alpha=alpha,
                               error_sum_square=error_sum_square,
                               hypothesis_sum_square=hypothesis_sum_square)
    elif len(inspect.signature(test).parameters) == 9:
        lower_power = test(rank_C=rank_C,
                           rank_U=rank_U,
                           total_N=lower_bound,
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
        total_N = lower_bound
        power = lower_power
    else:
        f = None
        if len(inspect.signature(test).parameters) == 8:
            f = lambda n: subtrtact_target_power(test( rank_C=rank_C,
                                                       rank_U=rank_U,
                                                       rank_X=rank_X,
                                                       total_N=sum([n * g for g in groups]),
                                                       alpha=alpha,
                                                       error_sum_square=error_sum_square,
                                                       hypothesis_sum_square=hypothesis_sum_square), targetPower)
        elif len(inspect.signature(test).parameters) == 9:
            f = lambda n: subtrtact_target_power(test(rank_C=rank_C,
                                                      rank_U=rank_U,
                                                      total_N=sum([n * g for g in groups]),
                                                      rank_X=rank_X,
                                                      error_sum_square=error_sum_square,
                                                      hypo_sum_square=hypothesis_sum_square,
                                                      sigma_star=sigma_star,
                                                      alpha=alpha,
                                                      optional_args=optional_args), targetPower)
        total_N = optimize.bisect(f, test_smallest_group_size//2, test_smallest_group_size)
        total_N = sum( [ math.ceil(total_N) * g for g in groups])
        error_sum_square = (total_N - rank_X) * sigma_star
        repeated_rows_in_design_matrix = total_N / sum([g for g in groups])
        hypothesis_sum_square = repeated_rows_in_design_matrix * np.transpose(t) * np.linalg.inv(m) * t


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
        raise ValueError('Samplesize cannot be calculated. Please check your design.')
    return total_N, power.power
