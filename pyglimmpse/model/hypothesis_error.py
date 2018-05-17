import numpy as np

class HypothesisError:
    """
        Come up with a decent docstring describing exactly what these are
    """

    def __init__(self, hypo_sum_square, sigma_star, rank_u):
        """
        Fill in Al

        :param hypo_sum_square:
        :param sigma_star:
        :param rank_U:
        """
        self.q1 = np.trace(sigma_star)
        self.q2 = np.trace(hypo_sum_square)
        self.q3 = self.q1 ** 2
        self.q4 = np.sum(np.power(sigma_star, 2))
        self.q5 = np.trace(sigma_star * hypo_sum_square)
        self.lambar = self.q1 / rank_u