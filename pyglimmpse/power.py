import numpy as np

class Power:
    """
    Object representing power and associated metadata returned by power calculations.
    """
    def __init__(self):
        self.power = 0
        self.noncentrality_parameter = 0
        self.fmethod = 'DEFAULT'
        self.lowerBound = Power()
        self.upperBound = Power()

class Epsilon:
    """
    This class produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.
    """

    def __init__(self, sigma_star, rank_U):
        """
        :param sigma_star: U` * (SIGMA # SIGSCALTEMP) * U
        :param rank_U: rank of U matrix

        d, number of distinct eigenvalues
        mtp, multiplicities of eigenvalues
        eps, epsilon calculated from U`*SIGMA*U
        deigval, first eigenvalue
        slam1, sum of eigenvalues squared
        slam2, sum of squared eigenvalues
        slam3, sum of eigenvalues
        """
        if rank_U != np.shape(sigma_star)[0]:
            raise Exception("rank of U should equal to nrows of sigma_star")

        # Get eigenvalues of covariance matrix associated with E. This is NOT
        # the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
        # The covariance matrix is normalized to minimize numerical problems
        esig = sigma_star / np.trace(sigma_star)
        seigval = np.linalg.eigvals(esig)
        deigval_array, mtp_array = np.unique(seigval, return_counts=True)
        self.slam1 = np.sum(seigval) ** 2
        self.slam2 = np.sum(np.square(seigval))
        self.slam3 = np.sum(seigval)
        self.eps = self.slam1 / (rank_U * self.slam2)
        self.d = len(deigval_array)
        self.deigval = np.matrix(deigval_array).T
        self.mtp = np.matrix(mtp_array).T