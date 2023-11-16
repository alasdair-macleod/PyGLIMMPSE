import numpy as np

from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException


class Epsilon:
    """
    This class produces matrices required for Geisser-Greenhouse,
    Huynh-Feldt or uncorrected repeated measures power calculations. It
    is the first step. Program uses approximations of expected values of
    epsilon estimates due to Muller (1985), based on theorem of Fujikoshi
    (1978). Program requires that U be orthonormal and orthogonal to a
    columns of 1's.

    Epsilon is a measure of dispersion.
    scale free measure of heterogeneity of the eigenvalues of the covariance
    U' * SIGMA * U = SIGMA_STAR
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
            raise GlimmpseValidationException("rank of U should equal to nrows of sigma_star")

        # Get eigenvalues of covariance matrix associated with E. This is NOT
        # the USUAL sigma. This cov matrix is that of (Y-YHAT)*U, not of (Y-YHAT).
        # The covariance matrix is normalized to minimize numerical problems
        self.esig = sigma_star / np.trace(sigma_star)
        # get the eigenvalues of self.esig using a singular value decomposition
        seigval = np.linalg.svd(self.esig, full_matrices=False, compute_uv=False, hermitian=True)
        deigval_array, mtp_array = np.unique(seigval, return_counts=True)
        self.slam1 = np.sum(seigval) ** 2
        self.slam2 = np.sum(np.square(seigval))
        self.slam3 = np.sum(seigval)
        self.eps = self.slam1 / (rank_U * self.slam2)
        self.d = len(deigval_array)
        self.deigval = np.matrix(deigval_array).T
        self.mtp = np.matrix(mtp_array).T

    def esigEvals(self):
        # get the eigenvalues of self.esig using a singular value decomposition
        # seval is an array of dimension 1 x b
        seval = np.matrix(np.linalg.svd(self.esig, full_matrices=False, compute_uv=False, hermitian=True)).T
        esigEvals = np.sum(seval * seval.T)
        return esigEvals
