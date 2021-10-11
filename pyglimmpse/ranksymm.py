import numpy as np

from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException


def ranksymm(matrix: np.matrix, tolerance: float) -> np.matrix:
    """This function computes the rank of a square symmetric nonnegative definite matrix via eigenvalues.

        Parameters
        ----------
            matrix:
                Matrix for which rank is to be calculated
            tolerance:
                Value below which numbers are declared zero

        Returns
        -------
        rankmatrix:
            if MATRIX is not symmetric or positive definite return .
            else returns the rank of the matrix

    """
    # empty matrix
    if np.shape(matrix)[1] == 0:
        raise GlimmpseValidationException("Matrix {0} does not exist.".format(matrix))

    # number of rows not equal to number of columns
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        raise GlimmpseValidationException("Matrix {0} is not square.".format(matrix))

    # matrix with all missing values
    if np.isnan(matrix).all():
        raise GlimmpseValidationException("Matrix {0} is all missing values.".format(matrix))

    maxabsval = abs(matrix).max()

    # matrix with all zero
    if maxabsval == 0:
        raise GlimmpseValidationException("Matrix {0} has MAX(ABS(all elements)) = exact zero.".format(matrix))

    nmatrix = matrix / maxabsval
    # matrix not symmetric
    if abs(nmatrix - nmatrix.T).max() >= tolerance ** 0.5:
        raise GlimmpseValidationException("Matrix {0} is not symmetric within sqrt(tolerance).".format(matrix))
    # get the eigenvalues of nmatrix using a singular value decomposition
    # evals is an array of dimension 1 x b
    evals, vecs = np.linalg.eigh(nmatrix)

    # matrix not non-negative definite
    if evals.min() < -tolerance ** 0.5:
        raise GlimmpseValidationException("Matrix {0} is *NOT* non-negative definite (and has at \
              least one eigenvalue strictly less than \
              zero). This may happen due to programming \
              error or rounding error of a nearly LTFR \
              matrix. This may be able to be fixed using \
              usual scaling/centering techniques. The \
              Eigenvalues/MAX(ABS(original matrix)) are: {1}. \
              The max(abs(original matrix)) is {2}.".format(matrix, evals, maxabsval))

    rankmatrix = sum(evals >= tolerance)
    return rankmatrix
