"""This module contains simple functions for linear regressions."""

import numpy as np

def sum_of_squared_residuals(x, y, beta):
    """
    Provide a measure for the difference between observed :math:`y` and
    predicted :math:`x^T*\\beta` data.

    This measure is also called the sum of the squared residuals (SSR).

    Parameters
    ----------
    x : np.array
        Corresponds to a matrix :math:`x` which contains all :math:`x_i` values
        (including :math:`x_0=1`).
    y : np.array
        Corresponds to vector :math:`y` which refers to the observed values.
    beta : np.array
        Corresponds to vector :math:`\\beta` which contains the parameters to
        calculate the predicted values.

    Returns
    -------
    ssr : numpy.array

    References
    --------
    .. [1] https://en.wikipedia.org/wiki/Residual_sum_of_squares

    Notes
    -----
    .. math:: SSR(\\beta) = \\displaystyle\\sum_{i=1}^{n}(x_i\\beta_i-y_i)^2
    .. math:: SSR(\\beta) = (x\\beta-y)^T (x\\beta-y)

    """

    diff_score = np.dot(x, beta) - y
    ssr = np.dot(diff_score.transpose(), diff_score)

    return ssr.ravel()

def ordinary_least_squares(x, y):
    """
    Analytically calculate the unknown parameters for a linear regression model
    which minimize the sum of the squared errors of observed and predicted data.

    Parameters
    ----------
    x : np.array
        Corresponds to a matrix :math:`x` which contains all :math:`x_i` values
        (including :math:`x_0=1`).
    y : np.array
        Corresponds to vector :math:`y` which refers to the observed values.

    Returns
    -------
    beta : numpy.array

    References
    --------
    .. [2] https://en.wikipedia.org/wiki/Ordinary_least_squares

    Notes
    -----
    .. math:: \\hat{\\beta} =  (x^Tx)^{-1}x^Ty

    """

    x_t = x.transpose()
    inverse = np.linalg.pinv(np.dot(x_t, x))
    beta = np.dot(inverse, np.dot(x_t, y))

    return beta

if __name__ == "__main__":
    x = np.array([[1, 11, 104],
                  [1, 15, 99],
                  [1, 22, 89],
                  [1, 27, 88]])

    y = np.array([[12],
                  [15],
                  [19],
                  [22]])

    beta = np.zeros((3,1))

    print(sum_of_squared_residuals(x, y, beta))
    print(ordinary_least_squares(x, y))

