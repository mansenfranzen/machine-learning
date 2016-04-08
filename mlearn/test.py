"""This module contains simple functions for linear regressions."""

import numpy as np

def cost_function(x, y, theta):
    """Computes the cost for given matrix of x values (including :math:`x_0`)
    and vector of y values for specified theta.
    Cost is defined as the sum of the squared differences between predicted and
    true values:
    .. math:: J(\\theta) = \\frac{1}{2n} \\displaystyle\\sum_{i=1}^{n}( f_\\theta (x_i)-y_i)^2
    .. math:: f_\\theta (x_i)= \\theta_0 + \\theta_1 x_1 + ... + \\theta_i x_i = X^T \\theta
    Parameters
    ----------
    x : np.array
        Corresponds to matrix :math:`X`.
    y : np.array
        Corresponds to vector :math:`y`.
    theta : np.array
        Corresponds to vector :math:`\\theta`.
    Returns
    -------
    j : float
    """

    n = len(y)

    y_predicted = np.dot(x, theta)
    y_diff = y_predicted - y

    y_diff_squared = y_diff ** 2
    sum_score = sum(y_diff_squared)

    j = sum_score / (2*n)

    return j


if __name__ == "__main__":
    x = np.array([[1, 11, 104],
                  [1, 15, 99],
                  [1, 22, 89],
                  [1, 27, 88]])

    y = np.array([[12],
                  [15],
                  [19],
                  [22]])

    theta = np.zeros((3,1))

    print(cost_function(x, y, theta))
