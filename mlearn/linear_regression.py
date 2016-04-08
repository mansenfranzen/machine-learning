"""This module contains simple functions for linear regressions."""

import numpy as np

def sum_of_squared_residuals(x, y, beta):
    """
    Calculate the sum of squared residuals for observed :math:`y` values,
    given :math:`x` predictor matrix and provided :math:`\\beta` parameters.

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
    .. math:: SSR(\\beta_m) = \\displaystyle\\sum_{i=1}^{n}(x_i\\beta_m-y_i)^2
    .. math:: SSR(\\beta) = (x\\beta-y)^T (x\\beta-y)

    Examples
    --------
    >>> # The predictor matrix
    >>> x = np.array([[1, 11, 104],
                      [1, 15, 99],
                      [1, 22, 89],
                      [1, 27, 88]])
    >>> # The observed values vector
    >>> y = np.array([[12],
                      [15],
                      [19],
                      [22]])
    >>> # The parameters vector
    >>> beta_zero = np.zeros((3,1))
    >>> sum_of_squared_residuals(x, y, beta_zero)
    [ 1214.]

    """

    diff_score = np.dot(x, beta) - y
    ssr = np.dot(diff_score.transpose(), diff_score)

    return ssr.ravel()

def gradient_descent(x, y, beta_init=None, gamma=1, max_iter=200,
                     threshold=0.01, scaling=True, regularize=False):
    """
    Numerically estimates the unknown parameters :math:`\\beta_i` for a linear
    regression model where :math:`x` refers to the predictor matrix and and
    :math:`y` to the observed values vector.

    The first derivative of the sum of the squared residuals is used to
    calculate the gradient for each parameter. In every iteration, the parameter
    is changed in decreasing direction of the gradient with the given step size
    :math:`\\gamma`. This is done until the maximum iteration amount is reached
    or the difference of sum of squared residuals between two iterations falls
    below a given threshold.

    Parameters
    ----------
    x : np.array
        Corresponds to a matrix :math:`x` which contains all :math:`x_i` values
        (including :math:`x_0=1`).
    y : np.array
        Corresponds to vector :math:`y` which refers to the observed values.
    beta_init : np.array, optional
        Initial :math:`\\beta_i` values may be provided. Otherwise they are set
        to zero.
    gamma : float, optional
        The step size :math:`\\gamma` of the gradient descent. Determines how
        much parameters change per iteration.
    max_iter : float, optional
        Sets the maximum number of iterations.
    threshold : float, optional
        Define the threshold for convergence. If the difference of  sum of the
        squared residuals between two consecutive iterations falls below this
        value, the gradient descent has converged and the function stops.
    scaling : boolean, optional
        By default, the predictors are z-transformed. This improves the gradient
        descent performance because all predictors behave on the same scale.
    regularize : float, optional
        Apply the regularization term :math:`\\lambda` to the estimation
        of :math:`\\beta`. It can prevent overfitting when :math:`x` contains a
        large number of higher order predictors. Increasing :math:`\\lambda`
        will decrease :math:`\\beta_i` values which causes the decision
        boundary to be smoother.

    Returns
    -------
    beta : numpy.array

    Notes
    -----
    .. math:: (1) \\text{The cost function is: } f(\\beta)=\\frac{1}{2n}\\
              \\displaystyle\\sum_{i=1}^{n}(x_i\\beta_m-y_i)^2 = \\
              \\frac{1}{2n}(x\\beta-y)^T (x\\beta-y)=\\frac{1}{2n}SSR(\\beta)
    .. math:: (2) \\text{The aim is: } \\min_\\beta f(\\beta)
    .. math:: (3) \\text{The gradient function is: }f'(\\beta_m)=\\beta_m-\\
              \\gamma\\frac{1}{n}\\displaystyle\\sum_{i=1}^{n}\\
              (x_i\\beta_m-y_i)^2 x_i
    .. math:: (4) f'(\\beta)=\\beta-\\gamma\\frac{1}{n}x^T(x\\beta-y)

    References
    ----------
    [4] https://en.wikipedia.org/wiki/Gradient_descent

    """

    pass

def ordinary_least_squares(x, y, regularize=False):
    """
    Analytically calculate the unknown parameters :math:`\\beta` for a linear
    regression model where :math:`x` refers to the predictor matrix and and
    :math:`y` to the observed values vector.

    Parameters
    ----------
    x : np.array
        Corresponds to a matrix :math:`x` which contains all :math:`x_i` values
        (including :math:`x_0=1`).
    y : np.array
        Corresponds to vector :math:`y` which refers to the observed values.
    regularize : float, optional
        Apply the regularization term :math:`\\lambda` to the estimation
        of :math:`\\beta`. It can prevent overfitting when :math:`x` contains a
        large number of higher order predictors. Increasing :math:`\\lambda`
        will decrease :math:`\\beta_i` values which causes the decision
        boundary to be smoother.

    Returns
    -------
    beta : numpy.array

    Notes
    -----
    .. math:: \\hat{\\beta} =  (x^Tx)^{-1}x^Ty
    .. math:: \\text{Regularization with m predictors: } \\hat{\\beta} = \\
              (x^Tx + \\lambda \\
              \\begin{bmatrix} 0 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\\\
              0 & 0 & \\ddots & \\vdots \\\\ 0 & 0 & \\cdots & 1_{m+1} \\
              \\end{bmatrix})\\
              ^{-1}x^Ty

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Ordinary_least_squares
    .. [3] https://en.wikipedia.org/wiki/Regularization_%28mathematics%29

    Examples
    --------
    >>> # The predictor matrix
    >>> x = np.array([[1, 11, 104],
                      [1, 15, 99],
                      [1, 22, 89],
                      [1, 27, 88]])
    >>> # The observed values vector
    >>> y = np.array([[12],
                      [15],
                      [19],
                      [22]])
    >>> # The parameters vector
    >>> beta_zero = np.zeros((3,1))
    >>> sum_of_squared_residuals(x, y, beta_zero)
    [ 1214.]
    >>> beta_min = ordinary_least_squares(x, y)
    >>> sum_of_squared_residuals(x, y, beta_min)
    [ 0.14455509]

    """

    x_t = x.transpose()
    x_sum = np.dot(x_t, x)

    if regularize:
        lambda_identity = np.identity(x.shape[1])
        lambda_identity[0,0] = 0
        lambda_regularization = regularize * lambda_identity

        inverse = np.linalg.pinv(x_sum + lambda_regularization)
    else:
        inverse = np.linalg.pinv(x_sum)

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

    beta_zero = np.zeros((3,1))

    print(sum_of_squared_residuals(x, y, beta_zero))
    beta_min = ordinary_least_squares(x, y)
    print(sum_of_squared_residuals(x, y, beta_min))

