"""This module contains simple functions for linear regressions."""

import numpy as np
import transform

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
    ----------
    .. [1] https://en.wikipedia.org/wiki/Residual_sum_of_squares

    Notes
    -----
    .. math:: SSR(\\beta) = \\displaystyle\\sum_{i=1}^{n}(x_i\\beta-y_i)^2
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

def gradient_descent(x, y, beta_init=None, gamma=0.01, max_iter=200,
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
    .. math:: f(\\beta)=\\frac{1}{2n}\\
              \\displaystyle\\sum_{i=1}^{n}(x_i\\beta-y_i)^2 = \\
              \\frac{1}{2n}(x\\beta-y)^T (x\\beta-y)=\\frac{1}{2n}SSR(\\beta)
    .. math:: f'(\\beta)=\\beta-\\
              \\frac{\\gamma}{n}\\displaystyle\\sum_{i=1}^{n}\\
              (x_i\\beta_m-y_i) x_i = \\beta-\\gamma\\frac{1}{n}x^T(x\\beta-y)
    .. math:: f'_{reg}(\\beta)=\\beta(1-\\gamma \\frac{\\lambda}{n} \\
              \\beta_{reg})-\\frac{\\gamma}{n}x^T(x\\beta-y) \\text{ where } \\
              \\beta_{reg} = \\begin{bmatrix} 0 \\\\ 1 \\\\ \\vdots \\\\ 1_m \\
              \\end{bmatrix}

    References
    ----------
    [4] https://en.wikipedia.org/wiki/Gradient_descent

    """

    n, m = x.shape

    if scaling:
        x = transform.standardize(x[:,1:])
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

    if not beta_init:
        beta_init = np.ones((m, 1))

    reg_term = np.ones((m,1))
    if regularize:
        beta_reg = np.ones((m,1))
        beta_reg[0,0] = 0
        reg_term = 1-(gamma*(regularize/float(n)) * beta_reg)

    beta = beta_init
    current_cost = None
    n_iter = 0

    while True:
        right_term = (gamma / float(n))
        right_term = right_term * np.dot(np.transpose(x), np.dot(x, beta) - y)

        left_term = beta * reg_term
        beta = left_term - right_term

        cost = sum_of_squared_residuals(x, y, beta)

        if current_cost is None:
            current_cost = cost
        elif np.abs(cost - current_cost) < threshold:
            break
        else:
            current_cost = cost

        if n_iter == max_iter:
            break
        else:
            n_iter += 1

    return beta


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
    """
    x = np.array([[1, 11, 104],
                  [1, 15, 99],
                  [1, 22, 89],
                  [1, 27, 88]])

    x2 = np.array([[1, 11],
                   [1, 15],
                   [1, 22],
                   [1, 27]])

    y = np.array([[12],
                  [15],
                  [19],
                  [22]])

    beta_zero = np.zeros((2,1))

    print(sum_of_squared_residuals(x2, y, beta_zero))
    beta_min = ordinary_least_squares(x2, y)
    print(sum_of_squared_residuals(x2, y, beta_min))

    beta_grad = gradient_descent(x2, y, scaling=False, gamma=0.001)
    print(sum_of_squared_residuals(x2, y, beta_grad))
    """

    from sklearn import datasets, linear_model
    diabetes = datasets.load_diabetes()

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes.data, diabetes.target)
    print(sum_of_squared_residuals(diabetes.data, diabetes.target, regr.coef_))

    x = np.append(np.ones((diabetes.data.shape[0], 1)), diabetes.data, axis=1)
    y = diabetes.target
    ols = ordinary_least_squares(x, y)
    print(sum_of_squared_residuals(x, y, ols))

    gd = gradient_descent(x,np.reshape(y,(442,1)))
    print(sum_of_squared_residuals(x, y, gd.ravel()))

