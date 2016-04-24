"""This module contains general transformation functions."""

import numpy as np


def standardize(x):
    """
    Perform z-transformation to get standard score of input matrix.

    Parameters
    ----------
    x : numpy.array
        Input matrix to be standardized.

    Returns
    -------
    x_stand : numpy.array
        Standardized matrix.

    Notes
    -----
    .. math:: z = \\frac{x - \\mu}{\\sigma}

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_score

    Examples
    --------
    >>> # The input matrix
    >>> x = np.array([[1, 11, 104],
                      [1, 15, 99],
                      [1, 22, 89],
                      [1, 27, 88]])
    >>> standardize(x)
    [[ 0.         -1.25412576  1.33424877]
     [ 0.         -0.60683505  0.59299945]
     [ 0.          0.52592371 -0.88949918]
     [ 0.          1.3350371  -1.03774904]]

    """

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # handle division by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        x_stand = np.true_divide(x-mean,std)
        x_stand[x_stand == np.inf] = 0
        x_stand = np.nan_to_num(x_stand)

    return x_stand

if __name__ == "__main__":
    x = np.array([[1, 11, 104],
                  [1, 15, 99],
                  [1, 22, 89],
                  [1, 27, 88]])

    print(standardize(x))
