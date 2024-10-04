#!/usr/bin/env python3
"""documentation"""
import numpy as np


def shuffle_data(X, Y):
    """Function that suffles data points in two matrices the same way

    X is the first numpy.ndarray of shape(n, nx) to shuffle
        m is the number of data points
        nx is the number of features in X

    Y is the second numpy.ndarray or shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns: the shuffled X and Y matrices
    """
    m = X.shape[0]

    shuffles = np.random.permutation(m)

    X_shuffled = X[shuffles]
    Y_shuffled = Y[shuffles]

    return X_shuffled, Y_shuffled
