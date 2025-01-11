#!/usr/bin/env python3
"""module zero"""
import numpy as np


def mean_cov(X):
    """function that calculates the mean and covariance of a data set"""
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    #dimensions
    n, d = X.shape

    #Are there multiple data points?
    if n < 2:
        raise ValueError("X must contain multiple data points")

    #mean calculation with reshape
    mean = np.mean(X, axis=0).reshape(1, -1)

    #center data
    X_centered = X - mean

    # cov = (1/(n - 1)) * (X - mean)^T * (X - mean)
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
