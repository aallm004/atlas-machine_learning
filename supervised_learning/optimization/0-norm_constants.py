#!/usr/bin/env python3
"""documentation"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix

    X is the numpy.ndarray or shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features

    Returns: the mean and standard deviation of each feature, respectively
    """
    m, nx = X.shape

    mean = [sum(X[i][:, j] / m for j in range(nx)]

    var = [sum((X[i, j] - mean[j])**2 for i in range(m)) /
           m for j in range(nx)]

    std = [var[j]**0.5 for j in range(nx)]

    return mean, std
