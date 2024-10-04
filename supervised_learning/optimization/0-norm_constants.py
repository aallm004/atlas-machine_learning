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
    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)
    
    return mean, std
