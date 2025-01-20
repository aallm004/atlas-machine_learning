#!/usr/bin/env python3
"""Module for Initialize K-means"""
import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-means:
        X is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        The cluster centroids should be initialized with a multivariate
        uniform distrubution along eah dimension in d:
            The minimum values for the distribution should be the ninimum
            values of X along each dimension in d
            The maximium values for the distribution should be the maximum
            values of X along each dimension in d
            You should use numpy.random.uniform exactly once
        Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure."""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    minimum = X.min(axis=0)
    maximum = X.max(axis=0)

    return np.random.uniform(low=minimum, high=maximum, size=(k, X.shape[1]))
