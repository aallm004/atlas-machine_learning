#!/usr/bin/env python3
"""module for k-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """function that performs K-means on a dataset:
        X is a numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum number of
        iterations that should be performed
        If no change in the cluster centroids occurs between iterations,
        your function should return
        Initialize the cluster centroids using a multivariate uniform
        distribution (based on 0-initialize.py)
        If a cluster contains no data points during the update step,
        reinitialize its centroid
        Returns: C, clss or None, None on failure
            C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
            clss is  numpy.ndarray of shape (n,) containing the index
            of the cluster in C that each data point belongs to"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    try:

        C = initialize(X, k)
        if C is None:
            return None, None

        for i in range(iterations):
            C_prev = C.copy()

            # Calculates distances and assign clusters
            distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
            clss = np.argmin(distances, axis=0)

            # Update centroids
            for j in range(k):
                points = X[clss == j]
                if len(points) == 0:
                    C[j] = initialize(X, 1)[0]
                else:
                    C[j] = np.mean(points, axis=0)

            if np.all(C == C_prev):
                break

        return C, clss

    except Exception as e:
        return None, None


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
