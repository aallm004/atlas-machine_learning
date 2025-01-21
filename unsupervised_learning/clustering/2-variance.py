#!/usr/bin/env python3
"""module for variance"""
import numpy as np


def variance(X, C):
    """Function that calculates the total intra-cluster variance for a dataset
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    Returns: var or None on failure
        var is the total variance"""

    if not isinstance(X, np.array) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.array) or len(C.shape) != 2:
        return None

    if C.shape[1] != X.shape[1]:
        return None
    
    try:
    # Calculate distances between each point and each centroid
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=1)

        var = np.sum(min_distances**2)

        return var
    except Exception:
        return None
