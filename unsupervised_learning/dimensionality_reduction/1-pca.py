#!/usr/bin/env python3
"""module for PCA dimensionality reduction"""
import numpy as np


def pca(X, ndim):
    """function that performs PCA on a dataset:
        X is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
        ndim is the new dimensionality of the transformed X
        Returns: T, a numpy.ndarray of shape (n, ndim) containing the
        transformed version of X"""

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # SVD
    _, _, Vh = np.linalg.svd(X_centered)

    # Get ndim components
    ndim = Vh[:ndim].T

    # Transformation
    transform = np.matmul(X_centered, ndim)

    return transform
