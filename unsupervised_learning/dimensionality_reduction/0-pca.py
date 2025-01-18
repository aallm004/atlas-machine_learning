#!/usr/bin/env python3
"""module"""
import numpy as np


def pca(X, var=0.95):
    """function that performs PCA o a dataset
        X is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimenstions in each point
        var is the fraction of the variance that the PCA transformation should
        maintain
    
        Returns: the weights matrix, W, that maintains var fraction of X's
        original variance
            W is a numpy.ndarray of shape (d, nd) where nd is the new
            dimensionality of the transformed X
        """
    # SVD decomposition
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    # Calculate explained variance ratio
    explained_variance = (s ** 2) / np.sum(s ** 2)

    # Calculate cumulative sum of variance ratios
    cumulative_variance = np.cumsum(explained_variance)

    # get number of components
    n_components = 1 + np.where(cumulative_variance >= var)[0][0]

    return Vh.T[:, :n_components]
