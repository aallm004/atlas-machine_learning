#!/usr/bin/env python3
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
    # Dimensions
    n, d = X.shape

    # Calculate covariance matrix (d x d)
    covariance = np.cov(X, rowvar=False)


