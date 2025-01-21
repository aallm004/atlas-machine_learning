#!/usr/bin/env python3
"""module for optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by variance
            X is a numpy.ndarray of shape (n, d) containing the data set
            kmin is a positive intneger containing the minimum number of
            clusters to check for (inclusive)
            kmax is a positive integer containing the maximum number of
            clusters to check for (inclsive)
            iterations is a positive integer containing the maximum number
            of iterations for K-means
            Returns: results, d_vars, or None, None on failure
                results is a list containing the outputs of K-means for each
                cluster size
                d_vars is a list containing the difference in variance from the
                smallest cluster size for each cluster size"""
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None

        if not isinstance(kmin, int) or kmin <= 0:
            return None, None

        if kmax is not None:
            if not isinstance(kmax, int) or kmax <= 0:
                return None, None
            if kmin >= kmax:
                return None, None
        else:
            kmax = X.shape[0]
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None

        # Initialize results lists
        results = []
        variances = []

        # Calculate k0means and variance for each k value
        for k in range(kmin, kmax + 1):
            # Get centroids and labels from k-means
            C, clss = kmeans(X, k, iterations)
            if C is None or clss is None:
                return None, None

            results.append((C, clss))

            # Calculate variance for this k
            var = variance(X, C)
            if var is None:
                return None, None
            variances.append(var)

        d_vars = [0]
        d_vars.extend([variances[0] - var for var in variances[1:]])

        return results, d_vars

    except Exception:
        return None, None
