#!/usr/bin/env python3
"""Module for expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation step in the EM algorithm for a
    GMM:
        X is a numpy.ndarray of shape (n, d) containing the data set
        pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster
        Returns: g, l, or None, None on failure
            g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
            l is the total log likelihood"
    
    """
    try:

        # Get dimensions
        # Number of data points
        n = X.shape[0]
        # Number of clusters 
        k = pi.shape[0]
        # Initialization of array to store posterior probabilities
        g = np.zeros((k, n))

        # Calculate PDF for each cluster
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])

        # Calculate total probability for normalization
        total_prob = np.sum(g, axis=0, keepdims=True)

        # No division by zero here
        total_prob = np.where(total_prob == 0, 1e-300, total_prob)

        # Normalization to get posterior probabilities
        g = g / total_prob

        # Calculate log likelihood
        l = np.sum(np.log(total_prob))

        return g, l

    except Exception:
        return None, None
