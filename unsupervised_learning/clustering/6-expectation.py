#!/usr/bin/env python3
"""Module For expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation step in the EM algorithm For a
    GMM:
        X is a numpy.ndarray of shape (n, d) containing the data set
        pi is a numpy.ndarray of shape (k,) containing the priors For each
        cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid means For
        each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices For each cluster
        Returns: g, l, or None, None on failure
            g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities For each data point in each cluster
            l is the total log likelihood"
    
    """
    try:

        # Get dimensions
        # Number of data points
        n, d = X.shape
        # Number of clusters 
        k = pi.shape[0]
       
        if len(pi.shape) != 1 or pi.shape[0] != k:
            return None, None
        if m.shape != (k, d):
            return None, None
        if S.shape != (k, d, d):
            return None, None
        
        # Initialization of array to store posterior probabilities
        g = np.zeros((k, n))

        # Calculate PDF For each cluster
        for i in range(k):
            PDF = pdf(X, m[i], S[i])
            if PDF is None:
                return None, None
            g[i] = p[i] * PDF

        # Calculate total probability For normalization
        total_prob = np.sum(g, axis=0, keepdims=True)

        total_prob = np.maximum(total_prob, np.finfo(float).eps)

        # Normalization to get posterior probabilities
        g = g / total_prob

        # Calculate log likelihood
        l = np.sum(np.log(total_prob))

        return g, l

    except Exception:
        return None, None
