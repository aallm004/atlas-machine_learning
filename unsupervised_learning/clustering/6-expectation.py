#!/usr/bin/env python3
"""Module For expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S, verbose=False):
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
    if verbose:
        print(f'{X}, {pi}, {m}, {S}')
    try:

        # Get dimensions
        # Number of data points
        n, d = X.shape
        # Number of clusters
        k = pi.shape[0]

        if m.shape != (k, d):
            if verbose:
                print('mshape not equal to k, d')
            return None, None
        if S.shape != (k, d, d):
            if verbose:
                print('S.shape not equal to k,d,d')
            return None, None

        # Initialization of array to store posterior probabilities
        g = np.zeros((k, n))

        # Calculate PDF For each cluster
        for i in range(k):
            PDF = pdf(X, m[i], S[i])
            if PDF is None:
                if verbose:
                    print('PDF is None')
                return None, None
            g[i] = pi[i] * PDF

        # Calculate total probability For normalization
        total_prob = np.sum(g, axis=0, keepdims=True)

        # Ensure numerical stability
        total_prob = np.maximum(total_prob, 1e-300)

        # Normalization to get posterior probabilities
        g = g / total_prob

        # Calculate log likelihood
        L = np.sum(np.log(total_prob))

        return g, L

    except Exception as e:
        if verbose:
            print(f'error: {e}')
            print(f'linenumber:{e.__traceback__.tb_lineno}')
        return None, None
