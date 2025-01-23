#!/usr/bin/env python3
"""module For maximization"""
import numpy as np


def maximization(X, g):
    """function that calculates the maximization step in the EM algorithm for a
    GMM
        X is a numpy.ndarray of shape (n, d) containing the data set
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
    Returns: pi, m, S, None, None, None on failure
        pi is a numpy.ndarray of shape (k,) contaaining the updated priors for
        each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster"""
    try:

        # Num of clusters and data points
        k, n = g.shape
        d = X.shape[1]

        # Calculate cluster responsibilities (sum of posterior probs)
        resp = np.sum(g, axis=1)


        if np.any(resp < 1e-8):
            return None, None, None
            
        # Prior probs (weight/cluster)
        pi = resp / n

        # Calculate cluster means
        m = np.dot(g, X) / resp[:, np.newaxis]

        # Calculate covariance matrices
        S = np.zeros((k, d, d))
        for j in range(k):
            # subract mean from cada data point
            X_centered = X - m[j]

            # weighted covariance calc
            S[j] = np.dot(g[j] * X_centered.T, X_centered) / resp[j]

        return pi, m, S

    except Exception:
        return None, None, None
