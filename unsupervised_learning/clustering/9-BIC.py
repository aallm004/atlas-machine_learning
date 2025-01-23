#!/usr/bin/env python3
"""module for BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Function that finds the best number of clusters for a GMM using the
    Bayesian Information Criterion:

        X is a numpy.ndarray of shape (n, d) containing the data set
        kmin is a positive integer containing the minimum number of clusters to
        check for (inclusive)
        kmax is a positive integer containing the maximum number of clusters to
        check for (inclusive)
    If kmax is None, kmax should be set to the maximum number of clusters
    possible
        iterations is a positive integer containing the maximum number of
        iterations for the EM algorithm
        tol is a non-negative float containing the tolerance for the EM
        algorithm
        verbose is a boolean that determines if the EM algorithm should print
        information to the standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
        pi is a numpy.ndarray of shape (k,) containing the cluster priors for
        the best number of clusters
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
        the best number of clusters
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested
        Use: BIC = p * ln(n) - 2 * l
        p is the number of parameters required for the model
        n is the number of data points used to create the model
        l is the log likelihood of the model """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    k_range = range(kmin, kmax + 1)
    l = []
    b = []
    results = []

    for k in k_range:
        pi, m, S, g, ll = expectation_maximization(X, k, iterations=iterations,
                                                   tol=tol, verbose=verbose)
        if pi is None:
            return None, None, None, None

        l.append(ll)
        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = p * np.log(n) - 2 * ll
        b.append(bic)
        results.append((pi, m, S))

        if k > kmin and b[-1] > b[-2]:
            break

    l = np.array(l)
    b = np.array(b)
    best_k = np.argmin(b) + kmin
    best_result = results[best_k - kmin]

    return best_k, best_result, l, b
