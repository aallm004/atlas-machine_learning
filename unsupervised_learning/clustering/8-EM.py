#!/usr/bin/env python3
"""module For EM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Write a function def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False): that performs the expectation maximization for a GMM:

        X is a numpy.ndarray of shape (n, d) containing the data set
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum number of iterations for the algorithm
        tol is a non-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal to tol you should stop the algorithm
        verbose is a boolean that determines if you should print information about the algorithm
    If True, print Log Likelihood after {i} iterations: {l} every 10 iterations and after the last iteration
        {i} is the number of iterations of the EM algorithm
        {l} is the log likelihood, rounded to 5 decimal places
        
        Returns: pi, m, S, g, l, or None, None, None, None, None on failure
            pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
            m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
            S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
            g is a numpy.ndarray of shape (k, n) containing the probabilities for each data point in each cluster
            l is the log likelihood of the model
        """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        if verbose:
            print('**a**')
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        if verbose:
            print('**b**')
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        if verbose:
            print('**c**')
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        if verbose:
            print('**d**')
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        if verbose:
            print('**e**')
        return None, None, None, None, None
    
    try:
        pi, m, S = initialize(X, k)
        if pi is None or m is None or S is None:
            if verbose:
                print('**f**')
            return None, None, None, None, None
        prev_l = 0

        g, prev_l = expectation(X, pi, m, S)
        if g is None:
            if verbose:
                print('**g**')
            return None, None, None, None, None

        if verbose:
            print('Log Likelihood after {} iterations: {:.5f}'.format(0, prev_l))

        for i in range(1, iterations + 1):

            pi, m, S = maximization(X, g)
            if pi is None or m is None or S is None:
                if verbose:
                    print('**h**')
                return None, None, None, None, None

            g, current_l = expectation(X, pi, m, S)
            if g is None or current_l is None:
                if verbose:
                    print('**i**')
                return None, None, None, None, None

            if np.abs(current_l - prev_l) <= tol:
                if verbose:
                    print('Log Likelihood after {} iterations: {:.5f}'.format(i, current_l))
                return pi, m, S, g, current_l

            if verbose and i % 10 == 0:
                print('Log Likelihood after {} iterations: {:.5f}'.format(i, current_l))

            prev_l = current_l

        if verbose and not i % 10 == 0:
            print('Log Likelihood after {} iterations: {:.5f}'.format(i, prev_l))

        return pi, m, S, g, l

    except Exception as e:
        if verbose:
            print(f'error: {e}')
            print(f'linenumber:{e.__traceback__.tb_lineno}')
        return None, None, None, None, None
