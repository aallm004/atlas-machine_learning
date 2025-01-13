#!/usr/bin/env python3
"""module for marginal"""
import numpy as np


def marginal(x, n, P, Pr):
    """function that calculates the marginal probability of obtaining the
    data"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or " +
                         "equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    margprob = likelihood(x, n, P)

    return np.sum(margprob * Pr)


def likelihood(x, n, P):
    """function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or " +
                         "equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    ant = np.math.factorial(n) / (np.math.factorial(x) *
                                  np.math.factorial(n - x))

    like = ant * (P ** x) * ((1 - P) ** (n - x))

    return like
