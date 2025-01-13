#!/usr/bin/env python3
import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """function that calculates the intersection of obtaining this data with
    the various hypotheerical probs"""

    if not isinstance(n, int) and n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) and x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    
    if not np.all((Pr >= 0) & (Pr <=1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    l = likelihood(x, n, P)

    intersect = l * Pr

    return intersect
