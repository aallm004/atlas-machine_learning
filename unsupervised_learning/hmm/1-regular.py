#!/usr/bin/env python3
"""Module for Regular Chain"""
import numpy as np


def regular(P):
    """Function that determines the steady state probabilities of a regular
    markob chain:
        P is a square 2D numpy.ndarray of shape (n, n) representing the
        transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markob chain
            Returns: a numpy.ndarray of shape (1, n) containing the steady
            state probabilities, or None on failure"""
    try:
        n = P.shape[0]

        # See if matrix is regular by computing high power
        power = np.linalg.matrix_power(P, n*n)
        if np.any(power <= 0):
            return None

        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(P.T)

        # Find index of eigenvalue closest to 1
        index = np.argmin(np.abs(eigenvals - 1))

        # Get corresponding eigenvector
        pi = eigenvecs[:, index].real

        # Normalize
        pi = pi / np.sum(pi)

        return pi.reshape(1, n)
    
    except (np.linalg.LinAlgError, ValueError):
        return None
