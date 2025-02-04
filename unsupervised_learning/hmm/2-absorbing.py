#!/usr/bin/env python3
"""Module for Absorbing Chain"""
import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing:
        P is a square 2D numpy.ndarray of shape (n, n) representing the
        standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
            Returns: True if it is absorbing for False on failure"""
    try:
        n = P.shape[0]
        # Find absorbing states
        absorbing_states = np.where(np.diag(P) == 1)[0]

        # Find absorbing states
        if len(absorbing_states) == 0:
            return False

        # Check that rows of absorbing states are correct
        # (probability of 1 for self-transition, 0 for others)
        for state in absorbing_states:
            if not np.allclose(P[state], np.eye(n)[state]):
                return False

        # Remove absorbing states to get second matrix of non-absorbing
        non_absorbing = np.setdiff1d(np.arange(n), absorbing_states)
        if len(non_absorbing) == 0:
            return True

        # Get submatrix of non-absorbing states transitioning to each other
        submatrix = P[non_absorbing][:, non_absorbing]

        # Check if 1 is not an eigenvalue of submatrix
        eigenvals = np.linalg.eigvals(submatrix)
        return not np.any(np.isclose(eigenvals, 1))

    except (np.linalg.LinAlgError, ValueError, IndexError):
        return False
