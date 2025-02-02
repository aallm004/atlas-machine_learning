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
