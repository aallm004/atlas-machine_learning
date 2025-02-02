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
