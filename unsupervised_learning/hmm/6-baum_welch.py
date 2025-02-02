#!/usr/bin/env python3
"""Module for The Baum-Welch Algorithm"""
import numpy as np


def baum_welch(Observatoins, Transition, Emission, Initial, iterations=1000):
    """Function that performs the Baum-Welch algorithm for a hidden markov
    model:
        Observations is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            T is the number of observations
        Transition is a numpy.ndarray of shape (M, M) that contains the
        initialized transition probabilites
            M is the number of hidden states
        Emission is a numpy.ndarray of shape (M, N) that contains the
        initialized emission probabilities
            N is the number of output states
        Initial is a numpy.ndarray of shape(M, 1) that containis the intialized
        starting probabilites
        iterations is the number of times expectation-maximization should be
        performed
        Returns: the converged Transition, Emission, or None, None on
        failure"""
