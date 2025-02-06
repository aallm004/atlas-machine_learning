#!/usr/bin/env python3
"""Module for The Backward Algorithm"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward algorithm for a hidden markov model:
        Observation is a numpy.ndarray of shape (T,) that contains the index of
        the observation
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the hidden
            state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
            Transition[i, j] is the probability of transitioning rom the hidden
            state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
        Returns: P, B, or None, None on failure
            P is the likelihood of the observations given the model
            B is a numpy.ndarray of shape (N, T) containing the backward path
            probabilities
                B[i, j] is the probability of generating the future
                observations from hidden state i at time j"""
    try:
        obvs = Observation.shape[0]
        hidden_states = Emission.shape[0]

        if not isinstance(Observation, np.ndarray):
            return None, None
        if len(Observation.shape) != 1:
            return None, None
        if not isinstance(Emission, np.ndarray):
            return None, None
        if len(Emission.shape) != 2:
            return None, None
        if not isinstance(Transition, np.ndarray):
            return None, None
        if Transition.shape != (hidden_states, hidden_states):
            return None, None
        if not isinstance(Initial, np.ndarray):
            return None, None
        if Initial.shape != (hidden_states, 1):
            return None, None

        backwards = np.zeros((hidden_states, obvs))

        backwards[:, obvs-1] = 1

        for t in range(obvs-2, -1, -1):
            for i in range(hidden_states):
                backwards[i, t] = np.sum(
                    Transition[i, :] *
                    Emission[:, Observation[t+1]] *
                    backwards[:, t+1]
                )

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * +
                   backwards[:, 0])

        return P, backwards

    except (TypeError, ValueError):
        return None, None
