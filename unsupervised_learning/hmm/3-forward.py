#!/usr/bin/env python3
"""Module for The Forward Algorithm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Function that performs the forward algorithm for a hidden markov model:
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
        transition probabilites
            Transition[i, j] is the probability of the transtioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
        Returns: P, F, None, None on failure
            P is the likelihood of the observations given the model
            F is a numpy.ndarray of shape (N, T) containing the forward path
            probabilities
                F[i, j] is the probability of being in hidden state i at time j
                given the previous observations"""
    try:
        obvs = Observation.shape[0]
        hidden_states, possible = Emission.shape

        # Shape Check
        if Transition.shape != (hidden_states, hidden_states):
            return None, None

        if Initial.shape != (hidden_states, 1):
            return None, None

        if not np.all(Observation < possible):
            return None, None

        # Probability constraints
        if not (np.allclose(np.sum(Emission, axis=1), 1) and
                np.allclose(np.sum(Transition, axis=1), 1) and
                np.allclose(np.sum(Initial), 1)):
            return None, None

        # Are there negative probabilities?
        if np.any(Emission < 0) or np.any(Transition < 0) or \
           np.any(Initial < 0):
            return None, None

    except (AttributeError, IndexError, ValueError):
        return None, None

    # Initialization of forward path probability matrix
    forward_path = np.zeros((hidden_states, obvs))

    # Initialization of first timestep
    forward_path[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

    # Forward algorithm recursion
    for i in range(1, obvs):
        for state in range(hidden_states):
            forward_path[state, i] = np.sum(forward_path[:, i-1] *
                                            Transition[:, state]) * \
                                            Emission[state, Observation[i]]

    likelihood = np.sum(forward_path[:, -1])

    return likelihood, forward_path
