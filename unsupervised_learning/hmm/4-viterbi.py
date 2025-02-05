#!/usr/bin/env python3
"""Module for The Viretbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Function that calculates the most likely sequence of hidden states for a
    hidden markov model
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
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
        Returns: path, P or None, None on failure
            path is a list of lenth T containing the most likely sequence of
            hidden states
            P is the probability of obtaining the path sequence"""
    try:
        obvs = Observation.shape[0]
        hidden_states = Emission.shape[0]

        viterbis = np.zeros((hidden_states, obvs))
        backpoint = np.zeros((hidden_states, obvs), dtype=int)

        viterbis[:, 0] = np.log(Initial.flatten()) + np.log(
            Emission[:, Observation[0]])

        for x in range(1, obvs):
            for i in range(hidden_states):
                probabilities = (viterbis[:, x-1] +
                                 np.log(Transition[:, i]) +
                                 np.log(Emission[i, Observation[x]]))

                viterbis[i, x] = np.max(probabilities)

                backpoint[i, x] = np.argmax(probabilities)

        path = [np.argmax(viterbis[:, -1])]

        for x in range(obvs-1, 0, -1):
            path.insert(0, backpoint[path[0], x])

        P = np.exp(viterbis[path[-1], -1])

        return path, P

    except (ValueError, IndexError, ZeroDivisionError):
        return None, None
