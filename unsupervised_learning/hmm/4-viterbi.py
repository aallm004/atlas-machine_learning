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
        hidden_states, possible = Emission.shape

        if Transition.shape != (hidden_states, hidden_states):
            return None, None

        if Initial.ndim == 1:
            Initial = Initial.reshape((-1, 1))

        if Initial.shape != (hidden_states, 1):
            return None, None

        if not np.all(Observation < possible):
            return None, None

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

    # Initialize viterbi trellis and backpointer
    viterbi_path = np.zeros((hidden_states, obvs))
    backpointer = np.zeros((hidden_states, obvs), dtype=int)

    # Initialization of first timestep
    viterbi_path[:, 0] = np.log(Initial.flatten()) + \
        np.log(Emission[:, Observation[0]])

    # Viterbi algorithm
    for t in range(1, obvs):
        for state in range(hidden_states):
            # Calculate probs for each previous state
            temp = (viterbi_path[:, t-1] +
                    np.log(Transition[:, state]) +
                    np.log(Emission[state, Observation[t]]))

            # Find most likely previous state and probability
            viterbi_path[state, t] = np.max(temp)
            backpointer[state, t] = np.argmax(temp)

    # Backtrack to find the most likely path
    path = []
    current_state = np.argmax(viterbi_path[:, -1])

    for t in range(obvs-1, -1, -1):
        path.append(int(current_state))
        if t > 0:
            current_state = backpointer[current_state, t]

    path.reverse()
    
        # Calculate prob of path
    prob = np.max(viterbi_path[:, -1])

    return path, prob
