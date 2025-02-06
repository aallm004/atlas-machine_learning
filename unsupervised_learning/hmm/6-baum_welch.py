#!/usr/bin/env python3
"""Module for The Baum-Welch Algorithm"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
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
    if not isinstance(Observations, np.ndarray):
        return None, None
    if len(Observations.shape) != 1:
        return None, None

    obvs = len(Observations)
    if not isinstance(Transition, np.ndarray):
        return None, None
    if len(Transition.shape) != 2:
        return None, None

    hidden_states = Transition.shape[0]
    if Transition.shape[1] != hidden_states:
        return None, None

    if not isinstance(Emission, np.ndarray):
        return None, None

    if len(Emission.shape) != 2:
        return None, None

    if Emission.shape[0] != hidden_states:
        return None, None

    num_output = Emission.shape[1]
    if not isinstance(Initial, np.ndarray):
        return None, None

    if Initial.shape != (hidden_states, 1):
        return None, None

    if not isinstance(iterations, int):
        return None, None

    if iterations <= 0:
        return None, None

    if not np.isclose(np.sum(Initial), 1):
        return None, None

    if not np.all(np.isclose(np.sum(Transition, axis=1), 1)):
        return None, None

    if not np.all(np.isclose(np.sum(Emission, axis=1), 1)):
        return None, None

    for _ in range(iterations):
        alpha = np.zeros((hidden_states, obvs))
        alpha[:, 0] = Initial.flatten() * Emission[:, Observations[0]]

        for t in range(1, obvs):
            for j in range(hidden_states):
                alpha[j, t] = alpha[:, t-1].dot(Transition[:, j]) * \
                            Emission[j, Observations[t]]

        beta = np.zeros((hidden_states, obvs))
        beta[:, -1] = 1

        for t in range(obvs-2, -1, -1):
            for j in range(hidden_states):
                beta[j, t] = np.sum(Transition[j, :] *
                                    Emission[:, Observations[t+1]] *
                                    beta[:, t+1])

        # Compute xi and gamma
        xi = np.zeros((obvs-1, hidden_states, hidden_states))
        for t in range(obvs-1):
            denominator = np.sum(alpha[:, t].reshape((-1, 1)) *
                                 Transition *
                                 Emission[:,
                                          Observations[t+1]].reshape((1, -1)) *
                                 beta[:, t+1].reshape((1, -1)))
            for i in range(hidden_states):
                numerator = alpha[i, t] * \
                            Transition[i, :] * \
                            Emission[:, Observations[t+1]] * \
                            beta[:, t+1]
                xi[t, i, :] = numerator / denominator

        gamma = np.zeros((hidden_states, obvs))
        gamma[:, :-1] = np.sum(xi, axis=2).T
        gamma[:, -1] = np.sum(xi[-1], axis=0)

        # Update parameters
        Transition = np.sum(xi, axis=0) / np.sum(gamma[:,
                                                 :-1],
                                                 axis=1).reshape((-1, 1))

        denominator = np.sum(gamma, axis=1).reshape((-1, 1))
        for obs in range(num_output):
            Emission[:, obs] = np.sum(gamma[:, Observations == obs], axis=1)

        Emission = Emission / denominator

        # Handle numerical stability
        Transition = np.where(Transition == 0, 1e-300, Transition)
        Emission = np.where(Emission == 0, 1e-300, Emission)

        # Normalize
        Transition = Transition / np.sum(Transition, axis=1).reshape(-1, 1)
        Emission = Emission / np.sum(Emission, axis=1).reshape(-1, 1)

    return Transition, Emission
