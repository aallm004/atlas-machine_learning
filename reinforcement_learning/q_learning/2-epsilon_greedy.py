#!/usr/bin/env python3
"""module to use epsilon-greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon-greedy to determine the next action
    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation

    Returns: the next action index"""

    # Sample random value between 0 and 1
    prob = np.random.uniform(0, 1)

    # Number of possible actions
    n_actions = Q.shape[1]

    # Exploration to select a random action
    if prob < epsilon:
        action = np.random.randint(0, n_actions)
    # Exploitation: select the action with highest Q value
    else:
        action = np.argmax(Q[state])

    return action
