#!/usr/bin/env python3
"""Module that intializes Q-table"""
import gymnasium as gym
import numpy as np


def q_init(env):
    """Function that initializes the Q-table
    env is the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray or zeros"""

    # Get the number of states and actions from the environment
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize the Q-table with zeros
    q_table = np.zeros((n_states, n_actions))

    return q_table
