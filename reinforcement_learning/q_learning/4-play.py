#!/usr/bin/env python3
"""Module that intializes Q-table"""
import numpy as np


def play(env, Q, max_steps=100):
    """Function that has the trained agent play an episode
    env is the FrozenLakeEnv instance
    Q is the numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode

    Returns: The total rewards for the episode and a list of rendered
    outputs representing the board state at each step"""

    # Reset env
    state, _ = env.reset()
    total_reward = 0
    frames = []

    # Display initial state
    frames.append(env.render())

    for step in range(max_steps):
        # Choosing best action using the Q-table
        action = np.argmax(Q[state])

        # Take action and see what the next state and reward are
        next_state, reward, done, _, _ = env.step(action)

        # Update state and total reward
        state = next_state
        total_reward += reward

        # Current state
        frames.append(env.render())

        # If episode is done, break loop
        if done:
            break

    return total_reward, frames
