#!/usr/bin/env python3
"""Module for implementing lambda algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Function that performs the TD(λ) algorithm
        env: the environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to
        take
        lambtha: the eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        
        Returns: V, the updated value estimate"""

    # Train for the given number of episodes
    for _ in range(episodes):
        # Reset env and get initial state
        state, _ = env.reset()

        # Initialize "eligibility traces" for all states
        E = np.zeros_like(V)

        # Loop untili episode is finished or max_steps reached
        for _ in range(max_steps):
            # Choose action based on policy
            action = policy(state)

            # Take action and observe next state as well as reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for current state
            E[state] += 1.0

            #Update value function and eligibility traces for all states
            V += alpha * delta * E
            
            # Decay eligibility traces
            E *= gamma * lambtha

            # To next state
            state = next_state

            # See if episode is finished
            if terminated or truncated:
                break
        
    return V
