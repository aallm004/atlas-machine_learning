#!/usr/bin/env python3
"""Module for implementing Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Function that performs the Monte Carlo algorithm:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        
        Returns: V, the updated value estimate"""
    for ep in range(episodes):
        # Generate episode
        rewards = []
        states = []

        state, _ = env.reset()
        states.append(state)


        for step in range(max_steps):
            # Retrieve action from policy
            action = policy(state)
            
            # Action is taken
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            state = next_state
            # See if episode is finished
            if terminated or truncated:
                break
            
            states.append(state)

        # Calculate returns and update value function
        G = 0
        
        # Go through the episode in reverse order
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            state_t = states[t]
            
            # Update the value function with incremental mean
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V

