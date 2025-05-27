#!/usr/bin/env python3
"""Module for implementing SARSA algorithm"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs SARSA:
        env: environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: min value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

        Returns: Q, the update Q table"""
    n_actions = env.action_space.n

    for episode in range(episodes):
        E = np.zeros_like(Q)
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # TD Error
            td_error = reward + gamma * Q[next_state, next_action] * \
                (not done) - Q[state, action]

            # Replacing traces
            E[state, :] = 0
            E[state, action] = 1

            # Update Q and traces
            Q += alpha * td_error * E
            E *= gamma * lambtha

            state, action = next_state, next_action

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q


def epsilon_greedy(Q, state, epsilon, n_actions):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])
