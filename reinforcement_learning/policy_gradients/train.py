#!/usr/bin/env python3
"""Module for training policy gradient"""
import numpy as np


def train(env, np_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

        Return: all values of the score (sum of all rewards during one
        episode loop)"""
    policy_gradient = __import__('policy_gradient').policy_gradient

    # Initialize randomized weights
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    weights = np.random.random((state_dim, action_dim))

    scores = []

    for episode in range(np_episodes):
        # reset env
        state, _ = env.reset()

        # Store episode data
        states = []
        actions = []
        rewards = []
        gradients = []

        # Run episode
        done = False
        truncated = False

        while not (done or truncated):
            # Get action and gradient
            action, gradient = policy_gradient(state, weights)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store data from episode
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)

            state = next_state

        # Calculation of discounted return
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = np.array(returns)

        # Update weights w policy gradient
        for i in range(len(gradients)):
            weights += alpha * gradients[i] * returns[i]

        # Calculate and save score
        score = sum(rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
