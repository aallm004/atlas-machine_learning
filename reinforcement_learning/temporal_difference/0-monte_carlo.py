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
    def run_single_episode():
        state = []
        reward = []

        # Initialize at episode's start
        current_pos, _ = env.reset()

        # Generate episode up to max length
        step_counter = 0
        while step_counter < max_steps:
            # Choose action by policy
            move = policy(current_pos)

            # Execute action and see result
            new_pos, immediate_reward, terminated, truncated, _ = \
                env.step(move)

            # Recort state-reward pair
            state.append(int(current_pos))
            reward.append(int(immediate_reward))

            # Check if episode has ended
            if terminated or truncated:
                break

            # Transition to next state
            current_pos = new_pos
            step_counter += 1

        return np.array(state), np.array(reward)

    # Monte Carlo learning loop
    for trial_num, _ in enumerate(range(episodes)):
        # Generate complete episode path
        path, rewards_seq = run_single_episode()

        # Process trajectory in reverse to calculate returns
        cumulative_return = 0

        # Process in reverse using negative indexing
        for position in range(len(path)):
            # Negative indexing to process from end to beginning
            reverse_idx = -(position + 1)
            location = path[reverse_idx]
            immediate_payoff = rewards_seq[reverse_idx]

            # Calculate discounted return
            cumulative_return = immediate_payoff + gamma * cumulative_return

            # Only update if state doesn't appear in first trial_num pos
            early_positions = path[:trial_num] if trial_num < len(path) else \
                path[:len(path)]

            # Update value function if first time
            if location not in early_positions:
                # Monte Carlo update
                prediction_error = cumulative_return - V[location]
                V[location] += alpha * prediction_error

    return V
