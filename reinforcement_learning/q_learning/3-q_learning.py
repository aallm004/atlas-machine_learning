#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs Q-learning
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsolon between episodes
        When the agent falls in a hole, the reward should be updated to be -1
        
        Returns: Q, total_rewards
            Q is the updated Q-table
            total_rewawrds is a list containing the rewards per episode"""
    
    rewards = []

    for episode in range(episodes):
        # reset the environment and get initial state
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for step in range(max_steps):
            # select action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)

            # If agent falls in a hole, reward should be -1
            if done and reward == 0:
                reward = -1

            # Update the Q-table
            # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Update state and total reward
            state = next_state
            episode_reward += reward

            # If episode is done, break loop
            if done:
                break

        # Append total reward for current episode
        rewards.append(episode_reward)

        # update epsilon using decay rate
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, rewards
