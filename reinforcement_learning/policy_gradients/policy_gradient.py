#!/usr/bin/env python3
"""module for policy gradient"""
import numpy as np


def policy(matrix, weight):
    """Function that computes the policy with a weight of a matrix
        matrix is the state matrix of shape (batch_size, state_dim)
        weight is the weight matrix of shape (state_dim, action_dim)"""

    # See if matrix is 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Find logits by matrix multiplication
    logits = np.dot(matrix, weight)

    # Apply softmax
    sm_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))

    # Normalize to get probabilities
    policy_probs = sm_logits / np.sum(sm_logits, axis=1, keepdims=True)

    return policy_probs


def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy gradient based on a state
    and a weight matrix
        state is a matrix representing the current observation of the
        environment
        weight is a matrix of random weight

        Return: the action and the gradient (in this order)"""

    # Get policy probabilities
    probs = policy(state, weight)

    # Sample action from the prob distribution
    action = np.random.choice(probs.shape[1], p=probs.flatten())

    # Create one-hot vector for selected action
    action_onehot = np.zeros(probs.shape[1])
    action_onehot[action] = 1

    # Compute gradient: state^T * (action_onehot - probs)
    gradient = np.outer(state.flatten(), action_onehot - probs.flatten())

    return action, gradient
