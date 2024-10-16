#!/usr/bin/env python3
"""documentation"""
import numpy as np


def droupout_forward_prop(X, weights, L, keep_prob):
    """
    function that conducts forward propagation using Dropout
        X is a numpy.ndarray with shape (nx, m) containing the input data
        for the network
            nx is the number of input features
            m is the number of data points
        weights is a dictionary of weights and biases for the network
        L is the number of layers in the network
        keep_prob is the probability that a node will be kept
        Returns: the output of the network after dropout is applied
    """

    cache = {"A0": X}
    dropout_mask = {}
    A_prev = X

    for l in range(1, L + 1):
        W = weights[f"W{l}"]
        b = weights[f"b{l}"]
        
        Z = np.dot(W, A_prev) + b
        
        if l == L:  # Output layer
            A = softmax(Z)
        else:  # Hidden layers
            A = np.tanh(Z)
            
            # Apply dropout
            mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
            A *= mask
            dropout_mask[f"D{l}"] = mask
        
        cache[f"A{l}"] = A
        A_prev = A

    return cache, dropout_mask

def softmax(Z):
    """Compute softmax activation."""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
