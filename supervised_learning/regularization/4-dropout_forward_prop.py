#!/usr/bin/env python3
"""documentation"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
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

    cache = {'A0': X}
    A = X

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        z = np.matmul(W, A) + b

        if layer == L:
            A = np.exp(z)
            cache[f'A{layer}'] = A / np.sum(A, axis=0, keepdims=True)
        else:
            A = np.tanh(z)
            dropout_mask = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A *= dropout_mask
            A /= keep_prob
            cache[f'A{layer}'] = A
            cache[f'D{layer}'] = dropout_mask

    return cache
