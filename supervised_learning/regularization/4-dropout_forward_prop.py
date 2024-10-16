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
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]
        Z = np.matmul(W, A_prev) + b
        
        if l < L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            A = softmax(Z)
        
        cache['A' + str(l)] = A

    return cache

def softmax(Z):
    t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return t / np.sum(t, axis=0, keepdims=True)
