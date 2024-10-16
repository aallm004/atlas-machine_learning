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
    A = X
    for i in range(1, L):
        Z = tf.matmul(A, weights['W' + str(i)]) + weights['b' + str(i)]
        if i < L - 1:
            A = tf.nn.tanh(Z)
            A = tf.nn.dropout(A, rate=1 - keep_prob)
        else:
            A = tf.nn.softmax(Z)
    return A
