#!/usr/bin/env python3
"""documentation"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """function that updates the weights of a neural network with dropout
    regularization using gradient descent
        Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
            classes is the number of classes
            m is the number of data points
        weights is a dictionary of weights and biases of the neural network
        cache is a dictionary of the weights and biases of the neural network
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
        All layers use the tanh activation function except the last, which
        uses the softmax activation function
        The weights of the network should be updated in place
    """
    m = Y.shape[1]
    
    for layer in reversed(range(1, L + 1)):
        A_curr = cache[f'A{layer}']
        if layer == L:
            dZ = A_curr - Y
        else:
            dA = np.matmul(weights[f'W{layer + 1}'].T, dZ) * (1 - np.power(A_curr, 2))
            dZ = dA * cache[f'D{layer}'] / keep_prob
        
        A_prev = cache[f'A{layer - 1}']
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
