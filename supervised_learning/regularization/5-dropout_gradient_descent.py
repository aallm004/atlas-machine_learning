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
    dz = {}
    for i in range(L, 0, -1):
        if i == L:
            dz['dz' + str(i)] = cache['A' + str(i)] - Y
        else:
            dz['dz' + str(i)] = da * tf.nn.tanh(cache['Z' + str(i)])
            da = np.dot(weights['W' + str(i + 1)].T, dz['dz' + str(i + 1)]) * (1 - cache['A' + str(i)] * cache['A' + str(i)])
        dw = np.dot(dz['dz' + str(i)], cache['A' + str(i - 1)].T) / m
        db = np.sum(dz['dz' + str(i)], axis=1, keepdims=True) / m
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
