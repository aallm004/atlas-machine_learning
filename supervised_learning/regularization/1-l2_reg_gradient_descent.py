#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the umber of classes
        m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs of each layer of the neural
        network
        alpha is the learning rate
        lambtha is the regularization parameter
        L is the number of layers of the network
    The neural network uses tanh activations of each layer except the last,
    which uses softmax
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for x in range(L, 0, -1):
        A_prev = cache["A" + str(x - 1)]
        W = weights["W" + str(x)]

        dW = np.dot(dZ, A_prev.T) / m + (lambtha * W) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.dot(W.T, dZ) * (1 - np.power(A_prev, 2))

        weights["W" + str(x)] -= alpha * dW
        weights["b" + str(x)] -= alpha * db
