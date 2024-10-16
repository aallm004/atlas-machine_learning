#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


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
