#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function that calculates the cost of a neural network with L2
        regularization
        cost is the cost of the network without L2 regularization
        lambtha is the regularization parameter
        weights is a dictionary of the weights and biases(numpy.ndarrays)
        or the
        neural network
        L is the number of layers in the neural network
        m is the number of data points used

        Returns: the cost of the network accounting for L2 regularization
        """
    sum_weights = 0
    """sum of squares of weights"""
    for i in range(1, L + 1):
        sum_weights += np.sum(np.square(weights["W" + str(i)]))

    """add regularization to the cost"""
    cost_l2 = cost + (lambtha / (2 * m)) * sum_weights
    return cost_l2
