#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
    cost (tensor): The cost of the network without L2 regularization
    model (keras.Model): A Keras model that includes layers with L2
    regularization

    Returns:
    tensor: The total cost for each layer of the network, accounting for L2
    regularization
    """
    cost_l2 = []
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kerel_regularizer:
            cost_l2 += \
                tf.math.reduce_sum(layer.kernel_regularizer(layer.kernel))

        if hasattr(layer, 'bias_regularizer') and layer.bias_regularizer:
            l2_cost += tf.math.reduce_sum(layer.bias_regularizer(layer.bias))

    total_cost = cost + l2_cost

    return total_cost
