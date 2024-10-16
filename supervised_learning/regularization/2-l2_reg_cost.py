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
    l2_costs = []
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
            l2_costs.append(tf.reduce_sum(layer.losses))

    while len(l2_costs) < 3:
        l2_costs.append(tf.constant(0.0))

    l2_costs = l2_costs[:3]

    total_costs = cost + tf.stack(l2_costs)

    return total_costs
