#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
    cost (tensor): The cost of the network without L2 regularization
    model (keras.Model): A Keras model that includes layers with L2 regularization

    Returns:
    tensor: The total cost for each layer of the network, accounting for L2 regularization
    """
    cost_l2 = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            cost_l2 += tf.math.reduce_sum(layer.losses)

    total_cost = cost + cost_l2

    return total_cost
