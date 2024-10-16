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
    reg_loss = tf.add_n(model.losses)
    total_cost = cost + reg_loss

    return total_cost
