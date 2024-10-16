#!/usr/bin/env python3
"""documentaion"""
import numpy as np
import tensorflow as tf
import os
import random


def l2_reg_cost(cost, model):
    """function that calculates the cost of a neural  network with L2 
    regularization
    cost is a tensor containing the cost of the network without L2
    regularization
    model is a Keras model that includes layers with L2 regularization
    
    Returns: a tensor containing the total cost for each layer of the
    network, accounting for L2 regularization
    """
    cost_l2 = []

    for layer in model.layers:
        if hasattr(layer, "losses"):
            cost_l2.append(tf.add_n(layer.losses))
        else:
            cost_l2.append(0.0)

    return cost + tf.add_n(cost_l2)
