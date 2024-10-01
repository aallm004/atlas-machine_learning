#!/usr/bin/env python3
"""tensorflow for beginners"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_size=[] activations=[]):
    """function that creates the forward propagation graph for the neural network"""
    for i in range(len(layer_size)):
        if i == 0:
            prev = x
        prev = create_layer(prev, layer_size[i], activations[i])
    return prev
