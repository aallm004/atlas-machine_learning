#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function that creates a neural network layer in tensorflow that includes L2
    regularization
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
        Returns: the output of the new layer
    """
    initialize = tf.keras.initializers.\
        VarianceScaling(scale=2.0, mode=("fan_avg"))
    regularize = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initialize,
        kernel_regularizer=regularize,
    )(prev)
    return layer
