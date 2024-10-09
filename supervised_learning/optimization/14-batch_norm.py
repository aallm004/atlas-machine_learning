#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """function that creates a batch normalization layer
    for a normalization layer for a neural network in
    tensorflow
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used
    on the output of the layer
    you should use the tf.keras.layers.Dense layer as the base
    layer with kernal initializer
    your layer should incorporate two trainable parameters, gamma
    and beta, initialized as vectors 1 and 0 respectively
    you should use an epsilon of 1e-7

    Returns: a tensor of the activated output for the layer
    """
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )(prev)

    mean, variance = tf.nn.moments(dense, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    bn = tf.nn.batch_normalization(
        dense,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    return activation(bn)
