#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """
    Function that creates a layer of a neural network using dropout
        prev is the previous layer
        n is the number of nodes in the new layer
        activation is the activation function to be used
        keep_prob is the probability that a node will be kept
        training is a boolean indicating whether the layer is in training mode
        Returns: the output of the new layer
    """

    initializer = tf.keras.initializers.GlorotUniform()
    W = tf.Variable(initializer(shape=(prev.shape[1], n)), name="W")
    b = tf.Variable(tf.zeros(shape=(1, n)), name="b")
    
    Z = tf.matmul(prev, W) + b
    A = activation(Z)
    
    if training:
        A = tf.nn.dropout(A, rate=1 - keep_prob)
    
    return A
