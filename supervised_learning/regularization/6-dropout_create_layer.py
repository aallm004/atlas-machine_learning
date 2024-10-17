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
    tf.random.set_seed(1)

    initializer = tf.keras.initializers.GlorotUniform(seed=1)
    W = tf.Variable(initializer(shape=(prev.shape[1], n)))
    b = tf.Variable(tf.zeros([n]))
    
    Z = tf.matmul(prev, W) + b
    A = activation(Z)
    
    if training:
        mask = tf.cast(tf.random.uniform(tf.shape(A)) < keep_prob, A.dtype)
        A = tf.divide(tf.multiply(A, mask), keep_prob)
    
    return A
