#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """sets up the RMSProp optimization algorithm in Tensorflow
    alpha is the learning rate
    beta2 is the RMSProp weight (discounting factor)
    epsilon is a small number to avoid division by zero

    Returns: optimizer
    """

    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
