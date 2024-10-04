#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """sets up gradient descent with momentum optimization algorithm in TF

    alpha is the learning rate
    beta1 is the momentum weight

    Returns: optimizes
    """
    chosen = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return chosen
