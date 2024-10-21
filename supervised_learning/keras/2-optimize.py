#!/usr/bin/env python3
"""documentation"""
import numpy as np
import tensorflow as tf


def optimize_model(network, alpha, beta1, beta2):
    """function that sets up Adam optimizer for a keras model with categorical
    crossentropy loss and accuracy metrics
    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    
    Returns: None
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2
    )
    train_op = optimizer.minimize(network.loss)
    return train_op, optimizer.variables()
