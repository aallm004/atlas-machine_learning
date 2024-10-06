#!/usr/bin/env python3
"""documentation"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """updates the learning rate using inverse time decay
    alpha is the original learning rate
    decay_rate the rate with which it decays
    global_step is the # of passes of GD that have elapsed
    decay_step # of passes of GD that should occur before alpha is
    decayed further
    """
    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2,
                                    epsilon=epsilon)
