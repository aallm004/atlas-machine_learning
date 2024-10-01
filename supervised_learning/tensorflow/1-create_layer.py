#!/usr/bin/env python3
"""tensorflow for beginners"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
        i = tf.keras.initializers.VarianceScaling(mode='fan_avg')

        layer = tf.keras.layers.Dense(
            units=n,
            activation=activation,
            kernel_initializer=i,
            name="layer"
        )

        return layer(prev)
