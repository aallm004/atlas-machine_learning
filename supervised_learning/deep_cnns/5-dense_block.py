#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block
        X is the output from the previous layer
        nb_filters is an i nteger representing the number of filters in X
        grownth_rate is the growth rate for the dense bloack
        layers is the number of layers in the dense block
        You should use the bottleneck layers used for DenseNet-B
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
        All convolutions should be preceded by Batch Normalization and a
        ReLU activation

        Returns: The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs"""

    init = K.initializers.he_normal(seed=0)

    for i in range(layers):
        batchnorm1 = K.layers.BatchNormalization(axis=3)(X)
        relu1 = K.layers.Activation('relu')(batchnorm1)

        conv1 = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                                kernel_initializer=init)(relu1)

        batchnorm2 = K.layers.BatchNormalization(axis=3)(conv1)
        relu2 = K.layers.Activation('relu')(batchnorm2)

        conv2 = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                kernel_initializer=init)(relu2)

        X = K.layers.Concatenate(axis=3)([X, conv2])

        nb_filters += growth_rate

    return X, nb_filters
