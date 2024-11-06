#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Function that buids a transition layer
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        compression is the compression factor for the transition layer
        Code should implement compression as used in DenseNet-C
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero
        All convolutions should be preceded by Batch Normalization and a
        ReLU activation

        Returns: The output of the transition layer and the number of filters
        within the output"""
    init = K.initializers.he_normal(seed=0)

    batchnorm1 = K.layers.BatchNormalization(axis=3)(X)
    relu1 = K.layers.Activation('relu')(batchnorm1)

    compressed_filters = int(nb_filters * compression)
    conv1 = K.layers.Conv2D(compressed_filters, (1, 1), padding='same',
                            kernel_initializer=init)(relu1)

    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same')(conv1)

    return avg_pool, compressed_filters
