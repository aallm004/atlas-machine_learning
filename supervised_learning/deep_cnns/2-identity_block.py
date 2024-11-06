#!/usr/bin/env python3
"""defines the identity_block method"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block
        A_prev is the output the previous layer
        filters is a tuple of list containing F11, F3, F12
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by batch
        normalization along the channels axis and a relu activation
        All weights should use he normal initialization
        The seeed for the he_normal initializer should be set to zero

        Returns: the activated output of the indentity block"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(0)

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                            padding='valid', kernel_initializer = init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(norm1)

    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same', kernel_initializer = init)(act1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(norm2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                            padding='valid', kernel_initializer = init)(act2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    add = K.layers.Add()([norm3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
