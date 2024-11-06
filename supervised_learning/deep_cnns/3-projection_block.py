#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection bock
        A_prev is the output de the previous layer
        filters is a tuple or list containing F11, F3, F12
            F11 is the number of filters in the first 1x1
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection
        s is the stride of the first convolution in both the main path and the
        shortcut connection
        All convolutions inside the block should be followed by batch
        normalization along the channels axis and a ReLU activation
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero

        Returns: the activated output of the projection block"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                            padding='valid', kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(norm1)

    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', kernel_initializer=init)(act1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(norm2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                            padding='valid', kernel_initializer=init)(act2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut_conv = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                    strides=(s, s), padding='valid',
                                    kernel_initializer=init)(A_prev)
    shortcut_norm = K.layers.BatchNormalization(axis=3)(shortcut_conv)

    add = K.layers.Add()([norm3, shortcut_norm])
    output = K.layers.Activation('relu')(add)

    return output
