#!/usr/bin/env python3
"""defines the identity_block method"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block
        A_prev is the output from the previous layer
        filters is a tuple of list containing F11, F3, F12
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by batch
        normalization along the channels axis and a relu activation
        All weights should use he normal initialization
        The seeed for the he_normal initializer should be set to zero
        
        Returns: the activated output of the indentity block"""
    pass
