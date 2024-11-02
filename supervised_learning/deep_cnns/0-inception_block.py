#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block as described in a website.
        A_prev is the output from the previous layer
        filters is a tuple or list containing F1, F3R, F3, F5R, F5, FPP, respectively.
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of fiters in the 1x1 convolution before the 3x3 convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution after the max pooling
        All convolutions inside the inceptoin block should use a rectified linear activation(ReLU)
        
        Returns: the concatenated output of the inception block"""
    
    F1, F3R, F3, F5R, F5, FPP = filters
    initializer = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=initializer)(A_prev)
    conv3r = K.layers.Conv2D(filters= F3R, kernel_size=(1, 1), padding='same',
                            activation='relu', kernel_initializer=initializer)(A_prev)
    conv3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             activation='relu', kernel_initializer=initializer)(conv3r)
    conv5r = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=initializer)(A_prev)
    
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=initializer)(conv5r)
    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)

    convpp = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=initializer)(pool)
    output = K.layers.concatenate([conv1, conv3, conv5, convpp], axis=-1)

    return output
    
