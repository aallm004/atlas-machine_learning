#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network
    input data will have shape (244, 244, 3)
    All convolutions inside and outside the inception block should use
    a ReLU activation
    
    Returns: the keras model"""
    input = K.Input(shape=(244, 244, 3))
    init = K.initializer.he_normal()

    conv = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=init)(input)

    max_pool = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv)

    conv2 = K.layers.Conv2D(64, (1, 1), activation='relu', kernel_initializer=init)(max_pool)

    conv3 = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu', kernel_initializer=init)(conv2)

    max_pool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(conv3)

    inception = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    inception1= inception_block(inception, [128, 128, 192, 32, 96, 64])

    max_pool3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception1)

    inception2a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    inception2b = inception_block(inception2a, [160, 112, 224, 24, 64, 64])

    inception2c = inception_block(inception2b, [128, 128, 256, 12, 64, 64])

    inception2d = inception_block(inception2c, [112, 144, 288, 32, 64, 64])

    inception2e = inception_block(inception2d, [256, 160, 320, 32, 128, 128])

    max_pool4 = K.layers.MaxPooling2d((3, 3), strides=(2, 2), padding='same')(inception2e)

    inception3a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    inception3b = inception_block(inception3a, [384, 192, 384, 48, 128, 128])

    pool_avg = K.layers.GlobalAveragePooling2d()(inception3b)

    dropout = K.layers.dropout(0.4)(pool_avg)

    output_layer = K.layers.Dense(1000, activations='softmax', kernel_initializer=init)(dropout)

    model = K.model(input=input, outputs=output_layer)
    
    return model
