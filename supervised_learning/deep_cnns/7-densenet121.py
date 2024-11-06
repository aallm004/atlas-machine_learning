#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture
        growth_rate is the growth rate
        compression is the compression factor
        You can assume the input data will have shape (224, 224, 3)
        All convolutions should be preceded by Batch Normalization and a
        ReLU activation
        All weights should use he normal initialization
        The seed for the he_normal intializer should be set to zero

        Returns: the keras model
        """
    input_layer = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=0)

    norm1 = K.layers.BatchNormalization(axis=3)(input_layer)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(64, pool_size=(7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=init)(act1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(conv1)

    dense_block1, nb_filters = dense_block(pool1, nb_filters=64,
                                           growth_rate=growth_rate, layers=6)

    transition_layer1, nb_filters = transition_layer(dense_block1, nb_filters,
                                                     compression=compression)

    dense_block2, nb_filters = dense_block(transition_layer1,
                                           nb_filters=nb_filters,
                                           growth_rate=growth_rate, layers=12)

    transition_layer2, nb_filters = transition_layer(dense_block2, nb_filters,
                                                     compression=compression)

    dense_block3, nb_filters = dense_block(transition_layer2,
                                           nb_filters=nb_filters,
                                           growth_rate=growth_rate, layers=24)

    transition_layer3, nb_filters = transition_layer(dense_block3, nb_filters,
                                                     compression=compression)

    dense_block4, nb_filters = dense_block(transition_layer3,
                                           nb_filters=nb_filters,
                                           growth_rate=growth_rate, layers=16)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7))(dense_block4)

    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avg_pool)

    model = K.Model(inputs=input_layer, outputs=output)

    return model
