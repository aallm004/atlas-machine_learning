#!/usr/bin/env python3
"""documentation"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the keras libray
       nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the
    network
    activations is a list containing the activation functions used for each
    layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Input class

    Returns: the keras model """
    model = K.Sequential()

    for i, layer_size in enumerate(layers):
        if i == 0:
            model.add(K.layers.Dense(
                layer_size,
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(
                layer_size,
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
