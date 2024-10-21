#!/usr/bin/env python3
import tensorflow.keras as k


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
    inputs = tf.keras.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = tf.keras.layers.Dense(layers[i], activation=activations[i],
                                  kernel_regularizer=tf.keras.regularizers.l2(lambtha))(x)
        if i < len(layers) - 1:
            x = tf.keras.layers.Dropout(1 - keep_prob)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
