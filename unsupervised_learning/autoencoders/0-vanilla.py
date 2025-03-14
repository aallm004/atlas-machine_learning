#!/usr/bin/env python3
"""vanilla ice cream"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """autoencoder
        input_dims: an integer containing the dimensions of the model input
        hidden_layers: a list containing the number of nodes for each hidden
        layer in the encoder
        latent_dims: an integer containing the dimensions of the latent space
        representation
    Returns: encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model"""

    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)
    encoded = keras.layers.Dense(units=latent_dims, activation='relu')(x)

    mean = keras.layers.Dense(units=latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(units=latent_dims, activation=None)(x)

    z =  keras.layers.Lambda(sampling)
    encoder_model = keras.models.Model(inputs=input_layer, outputs=[z, mean, log_var])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)
    decoded = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder_model = keras.models.Model(inputs=decoder_input, outputs=decoded)

    auto = keras.Model(input_layer, decoder_model(encoder_model(input_layer)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto

def sampling(args):
    """sampling"""
    mean, log_var = args
    batch_size = keras.backend.shape(mean)
    dim = keras.backend.shape(mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch_size, dim))

    return mean + keras.backend.exp(log_var / 2) * epsilon
