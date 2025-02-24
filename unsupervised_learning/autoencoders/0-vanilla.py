#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """autoencoder
        input_dims: an integer containing the dimensions of the model input
        hidden_layers: a list containing the number of nodes for each hidden layer in the encoder
        latent_dims: an integer containing the dimensions of the latent space representation
    Returns: encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model"""

    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)
    encoded = keras.layers.Dense(units=latent_dims, activation='relu')(x)
    encoder_model = keras.models.Model(inputs=input_layer, outputs=encoded)

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)
    decoded = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder_model = keras.models.Model(inputs=decoder_input, outputs=decoded)

    auto = keras.Model(input_layer, decoder_model(encoder_model(input_layer)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto
