#!/usr/bin/env python3
"""sparce"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """sparse autoencoder
        input_dims: an int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
        hidden_layer in the encoder, respectively
        latent_dims: is an integer containing the dimensions of the latent
        space representation
        lambtha: regularization parameter used for L1 regularizaion on the
        encoded output
            Returns: encoder, decoder, auto
                encoder: encoder model
                decoder: decoder model
                auto: sparse autoencoder model"""

    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)
    encoded = keras.layers.Dense(
        units=latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(x)
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
