#!/usr/bin/env python3
"""convolutional"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims: an integer containing the dimenstions of the model input
    hidden_layers: a list containing the number of nodes for each hidden layer
    in the encoder, respectively
        hidden_layers should be reversed for the decoder
    latent_dims: an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder: the encoder model, which should output the latent
        representation, the mean, and the log variance
        decoder: the decoder model
        auto: the full autoencoder model
    
        The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
        All layers should use a relu activation except for the mean and log
        variance layers in the encoder, which should use None, and the last
        layer in the decoder, whouch should use sigmoid
    """
    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)
    encoded = keras.layers.Dense(units=latent_dims, activation='relu')(x)

    mean = keras.layers.Dense(units=latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(units=latent_dims, activation=None)(x)

    z =  keras.layers.Lambda(sampling)([mean, log_var])
    encoder_model = keras.models.Model(inputs=input_layer, outputs=[z, mean, log_var])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)
    decoded = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder_model = keras.models.Model(inputs=decoder_input, outputs=decoded)

    encoded_outputs = encoder_model(input_layer)
    z = encoded_outputs[0]
    mean = encoded_outputs[1]
    log_var = encoded_outputs[2]

    auto = keras.Model(input_layer, decoder_model(z))

    kl_loss = -0.5 * keras.backend.sum(1 + log_var - keras.backend.square(mean) - keras.backend.exp(log_var), axis=-1)

    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto

def sampling(args):
    """sampling"""
    mean, log_var = args
    batch_size = keras.backend.shape(mean)[0]
    dim = keras.backend.shape(mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch_size, dim))

    return mean + keras.backend.exp(log_var / 2) * epsilon
