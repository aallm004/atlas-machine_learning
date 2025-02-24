#!/usr/bin/env python3
"""convolutional"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """convolutional autoencoder
    input_dims: tuple of integers containing the dimensions of the model input
    filters: list containing the number of filters for each convolutional layer
    in the encoder, respectively
        filters should be reversed for decoder
    latent_dims: tuple of integers containing the dimensions of the latent
    space representation
    Each convolution in the ENCODER should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the DECODER, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed by
    upsampling size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation and no
        upsampling
    Returns: encoder, decoder, auto"""

    input_layer = keras.layers.Input(shape=input_dims)
    x = input_layer
    for n_filters in filters:
        x = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    encoder = keras.models.Model(inputs=input_layer, outputs=x)

    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input

    rev_filters = list(reversed(filters))
    for n_filters in rev_filters[:-2]:
        x = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=rev_filters[-2], kernel_size=(3, 3),
                            padding='valid', activation='relu')(x)

    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                            padding='same', activation='sigmoid')(x)

    decoder = keras.models.Model(inputs=decoder_input, outputs=x)

    encoded_output = encoder(input_layer)

    decoded_output = decoder(encoded_output)

    auto = keras.Model(inputs=input_layer, outputs=decoded_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
