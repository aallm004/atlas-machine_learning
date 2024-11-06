#!/usr/bin/env python3
"""documentation"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-5 architecture
        Assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the blocks should be followed by
        batch normalization along the channels axis and a ReLU activation
        All weights should use he normal initialization
        The seed for the he_normal initializer should be set to zero

        Returns: the keras model
            """
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=init)(input)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(act1)

    proj_block1 = projection_block(pool1, [64, 64, 256], s=1)
    id_block1 = identity_block(proj_block1, [64, 64, 256])
    id_block2 = identity_block(id_block1, [64, 64, 256])

    proj_block2 = projection_block(id_block2, [128, 128, 512], s=2)
    id_block3 = identity_block(proj_block2, [128, 128, 512])
    id_block4 = identity_block(id_block3, [128, 128, 512])
    id_block5 = identity_block(id_block4, [128, 128, 512])

    proj_block3 = projection_block(id_block5, [256, 256, 1024], s=2)
    id_block6 = identity_block(proj_block3, [256, 256, 1024])
    id_block7 = identity_block(id_block6, [256, 256, 1024])
    id_block8 = identity_block(id_block7, [256, 256, 1024])
    id_block9 = identity_block(id_block8, [256, 256, 1024])
    id_block10 = identity_block(id_block9, [256, 256, 1024])

    proj_block4 = projection_block(id_block10, [512, 512, 2048], s=2)
    id_block11 = identity_block(proj_block4, [512, 512, 2048])
    id_block12 = identity_block(id_block11, [512, 512, 2048])

    avg_pool = K.layers.GlobalAveragePooling2D()(id_block12)

    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avg_pool)

    model = K.Model(inputs=input, outputs=output)

    return model
