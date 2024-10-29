#!/usr/bin/env python3
"""documentation"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5 architecture
    using tensorflow:
        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
        images for the network
            m is the number of images
        y is a tf.placeholder of shape (m, 10) containing the one-hot labels
        for the network
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
        All layers requiring initialization should initialize their kernels
        with the he_normal initialization method:
        tf.keras.initializers.VarianceScaling(scale=2.0)
        All hidden layers requiring activation should use the relu activation
        function
        Returns:
            a tensor for the softmax activated output
            a training operation that utilizes Adam optimization (with default
            hyperparameters)
            a tensor for the loss of the network
            a tensor for the accuracy of the network
    """
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = tf.layers.Flatten()(pool2)

    fcl1 = tf.layers.Dense(flat, units=120, activation=tf.nn.relu,
                           kernel_initializer=he_normal)

    fcl2 = tf.layers.Dense(fcl1, units=84, activation=tf.nn.relu,
                           kernel_initializer=he_normal)

    output = tf.layers.Dense(fcl2, units=10, activation=tf.nn.softmax,
                             kernel_initializer=he_normal)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, optimizer, loss, accuracy
