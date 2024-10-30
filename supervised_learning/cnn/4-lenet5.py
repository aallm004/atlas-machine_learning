#!/usr/bin/env python3
"documentation"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5 architecture
    using
    tensorflow
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
    for the network
        m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
    the network
    The model should consist of the following layers in order:
        Convolutional layer with 6 filters of shape 5x5 with same padding
        Max pooling layer with a pool size of 2 and stride of 2
        Convolutional layer with 16 filters of shape 5x5 with valid padding
        Max pooling layer with a pool size of 2 and stride of 2
        Flatten layer
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Output layer with 10 nodes (no activation)
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method
    All hidden layers should use the ReLU activation function
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the network
        a tensor for the accuracy of the network"""
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    flat = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=he_normal)

    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=he_normal)

    logits = tf.layers.dense(fc2, units=10, kernel_initializer=he_normal)
    output = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                          (logits=logits, labels=y))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, train_op, loss, accuracy
