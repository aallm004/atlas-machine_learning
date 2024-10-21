#!/usr/bin/env python3
"""documentation"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """function that conerts a label vector into a one-hot matrix
        The last dimension of the one-hot matrix must be the number of classes

        Returns: one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
