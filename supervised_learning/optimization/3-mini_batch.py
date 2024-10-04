#!/usr/bin/env python3
"""documentation"""
import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data

def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a nn using mini-batch gd

    X is a numpy.ndarray of shape (m, nx) representing input data
        m is the number of data points
        nx is the number of features in X
    Y is a numpy.ndarray of shape (m, ny) representing labels
        m is the same number of data points as in X
        ny is the number of classes for classification tasks
    batch_size is the number of data points in a batch

    Returns: list of mini_batches containing tuples(X_batch, Y_batch)
    """

    m = X.shape[0]

    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    complete_batches = m // batch_size

    mini_batches = []

    for i in range(complete_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start = complete_batches * batch_size
        X_batch = X_shuffled[start:]
        Y_batch = Y_shuffled[start:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches 
