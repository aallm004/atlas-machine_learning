#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix
    labels is a one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point
    m is the number of data points
    classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels

    Returns: a confusion numpy.ndarray of shape (classes, classes)
    with a row indices representing the correct labels and column
    indices representing the predicted labels
    """
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=float)
    for i in range(labels.shape[0]):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion[true_label, predicted_label] += 1

    return confusion
