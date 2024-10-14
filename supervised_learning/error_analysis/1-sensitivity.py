#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def sensitivity(confusion):
    """function that calculates the sensitivity
    for each class in a confusion matrix
    confusion is a confusion numpy.ndarray of
    shape (classes, classes) where row indices represent the
    correct labels and column indices represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containint the sensitivity
    of each class
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_neg = np.sum(confusion[i, :])

        sensitivity[i] = true_positives / false_neg

    return sensitivity
