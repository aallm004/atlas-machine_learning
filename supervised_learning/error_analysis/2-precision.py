#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def precision(confusion):
    """function that calculates the precision for each class in a confusion
    matrix:
        confusion: is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the precision of
    each class.
    """
    classes = confusion.shape[0]
    precision = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_neg = np.sum(confusion[:, i]) - true_positives

        precision[i] = true_positives / (true_positives + false_neg)

    return precision
