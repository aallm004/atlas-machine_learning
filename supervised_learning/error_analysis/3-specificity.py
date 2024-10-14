#!/usr/bin/env python3
"""documentaion"""
import numpy as np


def specificity(confusion):
    """function that calculates the specificity for each class in a confusion
    matrix:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            classes is the number of classes
    Return: a numpy.ndarray of shape (classes,) containing the specificity of
    each class
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for i in range(classes):
        true_neg = np.sum(confusion) - np.sum(confusion[i, :]) - \
            np.sum(confusion[:, i]) + confusion[i, i]
        false_pos = np.sum(confusion[:, i]) - confusion[i, i]
        specificity[i] = true_neg / (true_neg + false_pos)

    return specificity
