#!/usr/bin/env python3
"""module 1"""
import numpy as np


def correlation(C):
    """function that calculates a correlation matrix"""

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2 square matrix")

    diag = np.diagonal(C)

    std_dev = np.sqrt(diag)

    correlation = C / np.outer(std_dev, std_dev)

    return correlation
