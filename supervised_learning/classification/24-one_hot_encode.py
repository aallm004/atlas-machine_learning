#!/usr/bin/env python3
"""Module with one_hot_encode function"""
import numpy as np


def one_hot_encode(array):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(array, np.array):
        return None
    if array.ndim < 2:
        return None
    
    new_array = np.argmax(array, axis=0)
    return new_array
