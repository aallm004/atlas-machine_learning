#!/usr/bin/env python3
"""Module with one_hot_encode function"""
import numpy as np


def one_hot_encode(encoded):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(encoded, np.ndarray):
        return None
    if encoded.ndim < 2:
        return None
    
    new_array = np.argmax(encoded, axis=0)
    return new_array
