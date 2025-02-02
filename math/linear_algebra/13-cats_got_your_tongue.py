#!/usr/bin/env python3
"""function that concats two matrices along on a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specified axis"""
    return np.concatenate((mat1, mat2), axis=axis)
