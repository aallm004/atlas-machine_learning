#!/usr/bin/env python3
"""Module for from_numpy function"""
import pandas as pd


def from_numpy(array):
    """function that creates a pd.DataFrame from a np.ndarray
        array is the np.ndarray from which you should create the pd.DataFrame
        The columns of the pd.DataFrame should be labeled in alphabetical order
        and capitalized. There will not be more than 26 columns
        Returns: the newly created pd.DataFrame"""
    num_cols = array.shape[1] if array.ndim > 1 else 1
    column_names = [chr(ord('A') + i) for i in range(num_cols)]

    return pd.DataFrame(array, columns=column_names)
