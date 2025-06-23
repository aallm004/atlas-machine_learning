#!/usr/bin/env python3
"""Module for from_file function"""
import pandas as pd


def from_file(filename, delimiter):
    """function that loads data from a file as a pd.DataFrame
        filename: the file to load from
        delimiter: the column separator
        Returns: the loaded pd.DataFrame"""
    return pd.read_csv(filename, delimiter=delimiter)
