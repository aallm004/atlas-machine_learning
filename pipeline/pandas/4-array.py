#!/usr/bin/env python3
"""module for array function"""
import numpy as np
import pandas as pd


def array(df):
    """Function that takes a pd.DataFrame as input and performs the following:
        df is a pd.DataFrame containing columns named High and Close
        The function should select the last 10 rows of the High and Close col
        Covnert these selected values into a numpy.ndarray
        Returns: numpy array"""

    # Selecting the last 10 rows of the High and Close columns
    data = df[['High', 'Close']].tail(10)

    # Convert to nupy array
    return data.to_numpy()
