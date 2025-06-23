#!/usr/bin/env python3
"""Module for analyze function"""


def analyze(df):
    """Function that takes a pd.DataFrame and:
        Computes descriptive statistics for all columns except the Timestamp
        Returns: new pd.DataFrame containing these statistics"""
    return df.drop(columns=['Timestamp']).describe()
