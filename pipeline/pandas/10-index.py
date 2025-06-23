#!/usr/env/bin python3
"""Module for index function"""


def index(df):
    """Function that takes a pd.DataFrame and:
        Sets the Timestamp column as the index of the dataframe
        Returns: the modified pd.DataFrame"""
    return df.set_index('Timestamp')
