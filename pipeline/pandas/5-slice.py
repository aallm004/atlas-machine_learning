#!/usr/bin/env python3
"""Module for slice function"""


def slice(df):
    """Function that takes a pd.DataFrame and:
        Extracts the columns High, Low, Close, and Volume_BTC
        Selects every 60th row from these columns
        Returns: the sliced pd.DataFrame"""
    columns = df[['High', 'Low', 'Close', 'Volume_BTC']]

    return columns[::60]
