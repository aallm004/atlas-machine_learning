#!/usr/bin/env python3
"""Module for index"""
import pandas as pd


def concat(df1, df2):
    """Functoin that takes two pd.DataFrame objects and:
        Indexes both dataframes on their Timestamp columns
        Includes all timestamps from df2(bitstamp) up to and including
        timestamp 1417411920
        Concatenates the selected rows from df2 to the top of df1(coinbase)
        Adds keys to the concatenated data,  labeling the rows from df2 as
        bitstamp and the rows from df1 as coinbase
        Returns: the concatenated pd.DataFrame"""
    index = __import__('10-index').index

    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]

    return pd.concat([df2_filtered, df1_indexed],
                     keys=['bitstamp', 'coinbase'])
