#!/usr/bin/env python3
"""module for Hierarchy"""
import pandas as pd


def hierarchy(df1, df2):
    index = __import__('10-index').index

    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df1_filtered = df1_indexed[(df1_indexed.index >= 1417411980) &
                               (df1_indexed.index <= 1417417980)]
    df2_filtered = df2_indexed[(df2_indexed.index >= 1417411980) &
                               (df2_indexed.index <= 1417417980)]

    concatenated = pd.concat([df2_filtered, df1_filtered],
                             keys=['bitstamp', 'coinbase'])

    return concatenated.swaplevel().sort_index()
