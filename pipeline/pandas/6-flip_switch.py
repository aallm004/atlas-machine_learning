#!/usr/bin/env python3
"""module for flip_switch function"""


def flip_switch(df):
    """Function that takes a pd.DataFrame and:
        Sorts the data in reverse chronological order
        Transposes the sorted dataframe
        Returns: the transformed pd.DataFrae"""
    sort = df.sort_index(ascending=False)

    return sort.T
