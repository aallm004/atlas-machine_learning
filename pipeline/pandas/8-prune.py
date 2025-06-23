#!/usr/bin/env python3
"""Module for prune function"""
import pandas as pd


def prune(df):
    """Function that takes a pd.DataFrame and:
        Removes any entries where Close has NaN values
        Returns: the modified pd.DataFrame"""
    return df.dropna(subset=['Close'])
