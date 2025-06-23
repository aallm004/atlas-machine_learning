#!/usr/bin/env python3
"""module for rename function"""
import pandas as pd


def rename(df):
    """function that takes a pd.DataFrame as input and performs the following
        df: pd.DataFrame containing a column named Timestamp
        The function should rename the Timestamp column to Datetime
        Convert the timestamp values to datatime values
        Display only the Datetime and Close column
        Returns: the modified pd.DataFrame"""

    # Rename Timestamp column to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Conversion of values
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    return df[['Datetime', 'Close']]
