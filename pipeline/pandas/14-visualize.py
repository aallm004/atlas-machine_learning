#!/usr/bin/env python3
"""Module for Visualize"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the dataframe on Date
df = df.set_index('Date')

# Fill missing values in Close with prior row value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill missing values in High, Low, Open with same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Fill missing values in Volume columns with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 on
df_2017 = df[df.index >= '2017-01-01']

# Group by date and aggregate
df_daily = df_2017.groupby(df_2017.index.date).agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Convert index back to datetime for proper date handling
df_daily.index = pd.to_datetime(df_daily.index)

print(df_daily)
