#!/usr/bin/env python3
"""Module for number 1"""
import pandas as pd


# Data for the first column
first_column_data = [0.0, 0.5, 1.0, 1.5]

# Data for the second column
second_column_data = ['one', 'two', 'three', 'four']

# Labels for the rows
row_labels = ['A', 'B', 'C', 'D']

# Dictionary for placement of data
data = {
    'First': first_column_data,
    'Second': second_column_data
}

df = pd.DataFrame(data, index=row_labels)
