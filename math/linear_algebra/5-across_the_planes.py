#!/usr/bin/env python3
"""function that adds two martrices elemt-wise"""


def add_matrices2D(mat1, mat2):
    """adds two martrices elemt-wise"""
    added = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        added.append(row)
    return added
