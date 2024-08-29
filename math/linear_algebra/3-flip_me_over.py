#!/usr/bin/env python3
"""function that returns the transpose of a 2d matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2d matrix"""
    matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        matrix.append(row)
    return matrix
