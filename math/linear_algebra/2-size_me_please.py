#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Function to calculate the shape of a matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
