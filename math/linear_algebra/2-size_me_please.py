#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Function to calculate the shape of a matrix.
    """
    # Initialize an empty list to store the shape
    shape = []

    # Iterate until the matrix is not a list (i.e., a scalar value)
    while isinstance(matrix, list):
        # Append the length of the current list to the shape
        shape.append(len(matrix))

        # Move to the next nested list
        matrix = matrix[0]

    # Return the shape
    return shape"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
