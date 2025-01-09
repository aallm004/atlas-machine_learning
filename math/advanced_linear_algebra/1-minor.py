#!/usr/bin/env python3
"""module 1"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """function that calculates the minor matrix of a matrix
        matrix: list of lists whose minor matrix should be calculated
        Returns: the minor matrix of matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for
                                               row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if not all(len(row) == n for row in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    new_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for r in range(n):
        for c in range(n):
            x = sub_matrix(matrix, r, c)
            if x == [[]]:
                return [[1]]
            new_matrix[r][c] = determinant(x)
    return new_matrix


def sub_matrix(matrix, r, c):
    """creating a sub matrix"""
    n = len(matrix)
    if n == 1:
        return [[]]
    # Minor matrix excluding first row and current column
    return [[matrix[i][k] for k in range(n) if k != c]
            for i in range(n) if i != r]
