#!/usr/bin/env python3
"""module 4"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """function that calculates the inverse of a matrix"""
    d = determinant(matrix)
    if d == 0:
        return None

    ad_matrix = adjugate(matrix)
    n = len(matrix)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            result[r][c] = ad_matrix[r][c] * ((-1) ** (r + c)) / d
    return result
