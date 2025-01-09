#!/usr/bin/env python3
"""module 2"""
determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
sub_matrix = __import__('1-minor').sub_matrix


def cofactor(matrix):
    """function that calculates the cofactor matrix of a matrix
    
    """
    n = len(matrix)
    result = [[0 for _ in range(n)] for _ in range(n)]
    new_matrix = minor(matrix)
        
    for r in range(n):
        for c in range(n):
            x = sub_matrix(matrix, r, c)
            if x == [[]]:
                return [[1]]
            result[r][c] = determinant(x) * ((-1) ** (r + c))
    return result

