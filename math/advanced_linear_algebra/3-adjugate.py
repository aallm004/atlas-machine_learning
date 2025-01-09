#!/usr/bin/env python3
"""module 3"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """function that calcultes the adjugate matrix of a matrix"""
    cofactor_matrix = cofactor(matrix)
    n = len(matrix)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            result[c][r] = cofactor_matrix[r][c]
    return result
