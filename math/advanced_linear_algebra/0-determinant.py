#!/usr/bin/env python3
"""comments"""


def determinant(matrix):
    """Calculate the determinant of a matrix.
        matrix (list of lists): The matrix to calculate determinant for
        
    Returns: float/int: The determinant of the matrix
    """
    # Is matrix a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # empty matrix case
    if matrix == [[]]:
        return 1
    
    # Is matrix a square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    # 1x1 matrix
    if n == 1:
        return matrix[0][0]
    
    # 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Laplace expansion along first row
    det = 0
    for j in range(n):
        # Minor matrix excluding first row and current column
        minor = [[matrix[i][k] for k in range(n) if k != j] 
                 for i in range(1, n)]
        # Adds positive or negative to determinant by raising -1 to j
        det += matrix[0][j] * ((-1) ** j) * determinant(minor)
    
    return det
