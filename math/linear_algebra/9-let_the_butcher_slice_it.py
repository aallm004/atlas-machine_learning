#!/usr/bin/env python3
"""Complete the following source code (found below):

mat1 should be the middle two rows of matrix
mat2 should be the middle two columns of matrix
mat3 should be the bottom-right, square, 3x3 matrix of matrix
You are not allowed to use any loops or conditional statements
Your program should be exactly 10 lines"""
import numpy as np

matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 = matrix[1:3]
mat2 = matrix[:, 2:4]
mat3 = matrix[-3:, -3:]
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
