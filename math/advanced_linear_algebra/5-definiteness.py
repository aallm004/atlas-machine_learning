#!/usr/bin/env python3
"""module 5"""
import numpy as np


def definiteness(matrix):
    """write a function that calculates the definiteness of a matrix
        matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
        calculated
        Return: the string Positive definite, positive semi-definite, negative
        semi-defininte, negative defininte, or indefininte if the matrix is one
        of those, respectively
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
