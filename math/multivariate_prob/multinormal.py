#!/usr/bin/env python3
"""module 3"""
import numpy as np


class MultiNormal:
    """class that rpresents a Multivariate Normal distribution    """

    def __init__(self, data):
        """initialization of the NultiNormal distribution"""

        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")

        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Find mean
        self.mean = np.mean(data, axis=1).reshape(d, 1)

        # Calculate covariance and center data
        center = data - self.mean

        # Calculate covariance matrix
        self.cov = (1.0 / (n - 1)) * np.matmul(center, center.T)

    def pdf(self, x):

        d = x.cov.shape[0]

        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray")

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # deviation from mean
        diff = x - self.mean

        # get determinant and inverse
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # calculating quadratic term in exponent
        exp = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)

        # Calculating normalizating constant
        norm = 1 / (np.sqrt((2 * np.pi) ** d * det))

        # combining terms to get pdf value
        pdf = float(norm * np.exp(exp))

        return pdf
