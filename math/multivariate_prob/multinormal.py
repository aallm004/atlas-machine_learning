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
        self.mean = np.sum(data, axis=1).reshape(d, 1) / n

        # Calculate covariance and center data
        center = data - self.mean

        # Calculate covariance matrix (ASK SAJID)
        self.cov = np.dot(center, center.T) / (n - 1)
        # self.cov = (1.0 / (n - 1)) * np.matmul(center, center.T)

    def pdf(self, x):
        """calculates the PDF at a data"""

        d = self.mean.shape[0]

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # get determinant and inverse
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # deviation from mean
        diff = x - self.mean

        # calculating quadratic term in exponent
        exp = -0.5 * np.dot(np.dot(diff.T, inv), diff)

        # Calculating normalizating constant
        norm = 1 / np.sqrt((2 * np.pi) ** d * det)

        # combining terms to get pdf value
        pdf = float(norm * np.exp(exp))

        return pdf
