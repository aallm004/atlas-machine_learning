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
        self.cov = (1 / n) * np.dot(center, center.T) / n
