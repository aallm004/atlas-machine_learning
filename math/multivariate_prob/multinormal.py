#!/usr/bin/env python3
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
        mean = np.mean(data, axis=1)

        # Calculate covariance and center data
        center = data - mean

        # Calculate covariance matrix
        covariance = (1 / (n)) * (center @ center.T)

        self.mean = mean
        self.covariance = covariance
