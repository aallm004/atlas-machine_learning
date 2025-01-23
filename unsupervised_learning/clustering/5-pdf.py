#!/usr/bin/env python3
"""module for pdf"""
import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density function of a Gaussian
    distribution:
        X is a numpy.ndarray of shape (n, d) containing the data points whose
        PDF should be evaluated
        m is a numpy.ndarray of shape (d,) containing the mean of the
        distribution
        S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
        Returns: P, or None on failure
            P is a numpy.ndarray of shape (n,) containing the PDF values for
            each data point
        All values in P should have a minimum value of 1e-300"""
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(m, np.ndarray) or len(m.shape) != 1:
            return None
        if not isinstance(S, np.ndarray) or len(S.shape) != 2:
            return None

        n, d = X.shape

        if d != m.shape[0] or S.shape != (d, d):
            return None

        # Centered data points
        centered = X - m

        # Determinant and inverse of covariance matrix
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)

        if det_S <= 0:
            return None

        # Calculate exponent term
        exponent = -0.5 * np.sum(np.matmul(centered, inv_S) * centered, axis=1)

        # Calculate normalization constant
        normalize = 1 / (np.sqrt((2 * np.pi) ** d * det_S))

        # Calculate PDF: normalize * exp(exponent)
        P = normalize * np.exp(exponent)

        # Set min value to 1e-300
        P = np.maximum(P, 1e-300)

        return P

    except Exception:
        return None
