#!/usr/bin/env python3
"""module for initialize gaussian process"""
import numpy as np


class GaussianProcess:
    """Class that represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """a noiseless 1D Gaussian process:
        X_init: numpy.ndarray of shape (t, 1) representing the inputs already
        sampled with the black-box function
        Y_init: numpy.ndarray of shape (t, 1)" representing the outputs of the
        black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the black-box
        function
        """
        if not isinstance(X_init, np.ndarray):
            raise TypeError("X_init must be a numpy.ndarray")
        if not isinstance(Y_init, np.ndarray):
            raise TypeError("Y_init must be a numpy.ndarray")

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between to matrices:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)
            the kernel should use Radial Basis Function(RBF)
        Returns: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)"""
        X1_sq = np.sum(X1**2, 1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, 1)
        mult = np.dot(X1, X2.T)

        sqdist = X1_sq + X2_sq - 2 * mult

        self.K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

        return self.K
