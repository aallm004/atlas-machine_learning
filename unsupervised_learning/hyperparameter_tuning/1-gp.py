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

        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

        return K

    def predict(self, X_s):
        """predicts the mean and stdv of points is a Gaussin process:
        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and stdv should be calculated
            s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for each
            point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the variance for
            each point in X_s, respectively"""
        if not isinstance(X_s, np.ndarray):
            raise TypeError("X_s must be a numpy.ndarray")
        if len(X_s.shape) != 2 or X_s.shape[1] != 1:
            raise ValueError("X_s must have shape (s, 1)")

        if self.X.shape[0] == 0:
            raise ValueError("No training data available")
        # covariance between sample points and training points
        K_train = self.kernel(self.X, X_s)

        # covariance between sample points
        K_sample = self.kernel(X_s, X_s)

        # inverse of training covariance matrix
        K_train_inverse = np.linalg.inv(self.K)

        # mean
        mu = K_train.T.dot(K_train_inverse).dot(self.Y).reshape(-1)

        sigma = np.diag(K_sample - K_train.T.dot(K_train_inverse).dot(K_train))

        return mu, sigma
