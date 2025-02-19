#!/usr/bin/env python3
"""Module for initialize bayesian optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class constructor def __init__(self, f, X_init, Y_init, bounds,
    ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of the space
        in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed
        during acquisition
        l is the lnegth parameter for the kernel
        sigma_f is the standard deviation given to the output of the black-box
        function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be performed
        for minimization(True) or maximization(False)
        Sets the following public instance attributes:
            f: the black-box function
            gp: an instance of the class GaussianProcess
            X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
            acquisition sample points, evenly spaced between min and max
            xsi: the exploration-exploitation factor
            minimize: a bool for minimization versus maximisation"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Method that calculates the next best sample location:
            Uses the Expected Improvement acquisition function
                Returns: X_next, EI
                    X_next is a numpy.ndarray of shape (1,) representing the
                    next best sample point
                    EI is a numpy.ndarray of shape (ac_samples,) containing the
                    expected improvement of each potential sample"""
        from scipy.stats import norm

        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best_y = np.min(self.gp.Y)
        else:
            best_y = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            improve = (best_y - mu - self.xsi) if self.minimize else (mu - best_y - self.xsi)

        Z = np.zeros_like(sigma)
        mask = sigma > 0
        Z[mask] = improve[mask] / sigma[mask]

        ei = np.zeros_like(Z)
        mask_ei = mask & (sigma > 0)
        ei[mask_ei] = (improve[mask_ei] * norm.cdf(Z[mask_ei]) +
                      sigma[mask_ei] * norm.pdf(Z[mask_ei]))

        ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei
    
    def optimize(self, iterations=100):
        """method that optimizes the black-box function:
            iterations is the maximum number of iterations to perform
            If the next proposed point is one that has already been sampled,
            optimization should be stopped early
            
            Returns: X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,) representing the optimal
            point
            Y_opt is a numpy.ndarray of shape (1,) representing the optimal
            function value"""
        
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if any(np.allclose(X_next, x_existing) for x_existing in
                   self.gp.X):
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next.reshape(-1, 1), Y_next.reshape(-1, 1))

        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
        else:
            best_idx = np.argmax(self.gp.Y)

        return self.gp.X[best_idx], self.gp.Y[best_idx]
