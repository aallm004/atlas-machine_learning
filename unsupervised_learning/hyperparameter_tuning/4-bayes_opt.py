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

        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)

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

        posterior_mean, posterior_std = self.gp.predict(self.X_s)

        if self.minimize:
            best_y = np.min(self.gp.Y)
        else:
            best_y = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            if self.minimize:
                improve = best_y - posterior_mean - self.xsi
            else:
                improve = posterior_mean - best_y - self.xsi

        norm_imp = np.zeros_like(posterior_std)
        valid_std_mask = posterior_std > 0
        num = improve[valid_std_mask]
        dem = posterior_std[valid_std_mask]
        norm_imp[valid_std_mask] = num / dem

        ll = num * norm.cdf(norm_imp[valid_std_mask])
        rr = dem * norm.pdf(norm_imp[valid_std_mask])
        expected_improvement = np.zeros_like(norm_imp)
        expected_improvement[valid_std_mask] = ll + rr

        expected_improvement[posterior_std == 0.0] = 0

        next_sample_point = self.X_s[np.argmax(expected_improvement)]

        return next_sample_point, expected_improvement
