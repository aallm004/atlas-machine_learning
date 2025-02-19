#!/usr/bin/env python3
"""Module for Bayesian optimization with multiple acquisition functions"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization with customizable acquisition functions"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, ac_type='ei',
                 length_param=1, sigma_f=1, xsi=0.01, minimize=True,
                 xi=0.01, kappa=2.576):
        """Initialize Bayesian optimization
        Args:
            f: black-box function to optimize
            X_init: numpy.ndarray (t, 1) inputs already sampled
            Y_init: numpy.ndarray (t, 1) outputs from f(X_init)
            bounds: tuple (min, max) search space bounds
            ac_samples: number of acquisition points to consider
            ac_type: acquisition function type ('ei', 'ucb', or 'poi')
            length_param: kernel length parameter
            sigma_f: kernel signal variance
            xsi: acquisition trade-off parameter
            minimize: bool whether to minimize (True) or maximize (False)
            xi: exploration parameter for PI acquisition
            kappa: trade-off parameter for UCB acquisition
        """
        self.f = f
        self.gp = GP(X_init, Y_init, length_param, sigma_f)
        self.bounds = bounds
        self.ac_type = ac_type
        self.minimize = minimize
        self.xi = xi
        self.kappa = kappa

        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound,
                              ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.best_y = None
        self.iteration = 0
        self.history = []

    def acquisition(self):
        """Calculate next best sample location using chosen acquisition function"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best_y = np.min(self.gp.Y)
        else:
            best_y = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            if self.minimize:
                improve = best_y - mu - self.xsi
            else:
                improve = mu - best_y - self.xsi

            Z = improve / (sigma + 1e-9)
            ei = improve * norm.cdf(Z) + sigma * norm.pdf(Z)

            ei[sigma < 1e-9] = 0

        return self.X_s[np.argmax(ei)], ei

    def optimize(self, iterations=100):
        """Optimize the black-box function"""
        for i in range(iterations):
            X_next, ei = self.acquisition()

            if any(np.allclose(X_next, x_existing)
                  for x_existing in self.gp.X):
                break

            Y_next = self.f(X_next)
            if not isinstance(Y_next, np.ndarray):
                Y_next = np.array(Y_next)

            # Store iteration history
            self.history.append({
                'iteration': i + 1,
                'X': X_next,
                'Y': Y_next,
                'EI': np.max(ei)
            })

            self.gp.update(X_next.reshape(-1, 1), Y_next.reshape(-1, 1))

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt].flatten()
        Y_opt = self.gp.Y[idx_opt].flatten()

        # Save optimization report
        with open('bayes_opt.txt', 'w') as f:
            f.write("Bayesian Optimization Report\n")
            f.write("===========================\n\n")
            
            f.write("Configuration:\n")
            f.write(f"Optimization Type: {'Minimization' if self.minimize else 'Maximization'}\n")
            f.write(f"Search Space Bounds: {self.bounds}\n")
            f.write(f"Number of Samples: {len(self.X_s)}\n\n")
            
            f.write("Results:\n")
            f.write(f"Optimal X: {X_opt}\n")
            f.write(f"Optimal Y: {Y_opt}\n")
            f.write(f"Total Iterations: {len(self.history)}\n\n")
            
            f.write("Optimization History:\n")
            f.write("-----------------------\n")
            for entry in self.history:
                f.write(f"\nIteration {entry['iteration']}:\n")
                f.write(f"X = {entry['X'][0]:.6f}\n")
                f.write(f"Y = {entry['Y'][0]:.6f}\n")
                f.write(f"Expected Improvement = {entry['EI']:.6f}\n")

        return X_opt, Y_opt
