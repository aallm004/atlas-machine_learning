#!/usr/bin/env python3
"""Module for Bayesian optimization with multiple acquisition functions"""
import numpy as np
from scipy.stats import norm
import os
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization with customizable acquisition functions"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, ac_type='ei',
                 length_param=1, sigma_f=1, xsi=0.01, minimize=True,
                 xi=0.01, kappa=2.576):
        """Initialize Bayesian optimization"""
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
        self.optimization_history = []

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

    def _expected_improvement(self, mu, sigma):
        """Calculate Expected Improvement acquisition function"""
        if self.best_y is None:
            self.best_y = (np.min(self.gp.Y) if self.minimize
                          else np.max(self.gp.Y))

        if self.minimize:
            improve = self.best_y - mu - self.xsi
        else:
            improve = mu - self.best_y - self.xsi

        Z = improve / (sigma + 1e-9)
        return improve * norm.cdf(Z) + sigma * norm.pdf(Z)

    def _probability_improvement(self, mu, sigma):
        """Calculate Probability of Improvement acquisition function"""
        if self.best_y is None:
            self.best_y = (np.min(self.gp.Y) if self.minimize
                          else np.max(self.gp.Y))

        if self.minimize:
            improve = self.best_y - mu - self.xi
        else:
            improve = mu - self.best_y - self.xi

        Z = improve / (sigma + 1e-9)
        return norm.cdf(Z)

    def _upper_confidence_bound(self, mu, sigma):
        """Calculate Upper Confidence Bound acquisition function"""
        if self.minimize:
            return mu - self.kappa * sigma
        return mu + self.kappa * sigma

    def acquisition(self):
        """Calculate next best sample location using chosen acquisition
        function"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.ac_type == 'ei':
            acq_val = self._expected_improvement(mu, sigma)
        elif self.ac_type == 'poi':
            acq_val = self._probability_improvement(mu, sigma)
        elif self.ac_type == 'ucb':
            acq_val = self._upper_confidence_bound(mu, sigma)
        else:
            raise ValueError("Invalid acquisition function type")

        acq_val[sigma < 1e-9] = 0

        next_idx = np.argmax(acq_val)
        X_next = self.X_s[next_idx]

        return X_next, acq_val

    def _save_checkpoint(self, X, Y, acquisition_value):
        """Save checkpoint with hyperparameter values in filename"""
        filename = (f'model_X{X[0]:.6f}_Y{Y[0]:.6f}_'
                   f'acq{acquisition_value:.6f}.npz')
        filepath = os.path.join('checkpoints', filename)
        np.savez(filepath,
                 X=self.gp.X,
                 Y=self.gp.Y,
                 l=self.gp.l,
                 sigma_f=self.gp.sigma_f,
                 K=self.gp.K)
        return filepath

    def optimize(self, iterations=100, tolerance=1e-10):
        """Optimize the black-box function"""
        self.optimization_history = []

        for _ in range(iterations):
            self.iteration += 1
            X_next, acq_val = self.acquisition()

            if any(np.abs(X_next - x_existing) <= tolerance
                  for x_existing in self.gp.X):
                break

            Y_next = self.f(X_next)
            if not isinstance(Y_next, np.ndarray):
                Y_next = np.array(Y_next)

            checkpoint_path = self._save_checkpoint(X_next, Y_next,
                                                 np.max(acq_val))

            self.optimization_history.append({
                'iteration': self.iteration,
                'X': X_next,
                'Y': Y_next,
                'acquisition_value': np.max(acq_val),
                'checkpoint_path': checkpoint_path
            })

            self.gp.update(X_next.reshape(-1, 1), Y_next.reshape(-1, 1))
            self.best_y = None

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt].flatten()
        Y_opt = self.gp.Y[idx_opt].flatten()

        with open('bayes_opt.txt', 'w') as f:
            f.write("Bayesian Optimization Report\n")
            f.write("===========================\n\n")

            f.write("Configuration:\n")
            f.write(f"Acquisition Function: {self.ac_type}\n")
            f.write("Optimization Type: ")
            f.write("Minimization\n" if self.minimize else "Maximization\n")
            f.write(f"Search Space Bounds: {self.bounds}\n")
            f.write(f"Number of Samples: {len(self.X_s)}\n\n")

            f.write("Results:\n")
            f.write(f"Optimal X: {X_opt}\n")
            f.write(f"Optimal Y: {Y_opt}\n")
            f.write(f"Total Iterations: {self.iteration}\n\n")

            f.write("Optimization History:\n")
            f.write("-----------------------\n")
            for entry in self.optimization_history:
                f.write(f"\nIteration {entry['iteration']}:\n")
                f.write(f"X = {entry['X'][0]:.6f}\n")
                f.write(f"Y = {entry['Y'][0]:.6f}\n")
                val = entry['acquisition_value']
                f.write(f"Acquisition Value = {val:.6f}\n")
                f.write(f"Checkpoint: {entry['checkpoint_path']}\n")

        return X_opt, Y_opt
