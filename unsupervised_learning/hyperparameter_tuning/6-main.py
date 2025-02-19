#!/usr/bin/env python3
"""Main script for testing Bayesian optimization"""

import numpy as np
import matplotlib.pyplot as plt
BayesianOptimization = __import__('5-bayes_opt').BayesianOptimization


def f(x):
    """Black-box function to optimize"""
    return np.sin(5*x) + 2*np.sin(-2*x)


# Generate initial data
X_init = np.array([[-0.7], [0.5], [2]])
Y_init = np.array([[f(x[0])] for x in X_init])

# Domain space
bounds = (-3, 3)
n_samples = 20

# Run optimization
bo = BayesianOptimization(f, X_init, Y_init, bounds, n_samples)

# Run for multiple iterations
X_opt, Y_opt = bo.optimize(iterations=10)

print('Optimal X:', X_opt)
print('Optimal Y:', Y_opt)

# Open the output file to make sure it was created
try:
    with open('bayes_opt.txt', 'r') as f:
        print('\nContents of bayes_opt.txt:')
        print(f.read())
except FileNotFoundError:
    print('bayes_opt.txt was not created')
