#!/usr/bin/env python3
"""Main script for testing Bayesian optimization"""

import numpy as np
import matplotlib.pyplot as plt
BayesianOptimization = __import__('6-bayes_opt').BayesianOptimization


def f(x):
    """Black-box function to optimize"""
    return np.sin(5*x) + 2*np.sin(-2*x)


def plot_optimization(bo, X_opt, Y_opt):
    """Plot the optimization results"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot the true function and sampled points
    X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
    Y_plot = np.array([f(x[0]) for x in X_plot])

    ax1.plot(X_plot, Y_plot, 'b-', label='True Function')
    ax1.scatter(bo.gp.X, bo.gp.Y, color='red', marker='o',
                label='Sampled Points')
    ax1.scatter([X_opt], [Y_opt], color='green', marker='*',
                s=200, label='Optimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Optimization Results')
    ax1.legend()
    ax1.grid(True)

    # Plot the convergence
    iterations = range(1, len(bo.optimization_history) + 1)
    values = [entry['Y'][0] for entry in bo.optimization_history]
    ax2.plot(iterations, values, 'r.-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence Plot')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_plot.png')
    plt.close()


if __name__ == "__main__":
    # Generate initial data
    X_init = np.array([[-0.7], [0.5], [2]])
    Y_init = np.array([[f(x[0])] for x in X_init])

    # Domain space
    bounds = (-3, 3)
    n_samples = 20

    # Run optimization
    bo = BayesianOptimization(f, X_init, Y_init, bounds, n_samples)
    X_opt, Y_opt = bo.optimize(iterations=10)

    # Create visualization
    plot_optimization(bo, X_opt, Y_opt)

    print('Optimization Results:')
    print('-' * 20)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print(f'\nOptimization completed in {len(bo.optimization_history)} iterations')
    print('\nCheckpoint files saved in ./checkpoints/')
    print('Optimization plot saved as optimization_plot.png')

    # Display contents of bayes_opt.txt
    try:
        with open('bayes_opt.txt', 'r') as f:
            print('\nContents of bayes_opt.txt:')
            print('-' * 20)
            print(f.read())
    except FileNotFoundError:
        print('\nError: bayes_opt.txt was not created')
