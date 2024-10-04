#!/usr/bin/env python3
"""documentation"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm

        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        s is the previous second moment of var

        Returns: the updated variable and new moment
    """
    new_s = beta2 * s + (1 - beta2) * (grad ** 2)
    updated_var = var - alpha * grad / (np.sqrt(new_s) + epsilon)
    return updated_var, new_s
