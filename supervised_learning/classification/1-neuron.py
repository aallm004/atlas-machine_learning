#!/usr/bin/env python3
"""module documenatation is easy to forget sometimes"""
import numpy as np


class Neuron:
    """A class that defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """class constructor for Neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """getter for weights vector"""
        return self.__W
    
    @property
    def b(self):
        """getter for the bias"""
        return self.__b

    @property
    def A(self):
        """getter for the activated output"""
        return self.__A
