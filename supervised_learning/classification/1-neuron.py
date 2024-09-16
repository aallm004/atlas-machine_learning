#!/usr/bin/env python3
"""Module for Class Neuron"""
import numpy as np


class Neuron:
    """A class that defines a single neuron performing binary classification"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
