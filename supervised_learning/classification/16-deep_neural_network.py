#!/usr/bin/env python3
import numpy as np
"""Module for DeepNeuralNetwork"""


class DeepNeuralNetwork:
    """class for DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[l - 1], layers[l])
            self.weights["W" + str(l)] = he
            self.weights["b" + str(l)] = np.zeros((layers[l], 1))
