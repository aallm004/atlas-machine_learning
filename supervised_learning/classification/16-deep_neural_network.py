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
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)

        layers.insert(0, nx)
        
        self.cache = {}
        self.weights = {}
        
        for l in range(1, self.L + 1):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]

            self.weights['W' + str(l)] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)

            self.weights['b' + str(l)] = np.zeros((layer_size, 1))
