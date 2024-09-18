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

        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            layer_size = layers[i - 1]
            prev_layer_size = nx if i == 1 else layers[i - 2]

            self.weights['W' + str(i)] = (
                np.random.randn(layer_size, prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )

            self.weights['b' + str(i)] = np.zeros((layer_size, 1))
