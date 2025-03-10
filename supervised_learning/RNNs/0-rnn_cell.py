#!/usr/bin/env python3
"""module for RNN cell"""
import numpy as np


class RNNCell:
    """representation of a simple RNN"""
    def __init__(self, i, h, o):
        """class constructor"""

        # Weights for hidden state and input data
        self.Wh = np.random.normal(size=(i + h, h))
        # Weights for output
        self.Wy = np.random.normal(size=(h, o))
        # Bias for hidden state
        self.bh = np.zeros((1, h))
        # Bias for output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for
        the cell
            m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell"""

        # Concatenate previous hidden state and current input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate Output
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
