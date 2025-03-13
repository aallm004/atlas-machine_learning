#!/usr/bin/env python3
"""Module for Bidirectional Cell Forward"""
import numpy as np


class BidirectionalCell:
    """Class that represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """class constructor for class Bidirectional Cell
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output

        Creates the public instance attributes that represent
        the weights and biases of the cell"""
        # WEIGHTS
        # Weights forward for the hidden state
        self.Whf = np.random.normal(size=(i + h, h))
        # Weights backward for the hidden state
        self.Whb = np.random.normal(size=(i + h, h))
        # Weights for outputs
        self.Wy = np.random.normal(size=(2 * h, o))

        # BIAS
        # Bias forward for hidden state
        self.bhf = np.zeros((1, h))
        # Bian backward for hidden state
        self.bhb = np.zeros((1, h))
        # Bias for outputs
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in forward direction for one time
        step
            x_t is a numpy.ndarray of shape (m, i) that contains the data
            input for the cell
                m is the batch size for the data
                h_prev is a numpy.ndarray of shape (m, h) containing the
                previous hidden state
        Returns: h_next, the next hidden state
        """
        # concatenates h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculates next hidden state using tahn
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next
