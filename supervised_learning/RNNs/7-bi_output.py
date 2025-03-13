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

        # Calculates next hidden state using tanh
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward direction for one time
        step.
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
                m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h) containing the next
            hidden state
        Returns: h_prev, the previous hidden state"""
        # Concatenates h_next and x_t
        concat = np.concatenate((h_next, x_t), axis=1)

        # Calculate prev hidden state using tanh
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """Calculates all outputs for the RNN
            H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states
                t is the number of time steps
                m is the batch size for the data
                h is the dimensionality of the hidden states
            Returns: Y, the outputs
        """
        t, m, _ = H.shape

        # Array to store the outputs
        Y = np.zeros((t, m, self.by.shape[1]))

        # Output calculation for each time step
        for time_step in range(t):
            # Use softmax to get output here
            Y[time_step] = self.softmax(np.dot(H[time_step], self.Wy)
                                        + self.by)

        return Y

    def softmax(self, x):
        """Softmax activation function"""
        # For numerical stability, subtract the maximum value
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
