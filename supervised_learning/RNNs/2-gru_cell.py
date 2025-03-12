#!/usr/bin/env python3
"""module for GRU Cell"""
import numpy as np


class GRUCell:
    """Representation of a gated recurrent unit"""
    def __init__(self, i, h, o):
        """Class constructor
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output

            Creates the public instance attributes that represent
            the weights and bases of the cell"""

        # Weights initialization
        self.Wz = np.random.normal(size=(i + h, h))
        # Weights for the reset gate initialization
        self.Wr = np.random.normal(size=(i + h, h))
        # Weights for the intermediate hidden state intialization
        self.Wh = np.random.normal(size=(i + h, h))
        # Weights for output initialization
        self.Wy = np.random.normal(size=(h, o))

        # Biases for the update gate
        self.bz = np.zeros((1, h))
        # Biases for the reset gate
        self.br = np.zeros((1, h))
        # Biases for the intermediate hidden state
        self.bh = np.zeros((1, h))
        # Biases for the output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function that performs forward prop for simple RNN"""

        # concat input and prev hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Gate update
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)

        # Gate Reset
        r = sigmoid(np.dot(concat, self.Wr) + self.br)

        # Intermed hidden state
        reset_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_cand = np.tanh(np.dot(reset_concat, self.Wh) + self.bh)

        # next hidden
        h_next = (1 - z) * h_prev + z * h_cand

        # Outputs
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax activation function"""
    # For numerical stability, subtract the maximum value
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
