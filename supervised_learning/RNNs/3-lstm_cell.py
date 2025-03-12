#!/usr/bin/env python3
"""module for LSTM Cell"""
import numpy as np


class LSTMCell:
    """Representation of an LSTM unit"""
    def __init__(self, i, h, o):
        """Class constructor
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output

        Creates the public instance attributes that represent
        the weights and biases of the cell
        """
        # WEIGHTS UPDATES
        # Drop gate weights
        self.Wf = np.random.normal(size=(i + h, h))

        # Update gw
        self.Wu = np.random.normal(size=(i + h, h))

        # Intermediate cell weights
        self.Wc = np.random.normal(size=(i + h, h))

        # Output gw
        self.Wo = np.random.normal(size=(i + h, h))

        # Output weights
        self.Wy = np.random.normal(size=(h, o))

        # BIAS UPDATES
        # Drop gate bias
        self.bf = np.zeros((1, h))

        # Update gb
        self.bu = np.zeros((1, h))

        # Intermed cell state bias
        self.bc = np.zeros((1, h))

        # Output gate bias
        self.bo = np.zeros((1, h))

        # Output bias
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward prop for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for
        the cell
            m: batch size for the data
            h_prev: a numpy.ndarray of shape (m, h) containing the previous
            hidden state
            c_prev: a numpy.ndarray of shape (m, h) containing the previous
            cell state
        Returns: h_next, c_next, y
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """

        # Concatenate h_prev and x_t for input to gates
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = sigmoid(np.dot(concat, self.Wf) + self.bf)

        # Gate update
        u = sigmoid(np.dot(concat, self.Wu) + self.bu)

        # Intermed cell state
        c_cand = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # Gate output
        o = sigmoid(np.dot(concat, self.Wo) + self.bo)

        # Next cell state
        c_next = f * c_prev + u * c_cand

        # Next hidden
        h_next = o * np.tanh(c_next)

        # Output
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax activation function"""
    # For numerical stability, subtract the maximum value
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
