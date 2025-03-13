#!/usr/bin/env python3
"""Module for bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """function that performs forward propagation for bidirectional RNN
        bi_cell is an instance of BidirectinalCell that will be used for the
        forward propagation
            X is the data to be used, given as a numpy.ndarray of
            shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h)
        Returns: H, Y
            H is a numpy.ndarray containing all of the concatenated hidden
            states
            Y is a numpy.ndarray containing all of the outputs
    """
    # time steps, batch size, input dimension
    t, m, i = X.shape

    # hidden state dimension
    _, h = h_0.shape

    # array initialization
    h_forward = np.zeros((t, m, h))
    h_backward = np.zeros((t, m, h))

    # Forward prop
    h_prev = h_0
    for time_step in range(t):
        h_prev = bi_cell.forward(h_prev, X[time_step])
        h_forward[time_step] = h_prev

    # Back prop
    h_next = h_t
    for time_step in range(t-1, -1, -1):
        h_next = bi_cell.backward(h_next, X[time_step])
        h_backward[time_step] = h_next

    # Concat hidden states, initial state inclusive
    H = np.concatenate((h_forward, h_backward), axis=2)

    # Output calculation using hidden states
    Y = bi_cell.output(H)

    return H, Y
