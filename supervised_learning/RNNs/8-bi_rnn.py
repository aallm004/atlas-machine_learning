#!/urs/bin/env python3
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
    h_forward = np.zeros((t + 1, m, h))
    h_backward = np.zeros((t + 1, m, h))

    # Set hidden states
    h_forward[0] = h_0
    h_backward[t] = h_t

    # Forward prop
    for time_step in range(t):
        h_forward[time_step + 1] = bi_cell.forward(
            h_forward[time_step],
            X[time_step]
        )

    # Back prop
    for time_step in range(t, -1, -1, -1):
        h_backward[time_step] = bi_cell.backward(
            h_backward[time_step + 1],
            X[time_step]
        )

    # Concat hidden states, initial state inclusive
    H = np.concatenate((h_forward[1:], h_backward[:-1]), axis=2)

    # Output calculation using hidden states
    Y = bi_cell.output(H)

    return H, Y
