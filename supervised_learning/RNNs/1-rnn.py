#!/usr/bin/env python3
"""module for rnn"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """function that performs forward propogation for a simple RNN
    
    rnn_cell is an instance of RNNCell that will be used for the forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data

    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """

    # Time steps, batch size, input dimensionality
    t, m, i = X.shape
    # batch size and hidden state dimensionality
    m, h = h_0.shape

    # Gets output dimensionality from rnn_cell's Wy shape
    o = rnn_cell.Wy.shape[1]

    # arrays for space for hidden states and outputs
    # H includes h_0, so it has t+1 steps
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Sets the beginning hidden state
    H[0] = h_0

    # Loop for forward prob for each time step(t)
    for step in range(t):
        # See what input data is for the time step at hand
        x_t = X[step]

        # Retrieve prev hidden state
        h_prev = H[step]

        # Forward prop for current time step
        h_next, y = rnn_cell.forward(h_prev, x_t)

        # Store new hidden state and output
        H[step + 1] = h_next
        Y[step] = Y

    return H, Y
