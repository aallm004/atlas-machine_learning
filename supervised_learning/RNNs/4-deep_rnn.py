#!/usr/bin/env python3
"""Module for Deep RNN forward prop"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward prop for deep RNN
        rnn_cells is a list of RNNCell instances of length l that will be used
        for the forward propagation
            l is the number of layers
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray of
        shape (l, m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    # Time steps, batch size, input dimension
    t, m, i = X.shape
    # Layers, batch size, hidden dim
    l, m, h = h_0.shape

    # Get ouput dims from last RNN cell
    o = rnn_cells[-1].Wy.shape[1]

    # arrays for space for hidden states and outputs
    # H includes h_0, so it has t+1 steps
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    # Iterate through time steps
    for time_step in range(t):
        # Input at current time step
        x_t = X[time_step]

        # Iterate through the layers
        for layer in range(l):
            # If first layer, use input data
            if layer == 0:
                h_prev = H[time_step, layer]
                h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            # If not first layer, use hidden state from prev
            else:
                h_prev = H[time_step, layer]
                # Use ouput of prev as input
                x_t = H[time_step + 1, layer - 1]
                h_next, y = rnn_cells[layer].forward(h_prev, x_t)

            H[time_step + 1, layer] = h_next

            # If last leyer, store output
            if layer == l - 1:
                Y[time_step] = y

    return H, Y
