#!/usr/bin/env python3
"""documentation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolultional layer of
    a neural network:
        dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev is a numpy.ndarray or shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
            kh is the filter height
            kw is the filter width
        b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        padding is a string that is either same or valid, indicating the type
        of padding used
        stride is a tuple of (sh, sw) containing the strides for the
        convolution
            sh is the stride for the height
            sw is the stride for the width
        Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respecitvely
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)


    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')
    dA_prev_padded = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                            mode='constant')
    
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for c in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw
                    A_slice = A_prev_padded[i, vert_start:vert_end,
                                            horiz_start:horiz_end, :]
                    dA_prev_padded[i, vert_start:vert_end,
                                   horiz_start:horiz_end, :] += \
                        W[:, :, :, c] * dZ[i, j, k, c]
                    dW[:, :, :, c] += A_slice * dZ[i, j, k, c]
    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:-ph or None, pw:-pw or None, :]
    else:
        dA_prev = dA_prev_padded[:, :h_prev, :w_prev, :]

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    
    return dA_prev, dW, db
