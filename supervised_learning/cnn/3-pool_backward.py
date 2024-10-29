#!/usr/bin/env python3
"""documentation"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagatoin over a pooling layer of a neural
    network:
        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
        the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
        kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the hieght
            sw is the stride for the width
        mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respecitvely

        Returns: the partial derivatives with respect to the previous layer
        (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for c in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += \
                            mask * dA[i, j, k, c]

                    elif mode == 'avg':
                        dA_curr = dA[i, j, k, c]
                        dA_avg = dA_curr / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += dA_avg

    return dA_prev
