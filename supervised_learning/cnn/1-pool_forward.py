#!/usr/bin/env python3
"""documentation"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propogation over a pooling layer of a neural network
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernal_shape is a tuple of (kh, kw) containing the size of the kernel for the pooling
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode is a string containing either max or avg, indicating whether to perform maximum or average pooling, respectively

        Returns: the output of the pooling layer
        """
    
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )

    return output
