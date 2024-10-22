#!/usr/bin/env python3
"""documentation"""
import numpy as mp
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images
        IMAGES is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        KERNEL is a numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh is the height of the kernel
            kw is the width of the kernel

        Returns: a numpy.ndarray contining the convolved images
        """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = mp.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = mp.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
