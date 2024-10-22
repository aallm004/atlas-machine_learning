#!/usr/bin/env python3
"""documentation"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images
        IMAGES is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        KERNEL is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        Returns: a numpy.ndarray containing the convolved images
        """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    output = np.zeros((m, h, w))

    padded = np.pad(images,
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
