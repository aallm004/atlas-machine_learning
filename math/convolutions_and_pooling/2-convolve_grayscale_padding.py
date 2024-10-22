#!/usr/bin/env python3
"""documentation"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function taht performs a convolution on grayscale images to custom
    padding
    IMAGES is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    KERNEL is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    PADDING is tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the images should be padded with 0's

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
