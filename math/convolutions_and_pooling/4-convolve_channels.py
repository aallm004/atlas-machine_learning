#!/usr/bin/env python3
"""documentation"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Funtion that performs a conolution on images with channels
        IMAGES is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        KERNEL is a numpy.ndarray with shape (kh, kw, c) containing the kernel
        for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        PADDING is either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            the image should be padded with 0's
        STRIDE is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        Only allowed to use two for loops; any other loops of any kind
        are not allowed

        Returns: a numpy.ndarray containing the convolved images
        """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    assert c == kc, "Image channels and kernel channels must match"

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (kh % 2 == 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (kw % 2 == 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            roi = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j] = np.sum(roi * kernel, axis=(1, 2, 3))

    return output
