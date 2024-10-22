#!/usr/bin/env python3
"""documentation"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images using multiple kernels
        IMAGES is a numpy.ndarray with shape (m, h, w, c) containing
        multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        KERNELS is a numpy.ndarray with shape (kh, kw, c, nc) containing the
        kernels for the convolution
            kh is the height of a kernel
            kw is the width of a kernel
            c is the number of channels in the image
            nc is the number of kernels
        PADDING is either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple,
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            The image should be padded with 0's
        STRIDE is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images
        """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    assert c == kc, "Image channels and kernel channels must match"

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2

    elif padding == 'valid':
        ph, pw = 0, 0

    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    * kernels[..., k],
                    axis=(1, 2, 3)
                )

    return output
