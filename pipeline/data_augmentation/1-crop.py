#!/usr/bin/env python3
"""Module to crop image"""
import tensorflow as tf
import numpy as np


def crop_image(image, size):
    """Function that performs a random crop of an image
        image is a 3D tf.Tensor containing the image to crop
        size is a tuple containing the size of the crop
        Returns: cropped image"""
    return tf.image.random_crop(image, size=size)
