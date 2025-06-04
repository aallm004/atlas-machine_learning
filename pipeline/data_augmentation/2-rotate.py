#!/usr/bin/env python3
"""Module to rotate image"""
import tensorflow as tf


def rotate_image(image):
    """Function that rotates an image by 90 degrees counter-clockwise
        image: 3D tf.Tensor containing the image to rotate
        Returns: rotated image"""
    return tf.image.rot90(image)
