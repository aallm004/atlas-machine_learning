#!/usr/bin/env python3
"""Module to flip image"""
import tensorflow as tf


def flip_image(image):
    """function that flips an image horizontally
        image: 3D tf.Tensor containing the image to flip
        Returns: the flipped image"""
    return tf.image.flip_left_right(image)
