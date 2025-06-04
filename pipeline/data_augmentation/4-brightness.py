#!/usr/bin/env python3
"""Module to crop image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Function that randomly changes the brightness of an image
        image: 3D tf.Tensor containing the image to change
        max_delta is the maximum amount the image should be brightened
        (or darkened)
        Returns: altered image"""
    return tf.image.random_brightness(image, max_delta, seed=None)
