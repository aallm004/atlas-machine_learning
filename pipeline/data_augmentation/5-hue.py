#!/usr/bin/env python3
"""Module to crop image"""
import tensorflow as tf
import numpy as np


def change_hue(image, delta):
    """Function that changes the hue of an image
        image is a 3D tf.Tensor containing the image to change
        delta is the amount the hue should change
        Returns: the altered image"""
    return tf.image.adjust_hue(image, delta)
