#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def create_placeholders(nx, classes):
    """Returns placeholders for the input data and labels."""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
