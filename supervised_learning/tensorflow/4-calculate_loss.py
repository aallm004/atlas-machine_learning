#!/usr/bin/env python3
"""tensorflow for beginners"""
import tensorflow.compat.v1 as tf

def calculate_loss(y, y_pred):
    """calculates the loss of prediction"""
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
