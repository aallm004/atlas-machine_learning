#!/usr/bin/env python3
"""documentation"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay
    alpha is the original learning rate
    decay_rate determines the rate at which alpha decays
    global_step is the # of passes of gradient descent that
    have elapsed
    decay_step is the # of passes of GD that should occur
    before alpha is decayed further
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
