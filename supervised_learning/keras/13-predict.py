#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def predict(network, data, verbose=False):
    """ Function that makes a prediction using a neural network
    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be printed during the prediction process
    Returns: the prediction for the data"""

    return network.prediction(data, verbose=verbose)
