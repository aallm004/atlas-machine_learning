#!/usr/bin/env python3
"""documentation"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model's configuration in JSON format
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """Function that loads a model with a specific configuration
    filename is the path of the file containing the model's configuration
    in JSON format
    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
