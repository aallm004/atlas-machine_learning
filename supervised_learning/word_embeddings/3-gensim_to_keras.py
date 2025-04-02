#!/usr/bin/env python3
"""Function that converts a gensim word2vec model to a keras Embedding layer"""
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """model is a trained gensim word2vec models
    Returns: the trainable keras Embedding"""

    # Get vocab size and vector dimensions from model
    vocab = model.wv.vectors.shape[0]
    vector = model.wv.vectors.shape[1]

    # Extract word vectors from model
    weights = model.wv.vectors

    # Trainable Keras Embedding layer with word2vec weights
    embedding_layer = Embedding(
        input_dim=vocab,
        output_dim=vector,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
