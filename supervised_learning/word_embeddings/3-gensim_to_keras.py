#!/usr/bin/env python3
"""Function that converts a gensim word2vec model to a keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """model is a trained gensim word2vec models
    Returns: the trainable keras Embedding"""
    # .wv can have:
    # vectors: numpy array of shape (vacab size and vector size)
        # with the word embeddings
    # key_to_index: dict that maps words(as strings) to their correct indices
        # in vector array
    # index_to_key: list that maps indices back to words, so you can see which
        # word corresponds to which vector
    
    # Get vocab size and vector dimensions from model
    
    # NumPy array that contains all word vectors
    
    ## Each row goes to a word in the vocab
    ## Each column is a dimension of the embedding space
    vocab = model.wv.vectors.shape[0]
    
    ## Shows how many numbers are used to represent each word
    vector = model.wv.vectors.shape[1]

    # Extract word vectors from model
    weights = model.wv.vectors

    # Trainable Keras Embedding layer with word2vec weights
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab,
        output_dim=vector,
        weights=[weights],
        trainable=True
    )

    return embedding
