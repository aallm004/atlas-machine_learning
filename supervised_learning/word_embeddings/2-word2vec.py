#!/usr/bin/env python3
"""module for word2vec"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Function that creates, builds and trains a gensim word2vec model:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: min number of occurrences of a word for use in training
        window: max distance between current and predicted word within a
            sentence
        negative: size of negative sampling
        cbow: boolean to determine the training type;
            True is for CBOW
            False is for Skip-gram
        epochs: number of interations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model
    Returns: the trained model"""

    # Set training algorithm for cbow
    # Because it's CBOW sg = 0, otherwise it would be sg = 1
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   min_count=min_count,
                                   window=window,
                                   negative=negative,
                                   sg=sg,
                                   epochs=epochs,
                                   seed=seed,
                                   workers=workers)

    # Use build_vocab (builds vocab from sentences) by:
    # Counting word frequencies
    # Apply the min_count filter (used within build_vocab)
    # Build the vocab mapping
    model.build_vocab(sentences)

    # Train model
    # corpus_count is automatically updated when build_vocab was used
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model
