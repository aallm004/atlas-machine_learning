#!/usr/bin/env python3
"""module for fastext"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Function that creates, builds and trains a genism fastText model
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: min number of occurrences of a word for use in training
        window: max distance between current and predicted word within sentence
        negative: size of negative sampling
        cbow: a boolean to determine the training type;
            True is for CBOW
            False is for Skip-gram
        epoch: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

    Returns: the trained model"""

    # fastText model with specific parameters
    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        seed=seed,
        workers=workers,
        epochs=epochs,
        sg=not cbow
    )

    # Build the vocab from sentences
    model.build_vocab(sentences)

    # Train model on the input sentences
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model
