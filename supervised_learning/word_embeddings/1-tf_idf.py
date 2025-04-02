#!/usr/bin/env python3
"""module for tf_idf"""
import numpy as np
import tensorflow.keras as keras


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding
        sentences: a built list of sentences to analyze
        vocab: a list of the vocab words to use for the analysis
            If None, all words within sentences should be used
        Returns: embeddings, features
            embeddings is a numpy.ndarray of shape (s, f) containing the
            embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features is a list of the features used for embeddings"""

    sentence_tokenized = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        # Create an empty set because set removes duplicates
        vocab = sorted(set(word for tokens in sentence_tokenized for word in
                           tokens))

        vocab = np.array(vocab)

    # Document frequency for each word
    document_frequency = {word: sum(1 for tokens in sentence_tokenized if word
                                    in tokens) for word in vocab}

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, tokens in enumerate(sentence_tokenized):
        # Skip empty sentences
        if not tokens:
            continue

        frequency = {token: tokens.count(token) / len(tokens) for token in
                     tokens if token in vocab}

        for j, word in enumerate(vocab):
            if word in frequency:
                tf = frequency[word]
                idf = np.log((1 + len(sentences)) /
                             (1 + document_frequency[word])) + 1
                embeddings[i, j] = tf * idf

    # Normalize using L2 normalization
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norm, where=norm != 0)

    return embeddings, np.array(vocab)


def tokenize(sentence):
    """tokenization of sentence in to words"""

    # Convert all capital letters to lowercase so they match
    sentence = sentence.lower()
    sentence = sentence.replace("'s", "")

    # Remove punctuation
    for char in "{}()[];:\"'.,!?":
        sentence = sentence.replace(char, "")
    # Split into words
    return sentence.split()
