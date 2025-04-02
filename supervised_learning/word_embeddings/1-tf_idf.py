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
        unique_words = set()
        for tokens in sentence_tokenized:
            unique_words.update(tokens)
        vocab = list(unique_words)

    document_frequency = {}
    for word in vocab:
        count = 0
        for tokens in sentence_tokenized:
            if word in tokens:
                count += 1
            document_frequency[word] = count

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, tokens in enumerate(sentence_tokenized):
        frequency = {}
        for token in tokens:
            if token in vocab:
                frequency = {}
                for token in tokens:
                    if token in vocab:
                        frequency[token] = frequency.get(token, 0) + 1

        for j, word in enumerate(vocab):
            if word in frequency and document_frequency[word] > 0:
                tf = frequency[word] / len(tokens)

                # Inverse doc freq
                idf = np.log(len(sentences) / document_frequency[word])

                # TF - IDF score
                embeddings[i, j] = tf * idf

        # Normalize the vector with L2 normalization
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm

    return embeddings, np.array(vocab)


def tokenize(sentence):
    # Convert all capital letters to lowercase so they match
    sentence = sentence.lower()
    sentence = sentence.replace("'s", "")

    # Remove punctuation
    for char in "{}()[];:\"'.,!?":
        sentence = sentence.replace(char, "")
    # Split into words
    return sentence.split()
