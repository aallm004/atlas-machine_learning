#!/usr/bin/env python3
"""module for bag of words"""
import numpy as np
import tensorflow.keras as keras


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix:
        sentences: a list of sentences to analyze
        vocab: a list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used

        Returns: embeddings, features
            embeddings is a numpy.ndarray of shape(s, f) containing the
            embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features is a list of the features used for embeddings
    """
    sentence_tokenized = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        # Create an empty set because set removes duplicates
        unique_words = set()
        # Add words from sentences to set
        for tokens in sentence_tokenized:
            for word in tokens:
                unique_words.add(word)
        # Convert set to list to create ordered vocabulary
        vocab = sorted(list(unique_words))

    # For each sentence, count number of time words occur
    sentence_word_counts = []
    for tokens in sentence_tokenized:
        # Create dict for sentence
        word_count = {}
        # Count occurrences of every word
        for word in tokens:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        sentence_word_counts.append(word_count)

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, word_count in enumerate(sentence_word_counts):
        for word, count in word_count.items():
            if word in vocab:
                # Locating index of this word in the vocab
                j = vocab.index(word)
                # Set count in matrix
                embeddings[i, j] = count

    vocab = np.array(vocab)

    return embeddings, vocab


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
