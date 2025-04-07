#!/usr/bin/env python3
"""module for Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence
        references: a list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence: list containing the model proposed sentence
    Returns: the unigram BLEU score"""
    length_of_sentence = len(sentence)

    # Count each word that appears in sentence
    count = {}
    for word in sentence:
        if word not in count:
            max_count = 0
            for reference in references:
                max_count = max(max_count, reference.count(word))
            count[word] = max_count                

    # Calculate clipped count
    count_clipped = sum(min(sentence.count(word), count)
                        for word, count in count.items())

    bleu_score = count_clipped / length_of_sentence

    return bleu_score
