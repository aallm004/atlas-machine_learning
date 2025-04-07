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

    # Find the reference with the closest length to the candidate
    closest_ref_idx = np.argmin([abs(len(reference) - length_of_sentence)
                                for reference in references])
    closest_ref_length = len(references[closest_ref_idx])

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

    # Calculate precision
    precision = count_clipped / length_of_sentence

    # Brevity penalty
    bp = np.exp(1 - closest_ref_length / length_of_sentence)

    bleu_score = bp * precision

    return bleu_score
