#!/usr/bin/env python3
"""module for N-gram BLEU score"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence
        references: a list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence: list containing the model proposed sentence
        n: the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score"""

    # Extract n-grams from the sentence
    sentence_ngram = extract_ngrams(sentence, n)

    # Count number of n-grams that appear in sentence
    candidate_count = len(sentence_ngram)

    # Dictionary to keep tack of max counts across all
    max_counts = {}

    for reference in references:
        # Create n-grams for this reference
        reference_ngrams = extract_ngrams(reference, n)

        # Count occurences of each n-gram in this reference
        ref_ngram_counts = {}
        for ngram in reference_ngrams:
            if ngram in ref_ngram_counts:
                ref_ngram_counts[ngram] += 1
            else:
                ref_ngram_counts[ngram] = 1

        # update max counts for all sentences
        # so not to overcount repeated n-grams
        for ngram, count_val in ref_ngram_counts.items():
            if ngram in max_counts:
                max_counts[ngram] = max(max_counts[ngram], count_val)
            else:
                max_counts[ngram] = count_val

    # Calculate matches
    matches = 0
    # Make copy of max_counts so not to change the original
    clipped_counts = {k: v for k, v in max_counts.items()}

    # Count each n-gram as a match if it appears before
    # Take away from the count to avoid counting the same reference n-gram
    # multiple times
    for ngram in sentence_ngram:
        if ngram in clipped_counts and clipped_counts[ngram] > 0:
            matches += 1
            clipped_counts[ngram] -= 1

    # Calculate precision
    precision = matches / candidate_count

    # Calculate brevity penalty
    candidate_len = len(sentence)

    # Find the reference length that's closest to the candidate length
    ref_lengths = [len(reference) for reference in references]
    closest_ref_len = min(ref_lengths, key=lambda x: abs(x - candidate_len))

    # Apply brevity penalty if candidate is shorter than the closest ref
    if candidate_len > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / candidate_len)

    # Calculate final BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score


def extract_ngrams(sentence, n):
    """Helper funtion to extract n-grams from a sentence"""
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i+n])
        ngrams.append(ngram)
    return ngrams
