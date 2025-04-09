#!/usr/bin/env python3
"""module for N-gram BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence
        each reference translation is a list of the words in the
        translation
            sentence: list containing the model proposed sentence
            n: the size of the n-gram to use for evaluation
        Returns: the n-gram BLEU score"""

    # Calculating precision
    precisions = []
    for i in range(1, n+1):

        # Exctract n-grams from the sentence for this level
        sentence_ngram = extract_ngrams(sentence, i)

        # Count number of n-grams in sentence
        candidate_count = len(sentence_ngram)

        # Dictionary to keep track of max counts across all ref
        max_counts = {}

        for reference in references:
            # Create n-grams for this reference
            reference_ngrams = extract_ngrams(reference, i)

            # Count occurrences of each n-gram in this ref
            ref_ngram_counts = {}
            for ngram in reference_ngrams:
                if ngram in ref_ngram_counts:
                    ref_ngram_counts[ngram] += 1
                else:
                    ref_ngram_counts[ngram] = 1

            # Update max counts for all sentences
            # in order to not overcount repeats
            for ngram, count_val in ref_ngram_counts.items():
                if ngram in max_counts:
                    max_counts[ngram] = max(max_counts[ngram], count_val)
                else:
                    max_counts[ngram] = count_val

        # Calculate matches
        matches = 0
        # Copy of max_counts so not to change the original
        clipped_counts = {k: v for k, v in max_counts.items()}

        # Count each n-gram as a match if it has appeared before
        # Take away from the total count so to avoid counting the same ref
        # n-gram multiple times
        for ngram in sentence_ngram:
            if ngram in clipped_counts and clipped_counts[ngram] > 0:
                matches += 1
                clipped_counts[ngram] -= 1

        # Calculate precision for this n-gram level
        precision = matches / candidate_count
        precisions.append(precision)

    # If precision is 0, BLEU score is 0
    if 0 in precisions:
        return 0

    # Calculate geometric mean with equal weights
    weights = np.ones(n) / n
    score = np.exp(np.sum(weights * np.log(precisions)))

    # Calculate brevity penalty
    candidate_len = len(sentence)

    # Find the ref length that's closest to candidate length
    ref_lengths = [len(reference) for reference in references]
    closest_ref_len = min(ref_lengths, key=lambda x: abs(x - candidate_len))

    # Apply brevity penalty if candidate is shorder than the closest ref
    if candidate_len >= closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / candidate_len)

    # Calculate final BLEU score
    bleu_score = brevity_penalty * score

    return bleu_score


def extract_ngrams(sentence, n):
    """Helper funtion to extract n-grams from a sentence"""
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i+n])
        ngrams.append(ngram)
    return ngrams
