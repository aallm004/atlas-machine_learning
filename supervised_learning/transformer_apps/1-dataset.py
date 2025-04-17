#!/usr/bin/env python3
"""Dataset Module for Neural Machine Translation

This module defines a Dataset class that handles loading, preprocessing,
tokenization, and encoding of data for Portuguese to English machine
translation."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for machine translation (Portuguese to English)
    
    Handles loading training and validation splits, creating and training
    custon tokenizers based on BERT models, and storing tokenizers for later
    use in the translation pipeline"""

    def __init__(self):
        """Initializes the Dataset instance by loading data and creating
        tokenizers
        
        Loads the Portuguese to English dataset
        Calls tokenize_dataset to create and train tokenizers for both
        languages"""

        # Load training and valildation datasets
        # as_supervised=True returns the dataset as (input, target) pairs
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train", as_supervised=True)
        
        # Load Validation dataset with the same parameterss
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation", as_supervised=True)

        # Initialize tokenizers for both languages by processing the training
        # data. This creates and trains custom tokenizers based on pre-trained
        # Bert models
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates and trains tokenizers for both Portuguese and English
            Extracts sentences from the dataset
            Initializes BERT tokenizers for both languages
            Trains custom tokenizers with a vocabulary size of 8,192 tokens"""

        # Create empty lists for extracted sentences in both English and
        # Portuguese
        en_sentences = []
        pt_sentences = []

        # Extract and decode sentences from TensorFlow tensors
        # Each item in data is a (pt, en) pair of tensors containing UTF-8
        # encoded text
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create iterators from sentence lists for tokenizer training
        en_iterator = iter(en_sentences)
        pt_iterator = iter(pt_sentences)

        # Initialize base tokenizers from pre-trained BERT models
        # For Portuguese: Use NeuralMind's Portuguese-specific BERT
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        # For English; Use standard uncased BERT model
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

        # Set vocab size for both tokenizers
        # 2^13 = 8,192 tokens for each language
        vocab_size = 2**13

        # Train new tokenizer based on the initialized models
        # Customizes the tokenization to this specific dataset
        # train_new_from_iterator adapts the tokenizer to the specific dataset
        # vocabulary
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator, vocab_size=vocab_size)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator, vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation pair into token IDs with special tokens
            Decodes the tensor inputs to text strings
            Tokenizes both strings using the trained tokenizers
            Adds special tokens for start and end of sequence
            Converts token lists to NumPy arrays"""
        # Define vocabulary size (same as tokenize_dataset)
        vocab_size = 2**13

        # Decode the tensor inputs to UTF-8 text strings
        en_text = en.numpy().decode('utf-8')
        pt_text = pt.numpy().decode('utf-8')

        # Encode text strings to token IDs using the trained tokenizers
        # add_special_tokens=False to avoid BERT's default special tokens
        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)
        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)

        # Add custom special tokens:
        # - vocab_size (8192) as start-of-sequence token
        # - vocab_size + 1 (8192) as end-of-sequence token
        en_tokens = [vocab_size] + en_tokens + [vocab_size + 1]
        pt_tokens = [vocab_size] + pt_tokens + [vocab_size + 1]

        # Convert token lists to NumPy arrays for further processing
        en_tokens_array = np.array(en_tokens)
        pt_tokens_array = np.array(pt_tokens)

        return pt_tokens_array, en_tokens_array
