#!/usr/bin/env python3
"""Dataset Module for Neural Machine Translation

This module defines a Dataset class that handles the loading, preprocessing and
tokenization of data for Portuguese to English machine translation."""
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
        
        # Load Validation dataset with the same parameters
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
