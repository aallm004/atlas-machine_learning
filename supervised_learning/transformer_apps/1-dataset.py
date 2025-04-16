#!/usr/env/bin python3
"""class for dataset"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads a preps a dataset for machine translation"""

    def __init__(self):
        """creates instance attributes"""
        
        # Load training and valildation datasets
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en", split="train", as_supervised=True )
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en", split="validation", as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)


    def tokenize_dataset(self, data):
        """Word breakdown for dataset"""

        # Create empty lists for sentences in both English and Portuguese
        en_sentences = []
        pt_sentences = []

        # Extract sentences
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create iterators
        en_iterator = iter(en_sentences)
        pt_iterator = iter(pt_sentences)
        
        # tokenizer initializer
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Set vocab size for both en and pt
        vocab_size = 2**13

        # Train tokenizer with vocab_size
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_iterator, vocab_size=vocab_size)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_iterator, vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en


    def encode(self, pt, en):
        """encodes a translation into tokens"""
        vocab_size = 2**13
        
        en_text = en.numpy().decode('utf-8')
        pt_text = pt.numpy().decode('utf-8')

        en_tokens = self.tokenizer_en.encode(en_text)
        pt_tokens = self.tokenizer_pt.encode(pt_text)

        en_tokens_with_added = [vocab_size] + en_tokens + [vocab_size + 1]
        pt_tokens_with_added = [vocab_size] + pt_tokens + [vocab_size + 1]

        en_tokens_array = np.array(en_tokens_with_added)
        pt_tokens_array = np.array(pt_tokens_with_added)

        return pt_tokens_array, en_tokens_array
