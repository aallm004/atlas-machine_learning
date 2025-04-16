#!/usr/bin/env python3
"""class for dataset"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads a preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """creates instance attributes"""

        self.max_len = max_len
        self.batch_size = batch_size

        # Load raw datasets
        raw_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train", as_supervised=True)
        raw_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation", as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            raw_train)
        
        padding_values = (tf.constant(0, dtype=tf.int64),
                          tf.constant(0, dtype=tf.int64))
        
        # Training pipeline is built
        self.data_train = (raw_train
                            .map(self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
                            .filter(self.filter_max_length)
                            .cache()
                            .shuffle(20000)
                            .padded_batch(
                                self.batch_size,
                                padded_shapes=([None], [None]),
                                padding_values=padding_values)
                            .prefetch(tf.data.AUTOTUNE))
        self.data_valid = (raw_valid
                            .map(self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
                            .filter(self.filter_max_length)
                            .padded_batch(
                                self.batch_size,
                                padded_shapes=([None], [None]),
                                padding_values=padding_values
                            ))


    def tokenize_dataset(self, data):
        """Word breakdown for dataset"""

        # Create empty lists for sentences in both English and Portuguese
        en_sentences = []
        pt_sentences = []

        # Extract sentences
        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # tokenizer initializer
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

        # Set vocab size for both en and pt
        vocab_size = 2**13

        # Train tokenizer with vocab_size
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=vocab_size)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        vocab_size = 2**13

        en_text = en.numpy().decode('utf-8')
        pt_text = pt.numpy().decode('utf-8')

        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)
        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)

        en_tokens = [vocab_size] + en_tokens + [vocab_size + 1]
        pt_tokens = [vocab_size] + pt_tokens + [vocab_size + 1]

        en_tokens_array = tf.convert_to_tensor(en_tokens,
                                               dtype=tf.int64)
        pt_tokens_array = tf.convert_to_tensor(pt_tokens,
                                               dtype=tf.int64)

        return en_tokens_array, pt_tokens_array

    def tf_encode(self, pt, en):
        """Instance method that acts as a tensorflow wrapper for the encode
        instance method"""
        en_encoded, pt_encoded = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        # Shapes applied
        en_encoded.set_shape([None])
        pt_encoded.set_shape([None])

        return en_encoded, pt_encoded
    
    def filter_max_length(self, pt_tokens, en_tokens):
        """Filter examples that are too long"""
        correct_length = tf.logical_and(
            tf.size(en_tokens) <= self.max_len,
            tf.size(pt_tokens) <= self.max_len
        )
        return correct_length
