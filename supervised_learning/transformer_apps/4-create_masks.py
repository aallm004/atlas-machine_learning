#!/usr/bin/env python3
"""Mask Generation Module for Transformer-based Machine Translation

This module provides functionality to create attention masks required for
training and inference in transformer-based sequence-to-sequence models"""
import tensorflow as tf


def create_masks(input, target):
    """Function that creates all masks for training/validation
    
        This function generates three types of masks needed for transformer
        architecture:
            Encoder Mask: Prevents attenton to padding tokens in the input
            sequence
            Combined Mask: Combines lookahead mask and padding mask for the
            target sequence
            Decoder Mask: Prevents attention to padding tokens in cross-attention
            
            Function of mask:
                - Values of 1 in the masks bock attention to corresponding
                positions
                - Values of 0 allow attention to those positions"""
    
    # Create padding massks by identifying padding tokens (value of 0)
    # These are converted to float32 with values:
    # - 1.0 for positions that should be masked (padding tokens)
    # - 0.0 for positions that should be attended to (actual tokens)
    input_padding_mask = tf.cast(tf.equal(input, 0), tf.float32)
    target_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)

    # Encoder mask - prevents attention to padding tokens in the input
    # SHape: [bach_size, 1, 1, input_seq_len]
    # The added dimensions are for compatibility with the attention mechanism
    encoder_mask = input_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Create lookahead mask for decoder to prevent attending to future tokens
    # This creates a lower triangular matrix with zeros below the diagonal and
    # ones above
    target_len = tf.shape(target)[1]
    # tf.linalg.bad_part(input, -1, 0) creates a lower triangular matrix
    # 1 - that matrix gives us ones above the diagonal (the future positions)
    lookahead_mask = 1 - tf.linalg.band_part(tf.ones((target_len, target_len)), -1, 0)
    # Add bath and head dimensions [1, 1, target_len, target_len]
    lookahead_mask = lookahead_mask[tf.newaxis, tf.newaxis, :, :]
    
    # Combine lookahead with target padding mask
    # Reshape target padding mask to [batch_size, 1, 1, target_seq_len]
    target_padding_mask_x = target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Take element-wise maximum to combine the masks
    # A position is masked (1) if it's masked in either the lookahead mask
    # or padding mask 
    combined_mask = tf.maximum(lookahead_mask, target_padding_mask_x)
    # Decoder mask for encoder-decoder cross-attention
    # This prevents attending to padding tokens in the input sequence
    # Shape: [batch_size, 1, 1, input_seq_len]
    decoder_mask = input_padding_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
