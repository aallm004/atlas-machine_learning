#!/usr/bin/env python3
"""Module for question answer"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def tf_question_answer(question, reference, top_n=5, threshold=0.2):
    """Function that finds a snippet of text within a reference document to
    answer a question:
        question: a string containing the question to answer
        reference: a string containing the reference document from which to
        find the answer

        Returns: a string containing the answer"""

    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the input query and document
    encoded_inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        max_length=512,
        truncation="only_second",
        return_attention_mask=True,
        return_tensors="tf",
    )

    # Extract tokenized inputs
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # Prepare the model input in the expected format
    model_inputs = [input_ids, attention_mask, token_type_ids]

    # Run the model and get the start and end logits
    result = model(model_inputs)
    start_logits, end_logits = result[0], result[1]

    # Convert logits to probabilities
    start_probs = tf.nn.softmax(start_logits, axis=-1).numpy().squeeze()
    end_probs = tf.nn.softmax(end_logits, axis=-1).numpy().squeeze()

    # Number of tokens
    n_tokens = start_probs.shape[0]

    # Create a mask for the reference (context) tokens, excluding question tokens
    token_type_ids = encoded_inputs["token_type_ids"][0].numpy()
    context_mask = token_type_ids == 1

    # Initialize a score matrix to store the span scores
    span_scores = np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32)

    # Iterate over each token to calculate valid spans
    for i in range(n_tokens):
        if not context_mask[i]:
            continue
        for j in range(i, min(i + 30, n_tokens)):  # Limit span length to 30 tokens
            if context_mask[j]:
                span_scores[i, j] = start_probs[i] + end_probs[j]

    # Check if we have valid spans
    if np.all(np.isneginf(span_scores)):
        return None

    # Find the best start and end indices for the answer
    start_idx, end_idx = np.unravel_index(np.argmax(span_scores), span_scores.shape)

    # Decode the tokens for the best span
    tokens = encoded_inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    # Return the answer or None if it's empty
    if answer:
        return answer
    else:
        return None
