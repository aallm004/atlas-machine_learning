#!/usr/bin/env python3
"""Module for question answer"""
import os
import sys
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
import numpy as np
semantic_search = __import__("3-semantic_search").semantic_search
tf_question_answer = __import__("0-qa").tf_question_answer

EXIT_KEYWORDS = {"exit", "quit", "goodbye", "bye"}

def question_answer(corpus_path):
    """Function that answers questions from multiple reference texts
        corpus_path: the path to the corpus of reference documents"""
    # Check if the corpus path is valid
    if not os.path.exists(corpus_path) or not os.path.isdir(corpus_path):
        print(f"[ERROR] Invalid directory: {corpus_path}")
        sys.exit(1)

    print("Welcome! Ask anything about the Atlas School! Type 'exit' to quit.\n")

    while True:
        try:
            # Get user input and remove whitespace
            prompt = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA: Goodbye")
            break

        # Check for exit command
        if prompt.lower() in EXIT_KEYWORDS:
            print("A: Goodbye")
            break

        # Skip empty inputs
        if not prompt:
            continue

        # Find the most relevant document using semantic search
        document = semantic_search(corpus_path, prompt)
        if not document:
            print("A: Sorry, I don't have anything to reference.")
            continue

        # Extract answer from document using BERT
        reply = tf_question_answer(prompt, document)
        if not reply:
            reply = "Sorry, I do not know."

        print(f"A: {reply}")
