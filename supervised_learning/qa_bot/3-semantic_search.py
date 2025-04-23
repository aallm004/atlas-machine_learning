import os
import sys
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

def semantic_search(corpus_path, sentence):
    """Find and return the most semantically similar document to the sentence."""

    try:
        # List all the files in the directory
        # os.listdir returns a list of all entries in the dir
        # Filter to keep only actual files (not subdirs)
        entries = [f for f in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, f))]
    except Exception as e:
        # Exit program if the dir cannot be accessed due to permissions etc
        sys.exit(f"Cannot read directory: {e}")

    # empty list to store document contents
    texts = []
    # Process each file in the dir
    for entry in entries:
        # Create full path by joining dir path and filename
        full_path = os.path.join(corpus_path, entry)
        try:
            # FIrst try reading the file with UTF-8 encoding
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except UnicodeDecodeError:
            # If UTF-8 fails, try with Latin-1 encoding which can decode any
            # byte value, so is fallback and will hopefully work
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    texts.append(f.read())
            except Exception as e:
                # Skip file if it can't be read with Latin-1 encoding
                continue
        except Exception as e:
            # Skip file on any other exception
            continue
    # Return None if no valid docs
    if not texts:
        return None

    # Load the Universal Sentence Encoder (USE)
    try:
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    except Exception as e:
        sys.exit(f"Error loading USE model: {e}")

    # Encode sentence and documents
    # Takes list of strings and returns their embeddings
    embeddings = encoder([sentence] + texts).numpy()
    # Split the embeddings into query and docs
    # First embedding is the query sent
    sentence_embedding = embeddings[0]
    # Remaining are the docs
    document_embeddings = embeddings[1:]

    # Calculate cosine similarities
    similarities = np.dot(document_embeddings, sentence_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(sentence_embedding) + 1e-10
    )

    # Find the index of the most similar document
    best_match_idx = np.argmax(similarities)

    return texts[best_match_idx]
