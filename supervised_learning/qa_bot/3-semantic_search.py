import os, numpy as np
from typing import Optional, List
import tensorflow_hub as hub

def semantic_search(corpus_path: str, sentence: str) -> Optional[str]:
    # Grab the Universal Sentence Encoder - smaller model is faster
    encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Grab all documents
    all_docs: List[str] = []
    
    # Walk through directory 
    for filename in os.listdir(corpus_path):
        full_path = os.path.join(corpus_path, filename)
        # Skip directories
        if not os.path.isfile(full_path):
            continue
            
        # Try to read the file
        for encoding in ['utf-8', 'latin-1']:
            try:
                with open(full_path, 'r', encoding=encoding) as f:
                    all_docs.append(f.read())
                # If we got here, reading worked
                break
            except:
                # Just keep trying encodings
                pass
    
    # Bail if we found nothing
    if len(all_docs) == 0:
        return None
    
    # Get embeddings for everything at once 
    # (more efficient than one-by-one)
    embedded = encoder([sentence] + all_docs)
    
    # Pull apart the embeddings
    query_vec = embedded[0]
    doc_vecs = embedded[1:]
    
    # Do vector math for similarity scores
    # Formula: cos(θ) = (A·B)/(||A||·||B||)
    scores = []
    for doc_vec in doc_vecs:
        dot_product = np.dot(query_vec, doc_vec)
        magnitude_product = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        # Avoid division by zero
        if magnitude_product == 0:
            scores.append(0)
        else:
            scores.append(dot_product / magnitude_product)
    
    # Find the winner
    best_idx = np.argmax(scores)
    return all_docs[best_idx]
