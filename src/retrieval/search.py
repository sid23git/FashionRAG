"""
Module for running similarity searches against the built FAISS index.
"""

import os
import json
import numpy as np
from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.reranker import FashionReranker

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def perform_search(
    query_text: str = None,
    query_image_path: str = None,
    index_dir: str = "indexes",
    top_k: int = 5,
    enable_reranking: bool = True
):
    """
    Encodes the query and searches the FAISS index.
    Prints the retrieved top_k results.
    """
    index_path = os.path.join(index_dir, "fashion_catalog.index")
    mapping_path = os.path.join(index_dir, "index_mapping.json")

    # Validate index files existence
    numpy_fallback_path = index_path.replace(".index", ".npy")
    index_exists = (HAS_FAISS and os.path.exists(index_path)) or os.path.exists(numpy_fallback_path)

    if not index_exists or not os.path.exists(mapping_path):
        print(f"Error: Index files or mapping not found. Please run the indexing mode first to build them.")
        print(f"Index expected at: {index_path} or {numpy_fallback_path}")
        print(f"Mapping expected at: {mapping_path}")
        return

    # Load mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    # 1. Encode query
    print("Encoding query...")
    encoder = QueryEncoder()
    if query_image_path:
        query_vector = encoder.encode_image_query(query_image_path)
        print(f"Encoded query image: '{query_image_path}'")
    else:
        query_vector = encoder.encode_text_query(query_text)
        print(f"Encoded query text: '{query_text}'")

    # Reshape for search (1, D)
    query_vector = query_vector.reshape(1, -1).astype("float32")

    # 2. Search FAISS Index or NumPy matrix
    results = []  # List of dicts: {'path': str, 'score': float}
    
    if HAS_FAISS and os.path.exists(index_path):
        print("Searching FAISS index...")
        index = faiss.read_index(index_path)
        # Search index
        scores, indices = index.search(query_vector, top_k)
        
        # Parse scores
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            path = mapping.get(str(idx)) or mapping.get(idx)
            if path:
                results.append({"path": path, "score": float(score)})
    else:
        print("Using NumPy fallback search...")
        embeddings = np.load(numpy_fallback_path)
        # Compute Cosine similarity (Inner product of normalized vectors)
        scores = np.dot(embeddings, query_vector.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        for idx in top_k_indices:
            path = mapping.get(str(idx)) or mapping.get(idx)
            if path:
                results.append({"path": path, "score": float(scores[idx])})

    # 3. Optional Reranking
    if enable_reranking and results:
        print("Applying reranking pipeline...")
        reranker = FashionReranker()
        results = reranker.rerank(query_text or os.path.basename(query_image_path or ""), results)

    # 4. Display Results
    print("\n--- Search Results ---")
    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f} | Path: {res['path']}")
    print("----------------------\n")
    return results
