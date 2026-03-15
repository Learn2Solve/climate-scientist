"""Embedding utilities for semantic search over knowledge and papers.

Pure utility module - no DB dependency. Takes pre-loaded candidates
from database methods.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("daemon.embeddings")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_BATCH_SIZE = 2048


def embed_text(text: str, api_key: str) -> np.ndarray:
    """Embed a single text string. Returns a 1536-dim float32 array."""
    return embed_texts([text], api_key)[0]


def embed_texts(texts: list[str], api_key: str) -> list[np.ndarray]:
    """Batch-embed up to 2048 texts. Returns list of 1536-dim float32 arrays."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    results: list[np.ndarray] = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        # Truncate very long texts to avoid token limits
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for item in response.data:
            results.append(np.array(item.embedding, dtype=np.float32))

    return results


def serialize_embedding(arr: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes for SQLite BLOB storage."""
    return arr.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize bytes from SQLite BLOB back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(
    query: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of candidates.

    query: (D,) array
    candidates: (N, D) array
    Returns: (N,) array of similarities in [-1, 1]
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(candidates))
    candidate_norms = np.linalg.norm(candidates, axis=1)
    # Avoid division by zero
    candidate_norms = np.where(candidate_norms == 0, 1.0, candidate_norms)
    return (candidates @ query) / (candidate_norms * query_norm)


def search_by_embedding(
    query_text: str,
    candidates: list[tuple[str, bytes]],
    api_key: str,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """End-to-end semantic search.

    candidates: list of (id, embedding_blob) from DB methods.
    Returns: list of (id, similarity_score) sorted by score descending.
    """
    if not candidates:
        return []

    query_vec = embed_text(query_text, api_key)

    ids = [c[0] for c in candidates]
    matrix = np.array(
        [deserialize_embedding(c[1]) for c in candidates],
        dtype=np.float32,
    )

    scores = cosine_similarity(query_vec, matrix)

    # Sort by score descending
    ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:limit]
