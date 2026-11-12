"""Dense embeddings and cosine similarity."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

# Keep the CLI readable: the model is loaded from the local Hugging Face
# cache after the first run, but the libraries still print progress bars.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL):
    """Load a sentence-transformers model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    model=None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Return an L2-normalized float32 matrix of shape (n, dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if model is None:
        model = load_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity. Vectors are assumed already L2-normalized."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    # Re-normalize defensively in case a cache was written unnormalized.
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    matrix_n = matrix / row_norms
    return matrix_n @ q
