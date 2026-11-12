"""Dense and hybrid (BM25 + dense) retrieval, plus sentence re-ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from pdfqa.chunk import Chunk
from pdfqa.embed import cosine_scores, embed_texts
from pdfqa.index import PdfIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

DEFAULT_TOP_K = 5
DEFAULT_SENTENCES = 4
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int
    source: str  # "dense", "hybrid", or "sentence"


def _normalize_token(token: str) -> str:
    """Very light suffix stripping so BM25 matches suppress/suppresses, etc."""
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(tok) for tok in _TOKEN_RE.findall(text.lower())]


def dense_search(index: PdfIndex, query: str, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
    query_vec = embed_texts([query], model_name=index.model_name)[0]
    scores = cosine_scores(query_vec, index.embeddings)
    return _top_hits(index.chunks, scores, top_k, source="dense")


def hybrid_search(
    index: PdfIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Fuse dense cosine ranks with BM25 ranks using Reciprocal Rank Fusion."""
    n = len(index.chunks)
    if n == 0:
        return []

    query_vec = embed_texts([query], model_name=index.model_name)[0]
    dense_scores = cosine_scores(query_vec, index.embeddings)
    dense_order = list(np.argsort(-dense_scores))

    tokenized = [_tokenize(chunk.text) for chunk in index.chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = np.asarray(bm25.get_scores(_tokenize(query)), dtype=np.float64)
    bm25_order = list(np.argsort(-bm25_scores))

    fused = np.zeros(n, dtype=np.float64)
    for rank, idx in enumerate(dense_order, start=1):
        fused[idx] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_order, start=1):
        fused[idx] += 1.0 / (rrf_k + rank)
    # Break exact RRF ties with the dense cosine score so a 2-chunk
    # disagreement cannot flip the winner by original array order.
    fused = fused + 1e-6 * dense_scores.astype(np.float64)
    return _top_hits(index.chunks, fused, top_k, source="hybrid")


def retrieve(
    index: PdfIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = False,
) -> list[Hit]:
    if hybrid:
        return hybrid_search(index, query, top_k=top_k)
    return dense_search(index, query, top_k=top_k)


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def sentence_hits(
    index: PdfIndex,
    query: str,
    chunk_hits: list[Hit],
    max_sentences: int = DEFAULT_SENTENCES,
) -> list[Hit]:
    """Re-rank sentences from retrieved chunks by cosine similarity to the query."""
    candidates: list[Chunk] = []
    for hit in chunk_hits:
        for sentence in split_sentences(hit.chunk.text):
            if len(sentence.split()) < 4:
                continue
            candidates.append(
                Chunk(
                    chunk_id=hit.chunk.chunk_id,
                    page_number=hit.chunk.page_number,
                    text=sentence,
                    word_start=hit.chunk.word_start,
                    word_end=hit.chunk.word_end,
                )
            )
    if not candidates:
        return []
    query_vec = embed_texts([query], model_name=index.model_name)[0]
    matrix = embed_texts(
        [c.text for c in candidates],
        model_name=index.model_name,
    )
    scores = cosine_scores(query_vec, matrix)
    hits = _top_hits(candidates, scores, max(max_sentences * 3, max_sentences), source="sentence")
    return _dedupe_hits(hits, max_sentences)


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe_hits(hits: list[Hit], limit: int) -> list[Hit]:
    """Drop near-duplicate sentences caused by overlapping chunks."""
    seen: set[str] = set()
    unique: list[Hit] = []
    for hit in hits:
        key = _norm_text(hit.chunk.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            Hit(chunk=hit.chunk, score=hit.score, rank=len(unique) + 1, source=hit.source)
        )
        if len(unique) >= limit:
            break
    return unique


def _top_hits(
    chunks: list[Chunk],
    scores: np.ndarray,
    top_k: int,
    source: str,
) -> list[Hit]:
    if len(chunks) == 0:
        return []
    k = min(top_k, len(chunks))
    order = np.argsort(-np.asarray(scores))[:k]
    return [
        Hit(chunk=chunks[int(i)], score=float(scores[int(i)]), rank=rank, source=source)
        for rank, i in enumerate(order, start=1)
    ]
