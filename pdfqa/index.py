"""Build or reuse a retrieval index for a PDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pdfqa.cache import (
    DEFAULT_CACHE_DIR,
    cache_key,
    chunks_from_json,
    chunks_to_json,
    load_index,
    save_index,
    sha256_file,
)
from pdfqa.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Chunk, chunk_pages
from pdfqa.embed import DEFAULT_MODEL, embed_texts
from pdfqa.extract import EmptyPdfError, extract_pages


@dataclass
class PdfIndex:
    pdf_path: str
    pdf_sha256: str
    model_name: str
    chunk_size: int
    overlap: int
    chunks: list[Chunk]
    embeddings: np.ndarray
    cache_key: str
    from_cache: bool
    n_pages: int

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


def build_index(
    pdf_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    show_progress: bool = True,
) -> PdfIndex:
    path = Path(pdf_path).resolve()
    digest = sha256_file(path)
    key = cache_key(digest, model_name, chunk_size, overlap)

    if use_cache:
        cached = load_index(cache_dir, key)
        if cached is not None:
            metadata, embeddings = cached
            chunks = chunks_from_json(metadata["chunks"])
            return PdfIndex(
                pdf_path=str(path),
                pdf_sha256=digest,
                model_name=model_name,
                chunk_size=chunk_size,
                overlap=overlap,
                chunks=chunks,
                embeddings=embeddings,
                cache_key=key,
                from_cache=True,
                n_pages=int(metadata.get("n_pages", 0)),
            )

    pages = extract_pages(path)
    if not pages:
        raise EmptyPdfError(
            f"No extractable text in {path}. "
            "This often happens with scanned or image-only PDFs."
        )
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise EmptyPdfError(f"PDF produced no text chunks: {path}")

    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        model_name=model_name,
        show_progress=show_progress,
    )
    metadata = {
        "pdf_path": str(path),
        "pdf_sha256": digest,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "n_pages": len(pages),
        "chunks": chunks_to_json(chunks),
    }
    if use_cache:
        save_index(cache_dir, key, metadata, embeddings)

    return PdfIndex(
        pdf_path=str(path),
        pdf_sha256=digest,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        chunks=chunks,
        embeddings=embeddings,
        cache_key=key,
        from_cache=False,
        n_pages=len(pages),
    )
