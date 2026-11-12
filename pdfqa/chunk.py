"""Split page text into overlapping word windows that stay on a single page."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from pdfqa.extract import PageText

# MiniLM-L6-v2 truncates at 256 tokens. ~180 English words plus a little
# overlap stays comfortably under that limit after tokenization.
DEFAULT_CHUNK_SIZE = 180
DEFAULT_OVERLAP = 36

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page_number: int
    text: str
    word_start: int
    word_end: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=int(data["chunk_id"]),
            page_number=int(data["page_number"]),
            text=str(data["text"]),
            word_start=int(data["word_start"]),
            word_end=int(data["word_end"]),
        )


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk each page independently with a sliding word window.

    Chunks do not cross page boundaries so every citation maps to exactly
    one page. Short pages become a single chunk.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        words = _WORD_RE.findall(page.text)
        if not words:
            continue
        starts = _window_starts(len(words), chunk_size, overlap)
        for start in starts:
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    page_number=page.page_number,
                    text=text,
                    word_start=start,
                    word_end=end,
                )
            )
    return chunks


def _window_starts(n_words: int, chunk_size: int, overlap: int) -> list[int]:
    if n_words <= chunk_size:
        return [0]
    step = chunk_size - overlap
    starts = list(range(0, n_words - chunk_size + 1, step))
    last_start = n_words - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
