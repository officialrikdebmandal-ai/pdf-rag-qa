"""Extract text from a PDF, keeping 1-based page numbers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


class EmptyPdfError(ValueError):
    """Raised when a PDF has no extractable text (e.g. a scanned/image-only file)."""


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Return non-empty pages as ``PageText`` objects.

    Uses pypdf's default text extraction. Layout is not reconstructed, so
    tables, multi-column pages, and figures are often jumbled or missing.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _normalize_text(raw)
        if text:
            pages.append(PageText(page_number=i, text=text))
    return pages


def _normalize_text(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks reasonably intact.
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()
