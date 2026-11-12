"""On-disk cache for chunk metadata and embedding matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pdfqa.chunk import Chunk

DEFAULT_CACHE_DIR = Path(".pdfqa_cache")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(
    pdf_sha256: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> str:
    payload = f"{pdf_sha256}|{model_name}|{chunk_size}|{overlap}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_paths(cache_dir: str | Path, key: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{key}.npz", root / f"{key}.json"


def save_index(
    cache_dir: str | Path,
    key: str,
    metadata: dict,
    embeddings: np.ndarray,
) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = cache_paths(root, key)
    np.savez_compressed(npz_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return json_path


def load_index(cache_dir: str | Path, key: str) -> tuple[dict, np.ndarray] | None:
    npz_path, json_path = cache_paths(cache_dir, key)
    if not npz_path.is_file() or not json_path.is_file():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    embeddings = np.load(npz_path)["embeddings"].astype(np.float32)
    return metadata, embeddings


def chunks_to_json(chunks: list[Chunk]) -> list[dict]:
    return [chunk.to_dict() for chunk in chunks]


def chunks_from_json(rows: list[dict]) -> list[Chunk]:
    return [Chunk.from_dict(row) for row in rows]
