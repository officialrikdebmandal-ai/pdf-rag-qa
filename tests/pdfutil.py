"""Minimal PDF writer used by tests. Standard-14 Helvetica, ASCII text only."""

from __future__ import annotations

from pathlib import Path


def write_simple_pdf(path: str | Path, pages: list[str], font_size: int = 12) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    font_id = 3 + 2 * len(pages)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    page_ids: list[int] = []
    next_id = 3
    for page_text in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        stream = _content_stream(page_text, font_size=font_size)
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii")
    )

    body = bytearray(b"%PDF-1.1\n")
    offsets = {0: 0}
    for obj_id in range(1, font_id + 1):
        offsets[obj_id] = len(body)
        body.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        body.extend(objects[obj_id])
        body.extend(b"\nendobj\n")

    xref_pos = len(body)
    count = font_id + 1
    body.extend(f"xref\n0 {count}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, font_id + 1):
        body.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
    body.extend(
        f"trailer << /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(body)
    return path


def _content_stream(text: str, font_size: int) -> bytes:
    lines = _wrap(text, width=90)
    commands = ["BT", f"/F1 {font_size} Tf", f"{font_size + 4} TL", "72 720 Td"]
    for i, line in enumerate(lines):
        if i:
            commands.append("T*")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines[:40]
