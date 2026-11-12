"""Turn retrieved passages into an extractive answer, or an optional LLM answer."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from pdfqa.index import PdfIndex
from pdfqa.retrieve import (
    DEFAULT_SENTENCES,
    DEFAULT_TOP_K,
    Hit,
    retrieve,
    sentence_hits,
)


@dataclass
class Answer:
    question: str
    mode: str
    text: str
    sentences: list[Hit]
    chunks: list[Hit]


def extractive_answer(
    index: PdfIndex,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    max_sentences: int = DEFAULT_SENTENCES,
    hybrid: bool = False,
) -> Answer:
    chunks = retrieve(index, question, top_k=top_k, hybrid=hybrid)
    sentences = sentence_hits(index, question, chunks, max_sentences=max_sentences)
    if not sentences:
        text = "I could not find a relevant sentence in the extracted PDF text."
    else:
        lines = []
        for hit in sentences:
            lines.append(
                f"[{hit.rank}] (p. {hit.chunk.page_number}, score {hit.score:.3f}) "
                f"{hit.chunk.text}"
            )
        text = "\n".join(lines)
    return Answer(
        question=question,
        mode="hybrid-extractive" if hybrid else "extractive",
        text=text,
        sentences=sentences,
        chunks=chunks,
    )


def llm_answer(
    index: PdfIndex,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = False,
    backend: str = "auto",
    model: str | None = None,
) -> Answer:
    """Generate a short answer from retrieved chunks.

    backend:
      - auto: OpenAI-compatible API if OPENAI_API_KEY is set, else local Flan-T5
      - openai: OpenAI-compatible Chat Completions API
      - local: Hugging Face ``google/flan-t5-base`` (downloaded on first use)
    """
    extractive = extractive_answer(
        index, question, top_k=top_k, hybrid=hybrid, max_sentences=max(top_k, 4)
    )
    context = _format_context(extractive.chunks)
    if not context:
        return extractive

    chosen = backend
    if backend == "auto":
        chosen = "openai" if os.environ.get("OPENAI_API_KEY") else "local"

    if chosen == "openai":
        text = _openai_complete(question, context, model=model)
    elif chosen == "local":
        text = _flan_complete(question, context, model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

    return Answer(
        question=question,
        mode=f"llm-{chosen}",
        text=text.strip(),
        sentences=extractive.sentences,
        chunks=extractive.chunks,
    )


def _format_context(chunks: list[Hit]) -> str:
    blocks = []
    for hit in chunks:
        blocks.append(f"[p. {hit.chunk.page_number}] {hit.chunk.text}")
    return "\n\n".join(blocks)


def _openai_complete(question: str, context: str, model: str | None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Use the default extractive mode, "
            "or pass --llm local for a local Flan-T5 model."
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided PDF excerpts. "
                    "Cite page numbers like (p. 2). If the excerpts are insufficient, "
                    "say you cannot answer from the document."
                ),
            },
            {
                "role": "user",
                "content": f"Excerpts:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
    return body["choices"][0]["message"]["content"]


def _flan_complete(question: str, context: str, model: str | None) -> str:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Local LLM mode requires extra packages. Install them with:\n"
            "  pip install 'transformers>=4.40' torch\n"
            "Or unset --llm and use the default extractive mode (no extra deps)."
        ) from exc

    model_name = model or "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    prompt = (
        "Answer the question using only the context. "
        "If the answer is not in the context, say you cannot tell.\n\n"
        f"Context: {context[:3000]}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = lm.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
