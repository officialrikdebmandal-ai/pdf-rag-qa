"""Command-line interface for pdfqa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdfqa import __version__
from pdfqa.answer import extractive_answer, llm_answer
from pdfqa.cache import DEFAULT_CACHE_DIR
from pdfqa.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from pdfqa.embed import DEFAULT_MODEL
from pdfqa.extract import EmptyPdfError
from pdfqa.index import PdfIndex, build_index
from pdfqa.retrieve import DEFAULT_SENTENCES, DEFAULT_TOP_K


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except EmptyPdfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pdfqa",
        description="Answer questions over a PDF using retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"pdfqa {__version__}")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Ask a single question about a PDF")
    _add_index_args(ask)
    _add_query_args(ask)
    ask.add_argument("question", help="The question to ask")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Interactive question loop over a PDF")
    _add_index_args(chat)
    _add_query_args(chat)
    chat.set_defaults(func=cmd_chat)

    index_cmd = sub.add_parser("index", help="Build (or refresh) the retrieval index")
    _add_index_args(index_cmd)
    index_cmd.set_defaults(func=cmd_index)

    return parser


def _add_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", type=Path, help="Path to the PDF")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for cached indexes (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild the index and do not read or write the cache",
    )


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve")
    parser.add_argument(
        "--sentences",
        type=int,
        default=DEFAULT_SENTENCES,
        help="Sentences to show in an extractive answer",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse MiniLM cosine ranks with BM25 via reciprocal rank fusion",
    )
    parser.add_argument(
        "--llm",
        nargs="?",
        const="auto",
        choices=["auto", "openai", "local"],
        default=None,
        help=(
            "Optional abstractive answer. 'auto' uses an OpenAI-compatible API "
            "when OPENAI_API_KEY is set, otherwise local Flan-T5. "
            "Default mode is extractive and needs no API key."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the LLM model name (e.g. gpt-4o-mini or google/flan-t5-base)",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also print the retrieved chunks under the answer",
    )


def _load_index(args, show_progress: bool = True) -> PdfIndex:
    print(f"Indexing {args.pdf} ...", flush=True)
    index = build_index(
        pdf_path=args.pdf,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        show_progress=show_progress,
    )
    origin = "cache" if index.from_cache else "built"
    print(
        f"Ready ({origin}): {index.n_pages} pages, {index.n_chunks} chunks, "
        f"model={index.model_name}, cache_key={index.cache_key}",
        flush=True,
    )
    return index


def _answer_from_args(index: PdfIndex, question: str, args):
    if args.llm:
        return llm_answer(
            index,
            question,
            top_k=args.top_k,
            hybrid=args.hybrid,
            backend=args.llm,
            model=args.llm_model,
        )
    return extractive_answer(
        index,
        question,
        top_k=args.top_k,
        max_sentences=args.sentences,
        hybrid=args.hybrid,
    )


def _print_answer(answer, show_chunks: bool = False) -> None:
    print()
    print(f"Question: {answer.question}")
    print(f"Mode:     {answer.mode}")
    print()
    print("Answer:")
    print(answer.text)
    if show_chunks and answer.chunks:
        print()
        print("Retrieved chunks:")
        for hit in answer.chunks:
            preview = hit.chunk.text
            if len(preview) > 320:
                preview = preview[:317] + "..."
            print(
                f"  - chunk {hit.chunk.chunk_id} "
                f"(p. {hit.chunk.page_number}, {hit.source} {hit.score:.3f}): {preview}"
            )
    print()


def cmd_ask(args) -> int:
    index = _load_index(args)
    answer = _answer_from_args(index, args.question, args)
    _print_answer(answer, show_chunks=args.show_chunks)
    return 0


def cmd_index(args) -> int:
    _load_index(args)
    return 0


def cmd_chat(args) -> int:
    index = _load_index(args)
    print("Interactive mode. Type a question, or /quit, /hybrid, /dense, /help.")
    hybrid = bool(args.hybrid)
    while True:
        try:
            raw = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not raw:
            continue
        if raw in {"/quit", "/exit", "/q"}:
            break
        if raw == "/hybrid":
            hybrid = True
            print("Retrieval set to hybrid (MiniLM + BM25).")
            continue
        if raw == "/dense":
            hybrid = False
            print("Retrieval set to dense (MiniLM only).")
            continue
        if raw == "/help":
            print("Commands: /hybrid, /dense, /quit")
            continue
        args.hybrid = hybrid
        answer = _answer_from_args(index, raw, args)
        _print_answer(answer, show_chunks=args.show_chunks)
    return 0
