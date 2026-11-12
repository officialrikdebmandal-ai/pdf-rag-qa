from pathlib import Path

from pdfqa.answer import extractive_answer
from pdfqa.extract import extract_pages
from pdfqa.index import build_index
from pdfqa.retrieve import dense_search, hybrid_search

from tests.pdfutil import write_simple_pdf

PAGE_FRUIT = (
    "Mango trees grow in tropical climates and produce a sweet orange fruit "
    "in the summer harvest. Farmers often pick mangoes by hand when the skin "
    "begins to turn from green to yellow. This page does not mention medical imaging."
)

PAGE_MRI = (
    "Magnetic resonance imaging is used to study the brain. "
    "FLAIR, or Fluid-Attenuated Inversion Recovery, suppresses the bright signal "
    "of cerebrospinal fluid so that edema next to the ventricles is easier to see. "
    "T1-weighted images are used for anatomy, and gadolinium contrast highlights "
    "areas where the blood-brain barrier is disrupted. Mangoes are not discussed here."
)


def test_extract_pages_keeps_page_numbers(tmp_path: Path):
    pdf = write_simple_pdf(tmp_path / "tiny.pdf", [PAGE_FRUIT, PAGE_MRI])
    pages = extract_pages(pdf)
    assert [p.page_number for p in pages] == [1, 2]
    assert "Mango trees" in pages[0].text
    assert "FLAIR" in pages[1].text


def test_dense_retrieval_returns_the_correct_page(tmp_path: Path):
    pdf = write_simple_pdf(tmp_path / "tiny.pdf", [PAGE_FRUIT, PAGE_MRI])
    index = build_index(pdf, cache_dir=tmp_path / "cache", show_progress=False)

    mri_hits = dense_search(index, "Which sequence suppresses cerebrospinal fluid?", top_k=1)
    assert mri_hits, "expected at least one retrieval hit"
    assert mri_hits[0].chunk.page_number == 2
    assert "FLAIR" in mri_hits[0].chunk.text

    fruit_hits = dense_search(index, "When do farmers pick mangoes?", top_k=1)
    assert fruit_hits[0].chunk.page_number == 1
    assert "mangoes" in fruit_hits[0].chunk.text.lower()


def test_extractive_answer_cites_pages_and_cache_roundtrip(tmp_path: Path):
    pdf = write_simple_pdf(tmp_path / "tiny.pdf", [PAGE_FRUIT, PAGE_MRI])
    cache = tmp_path / "cache"
    first = build_index(pdf, cache_dir=cache, show_progress=False)
    assert first.from_cache is False
    second = build_index(pdf, cache_dir=cache, show_progress=False)
    assert second.from_cache is True
    assert second.n_chunks == first.n_chunks
    assert second.embeddings.shape == first.embeddings.shape

    answer = extractive_answer(
        second,
        "What does FLAIR suppress so that edema is easier to see?",
        top_k=2,
        max_sentences=3,
    )
    assert answer.sentences
    assert any(hit.chunk.page_number == 2 for hit in answer.sentences)
    assert "cerebrospinal fluid" in answer.text.lower()
    # Hybrid should also prefer the MRI page for this query.
    hybrid = hybrid_search(second, "What does FLAIR suppress?", top_k=1)
    assert hybrid[0].chunk.page_number == 2
