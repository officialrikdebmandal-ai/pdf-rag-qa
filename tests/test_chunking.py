from pdfqa.chunk import chunk_pages, _window_starts
from pdfqa.extract import PageText
import pytest


def _page(n: int, text: str) -> PageText:
    return PageText(page_number=n, text=text)


def test_window_starts_include_the_tail():
    # 25 words, size 10, overlap 2 -> step 8. Final window must start at 15
    # so words 16-25 are not dropped.
    starts = _window_starts(25, chunk_size=10, overlap=2)
    assert starts[0] == 0
    assert starts[-1] == 15
    assert 8 in starts


def test_short_page_is_a_single_chunk():
    pages = [_page(1, "alpha beta gamma")]
    chunks = chunk_pages(pages, chunk_size=180, overlap=36)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "alpha beta gamma"
    assert chunks[0].word_start == 0
    assert chunks[0].word_end == 3


def test_chunks_do_not_cross_page_boundaries():
    page1 = " ".join(f"p1w{i}" for i in range(20))
    page2 = " ".join(f"p2w{i}" for i in range(20))
    chunks = chunk_pages([_page(1, page1), _page(2, page2)], chunk_size=8, overlap=2)
    assert chunks
    assert all("p1w" in c.text and "p2w" not in c.text for c in chunks if c.page_number == 1)
    assert all("p2w" in c.text and "p1w" not in c.text for c in chunks if c.page_number == 2)
    assert {c.page_number for c in chunks} == {1, 2}


def test_overlap_repeats_boundary_words():
    # 18 words, size 8, overlap 3 -> step 5, starts 0,5,10 (tail already aligned).
    words = [f"w{i}" for i in range(18)]
    chunks = chunk_pages([_page(1, " ".join(words))], chunk_size=8, overlap=3)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:3]
    assert first_tail == second_head


def test_invalid_overlap_raises():
    pages = [_page(1, "just a few words")]
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=0, overlap=0)
