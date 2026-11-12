# pdfqa

A small Python tool that answers questions over a PDF using retrieval.

It extracts text with page numbers, splits it into overlapping word chunks, embeds those chunks with `all-MiniLM-L6-v2`, and returns the best-matching sentences with page citations. The default path is **extractive** and needs **no paid API key**. An optional LLM mode can rewrite the retrieved passages if you set `OPENAI_API_KEY` or pass `--llm local`.

This was built as a take-home for the AI Automation Pipeline Intern role at Araskova Labs.

Author: Rikdeb Mandal.

## Quickstart

The commands below assume Python 3.10+ and should be run from the repository root. The first run downloads the MiniLM model from Hugging Face (about 80 MB) and may take a minute. Later runs reuse the local model cache and a disk index under `.pdfqa_cache/`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you are not logged in to Hugging Face, the libraries may print a one-line tip on stderr about unauthenticated rate limits. It is harmless and does not change the answer.

### Index the sample PDF, then ask a question

```bash
python -m pdfqa index samples/brain_mri_overview.pdf
python -m pdfqa ask samples/brain_mri_overview.pdf "What is a glioma?"
```

Actual output from this repository (the first `index` call built the cache; the `ask` call below reused it):

```
Indexing samples/brain_mri_overview.pdf ...
Ready (cache): 3 pages, 7 chunks, model=all-MiniLM-L6-v2, cache_key=60ade482199bc05c

Question: What is a glioma?
Mode:     extractive

Answer:
[1] (p. 1, score 0.823) A glioma is a tumor that originates from glial cells — astrocytes, oligodendrocytes, or ependymal cells.
[2] (p. 1, score 0.654) High-grade gliomas, especially glioblastoma, are aggressive, infiltrate surrounding white matter, and are difficult to resect completely.
[3] (p. 1, score 0.631) Low-grade gliomas grow more slowly but may still undergo malignant transformation over time.
[4] (p. 2, score 0.577) about 15% Transsphenoidal surgery or medical therapy, depending on type Low-grade glioma 20–45 years Cerebral hemispheres, often frontal or temporal under 15% Surgery, with radiotherapy or chemotherapy in selected cases Table 1.
```

The first sentence is the right definition, taken from page 1. Sentence 4 is leftover table text; that noise is discussed under [Limitations](#limitations-and-next-steps).

### Interactive mode

```bash
python -m pdfqa chat samples/brain_mri_overview.pdf
```

Type a question, or `/hybrid`, `/dense`, or `/quit`. A scripted session from this repo:

```
Indexing samples/brain_mri_overview.pdf ...
Ready (cache): 3 pages, 7 chunks, model=all-MiniLM-L6-v2, cache_key=60ade482199bc05c
Interactive mode. Type a question, or /quit, /hybrid, /dense, /help.

> 
Question: What is a glioma?
Mode:     extractive

Answer:
[1] (p. 1, score 0.823) A glioma is a tumor that originates from glial cells — astrocytes, oligodendrocytes, or ependymal cells.
[2] (p. 1, score 0.654) High-grade gliomas, especially glioblastoma, are aggressive, infiltrate surrounding white matter, and are difficult to resect completely.
[3] (p. 1, score 0.631) Low-grade gliomas grow more slowly but may still undergo malignant transformation over time.
[4] (p. 2, score 0.577) about 15% Transsphenoidal surgery or medical therapy, depending on type Low-grade glioma 20–45 years Cerebral hemispheres, often frontal or temporal under 15% Surgery, with radiotherapy or chemotherapy in selected cases Table 1.
```

### Useful flags

```bash
# Fuse MiniLM ranks with BM25 via reciprocal rank fusion
python -m pdfqa ask --hybrid samples/brain_mri_overview.pdf "How is glioblastoma generally managed?"

# Also print the retrieved chunks
python -m pdfqa ask --show-chunks samples/brain_mri_overview.pdf "What is a glioma?"

# Optional abstractive answer (not the default)
python -m pdfqa ask --llm openai samples/brain_mri_overview.pdf "What is a glioma?"
python -m pdfqa ask --llm local  samples/brain_mri_overview.pdf "What is a glioma?"
```

`--llm openai` calls an OpenAI-compatible Chat Completions API when `OPENAI_API_KEY` is set (`OPENAI_BASE_URL` and `OPENAI_MODEL` are honoured). `--llm local` uses `google/flan-t5-base` and needs extra packages: `pip install 'transformers>=4.40' torch`. `--llm` with no value auto-selects OpenAI if a key is present, otherwise the local model. The examples in this README all use the default extractive mode, which is what I actually ran.

Rebuild without reading or writing the cache with `--no-cache`. The cache directory can be changed with `--cache-dir`.

### Sample PDF

`samples/brain_mri_overview.pdf` is a short original teaching note about brain tumors and MRI, written for this demo and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is not clinical advice. The numbers in Table 1 and Figure 1 are rounded illustrations.

To regenerate it (optional, not needed to run the commands above):

```bash
pip install reportlab pillow
./scripts/download_sample.sh
```

### Tests

```bash
pytest
```

This run:

```
........                                                                 [100%]
```

The tests cover word-window chunking (overlap, page boundaries, the tail of a page) and retrieval over a tiny two-page PDF generated at test time: a question about FLAIR must come back from page 2, a question about mangoes from page 1, and a second `build_index` call must load the cache rather than re-embed.

## Design choices

**Text extraction with page numbers.** `pypdf` is a modest, well-maintained dependency and gives a 1-based page number for every snippet I later cite. It reads the text layer only, so scanned pages, figures, and some tables are incomplete or jumbled. That trade-off is accepted in the default path and is the source of the failure case below.

**Page-bounded word chunks, 180 words with 36-word overlap.** `all-MiniLM-L6-v2` truncates at 256 tokens, so a window of about 180 English words stays safely under that limit after tokenization. Overlap of 20% reduces the chance that a sentence is split on the boundary. Chunks never cross a page break, so every citation points to exactly one page. The last window is aligned to the end of the page so the tail is not dropped.

**MiniLM + cosine similarity.** I wanted a default that runs offline after one model download, on CPU, with no API bill. MiniLM is small, fast, and strong enough for short factual questions over a few pages. Embeddings are L2-normalized and compared with a NumPy dot product. The index is stored as a `.npz` matrix plus a JSON metadata file, keyed by the SHA-256 of the PDF together with the model name and chunk settings, so editing the file or the parameters invalidates the cache automatically.

**Optional hybrid retrieval.** `--hybrid` still runs MiniLM, but also scores chunks with BM25 (`rank_bm25`) and fuses the two ranked lists with Reciprocal Rank Fusion. RRF does not need the two score scales to be calibrated. A light suffix strip helps BM25 match simple variants such as *suppress* / *suppresses*, and exact RRF ties are broken with the dense cosine score so a two-document disagreement cannot be decided by array order.

**Extractive answers with sentence re-ranking.** The default answer is not a generated paragraph. The top chunks are split into sentences, those sentences are embedded, and the best ones are printed with a page number and a cosine score. Overlapping chunks can produce the same sentence twice; near-duplicates are dropped before printing. I chose this default because it stays inspectable: you can see the evidence, and it works with no API key. The optional `--llm` path is a thin layer on top of the same retrieved context.

## Evaluation note

I asked five factual questions of `samples/brain_mri_overview.pdf` in extractive mode and judged the **top-ranked sentence**. All five were correct. A sixth question, about a number that exists only in Figure 1, failed; that run is documented in the next section.

| Question | Top sentence (abridged) | Verdict |
| --- | --- | --- |
| What is a glioma? | "A glioma is a tumor that originates from glial cells..." (p. 1, 0.823) | Correct |
| Which MRI sequence is particularly useful for detecting edema next to the ventricles? | "FLAIR is particularly useful for detecting edema and lesions that lie next to the ventricles..." (p. 1, 0.604) | Correct |
| What contrast agent is used with T1-weighted images to show blood-brain barrier disruption? | "After gadolinium contrast is given, a T1-contrast (T1C) sequence shows regions where the blood–brain barrier is disrupted." (p. 1, 0.681) | Correct |
| What four labels commonly appear in public MRI datasets used for research? | "...four labels appear again and again: glioma, meningioma, pituitary tumor, and no tumor..." (p. 1, 0.769) | Correct |
| How is glioblastoma generally managed? | "Glioblastoma is generally managed with maximal safe resection followed by radiotherapy with temozolomide." (p. 2, 0.732) | Correct |

The same glioblastoma-management question with `--hybrid` returned the same top sentence (p. 2, 0.732). Lower-ranked rows sometimes include smashed Table 1 text; the top hit was still right.

Full captured output for the FLAIR question:

```
Question: Which MRI sequence is particularly useful for detecting edema next to the ventricles?
Mode:     extractive

Answer:
[1] (p. 1, score 0.604) FLAIR is particularly useful for detecting edema and lesions that lie next to the ventricles or the cerebral sulci, where bright cerebrospinal fluid would otherwise conceal the finding.
[2] (p. 1, score 0.483) MRI sequences used to characterise brain tumors Magnetic resonance imaging is the primary non-invasive modality for characterising brain tumors before surgery.
[3] (p. 2, score 0.461) Page 2 A typical research MRI study for tumor classification therefore includes at least T1, T1-contrast, T2, and FLAIR, although many open datasets also provide only a single sequence or a single two-dimensional slice.
[4] (p. 3, score 0.438) The most common failure modes in automated MRI classification include confusing glioma with meningioma when the mass sits near the cortical surface, missing small pituitary lesions on slices that barely include the sella, and over-calling tumor on scans that actually show post-treatment change or inflammation.
```

This is a short, single-document check, not a benchmark. It shows the tool works on prose questions when the answer is in the extracted text, and that lower-ranked sentences can still be messy.

## A case where it fails

**Question:** *What five-year survival percentage is shown for glioblastoma in Figure 1?*

**Ground truth, visible only in the figure graphic:** 7%. The bundled PDF states in prose that the exact percentages "are shown only in the figure and are not restated in this accompanying text." WHO grade 1 meningioma is shown as 88% and pituitary adenoma as 93%, also only in the image pixels.

**Actual output** from `python -m pdfqa ask samples/brain_mri_overview.pdf "What five-year survival percentage is shown for glioblastoma in Figure 1?"`:

```
Question: What five-year survival percentage is shown for glioblastoma in Figure 1?
Mode:     extractive

Answer:
[1] (p. 2, score 0.650) Figure 1 below is a teaching graphic of illustrative five-year survival by tumor type.
[2] (p. 3, score 0.635) Illustrative five-year survival by tumor type.
[3] (p. 2, score 0.602) about 15% Transsphenoidal surgery or medical therapy, depending on type Low-grade glioma 20–45 years Cerebral hemispheres, often frontal or temporal under 15% Surgery, with radiotherapy or chemotherapy in selected cases Table 1.
[4] (p. 1, score 0.561) Low-grade gliomas grow more slowly but may still undergo malignant transformation over time.
```

The tool never returns 7%. The best sentences are the figure *caption*, which does not contain the number. Worse, sentence 3 surfaces **about 15%**, a Table 1 figure for the *share of primary brain tumors*, not five-year survival. A reader who trusted that row would get the wrong quantity and the wrong number.

**Why it fails.** `pypdf` only extracts the PDF text layer. Figure 1 was embedded as a raster image so that the percentages exist only as pixels. There is nothing in the index for MiniLM to retrieve except the caption and nearby table text. Because the question mentions a percentage, retrieval also latches onto "about 15%" and "under 15%" from Table 1, whose column structure has already been flattened into a word stream.

Passing `--show-chunks` confirms the same gap: the highest-scoring chunks are table fragments and the caption ("Values appear only in the graphic itself"). None of them contain 7%.

**How to fix it.** A text-only pipeline cannot solve this. Practical next steps are: (1) render pages or embedded images and run OCR or a vision-language model over Figure 1, (2) parse tables with a layout-aware extractor (for example PyMuPDF or a table parser) so "15%" stays tied to the "share of primary brain tumors" column instead of floating next to a survival question, and (3) decline to answer when the retrieved text never mentions a number, rather than returning a nearby but unrelated percentage.

A related, fully reproduced failure: an image-only PDF with no text layer is rejected before retrieval. Against a one-page scanned-style file whose pixels said "The secret word is pineapple," the tool printed:

```
error: No extractable text in /tmp/pdfqa-runs/scanned_like.pdf. This often happens with scanned or image-only PDFs.
```

That behaviour is honest, but it is still a failure to answer, and the same OCR/vision fix would be required.

## Limitations and next steps

- **Tables** are extracted as a flattened word stream. Even when the top sentence is right, later sentences can be smashed rows from Table 1, as in the glioma and treatment answers above.
- **Running headers and footers** ("Page 2", the licence line) leak into chunks because they are ordinary text on every page.
- **Headings without a trailing full stop** get glued to the next sentence ("MRI sequences used to characterise brain tumors Magnetic resonance imaging...").
- **Extractive answers cannot combine facts** from distant pages into a single synthesized sentence. `--llm` can, but only from the text that retrieval actually found, and I did not rely on that path in the documented runs.
- **One short teaching PDF** is not a retrieval benchmark. A fair evaluation would use held-out questions, more than one document, and a metric such as recall@k or citation accuracy.
- **The default model is general-domain English.** A biomedically adapted embedder might help on clinical wording, at the cost of a larger download.

Next steps I would take if this were going further: a layout-aware parser for tables, OCR or a vision model for figures, header/footer stripping, a refusal when no retrieved sentence actually answers the question, and a small evaluation set with cited gold answers.

## License

The code is under the MIT License. Copyright (c) 2026 Rikdeb Mandal. The sample PDF is original work licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
