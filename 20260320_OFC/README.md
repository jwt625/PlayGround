# OFC 2026 Corpus Extraction and Word Map

This workspace builds a corpus-wide term map from the OFC 2026 paper archive at `/home/ubuntu/ofc_pdfs_20260321_full.zip`.

## Environment

The repo uses the uv-managed virtual environment at `.venv`.

Install the extra analysis dependencies used by the scripts:

```bash
./.venv/bin/python -m pip install nltk wordcloud matplotlib pypdf
```

## Commands

Run the full pipeline:

```bash
./scripts/run_ofc2026_pipeline.sh /home/ubuntu/ofc_pdfs_20260321_full.zip
```

The full runner launches two extraction shards in parallel, one per GPU, writes shard progress under `tmp/ofc2026_shards/`, merges those shard indexes into `paper_text_index.json`, then runs the corpus analysis step.

Run extraction only:

```bash
./.venv/bin/python scripts/ofc2026_extract_marker.py \
  --archive /home/ubuntu/ofc_pdfs_20260321_full.zip \
  --pdf-dir data/ofc2026_pdfs \
  --output-dir extracted_text \
  --index paper_text_index.json \
  --fail-log extraction_failures.jsonl
```

Run corpus analysis only:

```bash
./.venv/bin/python scripts/ofc2026_build_wordmap.py \
  --index paper_text_index.json \
  --top-terms corpus_top_terms.csv \
  --top-bigrams corpus_top_bigrams.csv \
  --top-trigrams corpus_top_trigrams.csv \
  --wordmap-png wordmap.png \
  --wordmap-svg wordmap.svg \
  --tfidf-json paper_tfidf_keywords.json
```

## Outputs

- `extracted_text/`: one Markdown file and one plain-text file per paper
- `paper_text_index.json`: filename-to-output-path mapping with extraction status and method
- `tmp/ofc2026_shards/`: shard-local indexes and failure logs used by the parallel extractor
- `extraction_failures.jsonl`: per-paper failure log for Marker and fallback extraction failures
- `corpus_top_terms.csv`: top normalized unigrams
- `corpus_top_bigrams.csv`: top normalized bigrams
- `corpus_top_trigrams.csv`: top normalized trigrams
- `paper_tfidf_keywords.json`: optional per-paper TF-IDF keywords
- `wordmap.png`: raster word cloud for the full corpus
- `wordmap.svg`: vector word cloud for the full corpus

## Cleaning Choices

- Primary extraction uses Marker via `PdfConverter`.
- If Marker fails or returns empty text, the script falls back to `pypdf` and records the fallback in `paper_text_index.json`.
- The extraction step saves both raw Markdown-like output and a plain-text version with HTML tags, Markdown links, code fences, image syntax, and repeated formatting markers stripped.
- The analysis step lowercases text, removes URLs and email addresses, and tokenizes with support for mixed technical strings such as `16-qam`, `si3n4`, and slash-separated optical terms.
- Boilerplate removal is corpus-aware:
  - repeated short lines that appear across many papers are dropped
  - known conference/header noise like `OFC 2026`, DOI/ISBN lines, page labels, copyright lines, and session-code style lines are dropped
- Stopword removal starts from scikit-learn English stopwords and adds a small paper-generic list such as `figure`, `table`, `paper`, and `results`.
- Lemmatization is applied to alphabetic tokens with WordNet; mixed alphanumeric technical tokens are preserved as-is so domain vocabulary is not collapsed incorrectly.

## Reruns and Robustness

- Already processed PDFs are skipped on rerun if their indexed text outputs already exist.
- Extraction failures are appended to `extraction_failures.jsonl` instead of stopping the batch.
- The ZIP archive is unpacked incrementally into `data/ofc2026_pdfs`, ignoring `__MACOSX` entries.
