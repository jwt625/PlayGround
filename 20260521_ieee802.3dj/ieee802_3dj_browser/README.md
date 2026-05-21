# IEEE 802.3dj Browser Corpus

This directory is the prepared text corpus for a future IEEE 802.3dj document browser.

## Layout

- `index.html`: local three-panel browser for navigating meetings, presentations, metadata, PDFs, and extracted text.
- `app.js`: browser logic for filtering, sorting, metadata display, and extracted-text rendering.
- `styles.css`: browser styling.
- `metadata/documents.json`: canonical manifest for the discovered PDF presentations.
- `metadata/talks.json`: browser-oriented copy of the presentation metadata and extracted-file locations.
- `metadata/progress.jsonl`: append-only extraction progress log, one JSON object per processed PDF.
- `extracted_text/<meeting>/<document>.md`: markdown-flavored extracted text for analysis and browser rendering.
- `extracted_text/<meeting>/<document>.txt`: plain text copy for search, indexing, and lightweight analysis.

The manifest was generated from `../IEEE-802p3dj-cache-checklist.md` and enriched from each cached meeting `index.html`. It keeps source parent URL/path, presentation URL, actual cached source URL/path, website title, presentation date, presenters, affiliations, meeting bucket, file size, SHA-256, output paths, and extraction metadata.

## Current Status

- Documents discovered: 986 PDFs across 25 meeting directories.
- Baseline extraction: complete for all 986 PDFs with PyMuPDF.
- Markdown outputs: 986.
- Plain text outputs: 986.
- Marker OCR: installed in `../.venv-marker` and smoke-tested successfully on one PDF.
- Website metadata: populated for all 986 PDFs.

Marker produced a successful result locally, but the smoke test processed a 49-page presentation in about 7.5 minutes. A full Marker OCR pass across the whole cache should be run as a resumable batch or overnight job.

## Commands

From the repository root:

```bash
python3 -m http.server 8765
```

Then open `http://localhost:8765/ieee802_3dj_browser/`.
If that port is busy, use another port such as `8766`.

Extraction commands:

```bash
source .venv-marker/bin/activate
python tools/extract_pdf_text.py --manifest-only
python tools/extract_pdf_text.py --method pymupdf --resume
python tools/extract_pdf_text.py --method marker --resume --timeout 1800
```

Use PyMuPDF for fast full-corpus text extraction and Marker when layout/OCR quality matters enough to pay the runtime cost.
