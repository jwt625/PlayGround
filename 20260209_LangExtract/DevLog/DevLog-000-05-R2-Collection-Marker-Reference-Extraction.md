# DevLog-000-05: Round 2 Collection, Marker Processing, and Reference Extraction

## Objective

Complete Round 2 (R2) for:
1. External reference paper sourcing and download into `raw_documents_R2`
2. Marker markdown ingestion for R2 corpus
3. Reference section extraction on R2 markdown and quality improvements

---

## R2 Collection Summary

### Input Candidate Set

- Source file: `tests/inference_test/output/external_refs_cited_gt10.json`
- Candidate references: 250

### Collection Workflow and Scripts

1. OA-first batch collection:
   - Script: `tests/inference_test/collect_external_refs_r2.py`
   - Outputs:
     - `semiconductor_processing_dataset/raw_documents_R2/manifest_documents_r2.jsonl`
     - `semiconductor_processing_dataset/raw_documents_R2/collection_attempts_r2.jsonl`
     - `semiconductor_processing_dataset/raw_documents_R2/collection_summary_r2.json`
2. Manual retry collection:
   - Script: `tests/inference_test/manual_retry_unresolved_r2.py`
3. Browser-assisted retry (Firefox profile clone, DOI navigation):
   - Script: `tests/inference_test/browser_retry_unresolved_r2.py`
   - Output:
     - `semiconductor_processing_dataset/raw_documents_R2/browser_retry_summary_r2.json`
4. User-assisted manual download ingest from `~/Downloads`:
   - Ingested and logged into the same R2 manifest/attempt logs.

### Curation Adjustments

- Excluded book/chapter/duplicate-edition entries from active paper corpus.
- Exclusion reflected in manifest as `status=skipped` and logged in attempts.

### Final R2 Document State

From `semiconductor_processing_dataset/raw_documents_R2/manifest_documents_r2.jsonl`:

| Metric | Count |
|--------|------:|
| Total records | 250 |
| `succeeded` | 238 |
| `skipped` | 10 |
| `failed` | 2 |

Additional artifacts:
- R2 papers folder: `semiconductor_processing_dataset/raw_documents_R2/papers/` (238 PDFs)
- Remaining unresolved list: `semiconductor_processing_dataset/raw_documents_R2/unresolved_references_r2.jsonl` (2 rows)
- Transfer archive used for remote sync:
  - `semiconductor_processing_dataset/raw_documents_R2/raw_documents_R2_papers_20260213.zip`

---

## Marker Processing (R2)

### Source and Ingestion

- Received marker conversion bundles:
  - `marker_processing_logs.zip`
  - `marker_r2_converted_md.zip`
- Extracted R2 markdown into:
  - `semiconductor_processing_dataset/processed_documents_R2/text_extracted/marker_r2/`

### R2 Markdown Corpus Size

- Markdown files present in R2 marker folder: 235

---

## Reference Section Extraction (R2)

### Baseline and Improvements

- Script updated: `tests/inference_test/extract_reference_sections.py`
- Improvements added:
  - CLI args (`--input-dir`, `--output-file`)
  - Expanded reference header patterns (including numbered headers)
  - Expanded line patterns for Marker variants (`</span>` transitions, malformed superscripts)
  - HTML unescape + normalization before regex checks
  - Lower cluster threshold and wider fallback window

### R2 Extraction Run

Command used:

```bash
python3 tests/inference_test/extract_reference_sections.py \
  --input-dir semiconductor_processing_dataset/processed_documents_R2/text_extracted/marker_r2 \
  --output-file tests/inference_test/output/reference_sections_r2.jsonl
```

Output:
- `tests/inference_test/output/reference_sections_r2.jsonl`

Final extraction status after script improvements + manual overrides:

| Metric | Count |
|--------|------:|
| Total markdown files | 235 |
| `success` | 229 |
| `no_references` | 6 |
| `error` | 0 |

Manual override note:
- For several malformed/edge documents, start/end ranges were manually inspected and patched into `reference_sections_r2.jsonl` to recover valid sections where present.

---

## Remaining Gaps

### Collection Gaps (2 unresolved references)

See:
- `semiconductor_processing_dataset/raw_documents_R2/unresolved_references_r2.jsonl`

### Extraction Gaps (6 markdowns with `no_references`)

These are primarily non-standard/partial documents (e.g., abstract/errata/proceedings fragments) or files without a recoverable bibliography block.

---

## Phase 2: LLM-Based Structured Reference Extraction (R2)

### 2026-02-14: Phase 2 Complete

**Script:** `tests/inference_test/batch_extract_refs_phase2.py`

Script updated with CLI arguments for generic R1/R2 support:
- `--input-file`: Input JSONL with reference sections
- `--output-file`: Output JSONL for extracted references
- `--progress-file`: Resume checkpoint file
- `--max-concurrent`: Concurrent API calls (default: 5)

**Command used:**

```bash
python3 tests/inference_test/batch_extract_refs_phase2.py \
  --input-file tests/inference_test/output/reference_sections_r2.jsonl \
  --output-file tests/inference_test/output/phase2_extracted_refs_r2.jsonl \
  --progress-file tests/inference_test/output/phase2_progress_r2.json
```

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `zai-org/GLM-4.7-FP8` |
| Max concurrent API calls | 5 |
| References per chunk | 10 (line-based splitting) |
| Max tokens | 8192 |
| Timeout | 600s |
| Max retries on JSON error | 3 |

**Results:**

| Metric | Count |
|--------|------:|
| Documents processed | 229/229 (100%) |
| Total references extracted | 11,914 |
| Documents with 0 refs | 0 |
| Chunk warnings (retried) | 85 |
| Chunk errors (failed after 3 retries) | 15 |

**Runtime:** ~10.5 hours (started 2026-02-13 19:50, completed 2026-02-14 06:24)

**Top 5 documents by reference count:**

| Document | Ref Count |
|----------|----------:|
| r2_10_1103_revmodphys_93_025005 | 1,573 |
| r2_10_1103_revmodphys_86_153 | 949 |
| r2_arxiv_1311_6759 | 279 |
| r2_10_1088_1361_6633_ab3a7e | 266 |
| r2_10_1103_revmodphys_82_1155 | 174 |

**Quality Check (sampled entries):**

References are correctly parsed with structured fields including:
- `ref_num`, `title`, `authors`, `year`, `journal`, `volume`, `pages`
- Optional: `doi`, `arxiv_id`

Example extracted reference:
```json
{
  "ref_num": 16,
  "title": "Lifetime renormalization of weakly anharmonic superconducting qubits...",
  "authors": ["M. Malekakhlagh", "A. Petrescu", "H. E. Tureci"],
  "year": 2020,
  "journal": "Physical Review B",
  "volume": "101",
  "pages": "134509",
  "doi": "10.1103/PhysRevB.101.134509",
  "arxiv_id": "1809.04667"
}
```

**Output Files:**
- `tests/inference_test/output/phase2_extracted_refs_r2.jsonl` - 229 documents, 11,914 refs
- `tests/inference_test/output/phase2_progress_r2.json` - Resume checkpoint
- `tests/inference_test/output/phase2_extracted_refs_r2_run_20260213_195056.log` - Run log

---

## Remaining Gaps

### Collection Gaps (2 unresolved references)

See:
- `semiconductor_processing_dataset/raw_documents_R2/unresolved_references_r2.jsonl`

### Extraction Gaps (6 markdowns with `no_references`)

These are primarily non-standard/partial documents (e.g., abstract/errata/proceedings fragments) or files without a recoverable bibliography block.

---

## Phase 3: Combined Citation Graph (R1 + R2)

**Date:** 2026-02-14

**Script:** `tests/inference_test/build_citation_graph.py`

The script was updated to accept multiple input files and metadata directories via `--input` and `--metadata-dir` arguments (can be specified multiple times).

**Command:**
```bash
cd tests/inference_test && python3 build_citation_graph.py \
  -i output/phase2_extracted_refs.jsonl \
  -i output/phase2_extracted_refs_r2.jsonl \
  -m ../../data/collection_r1/metadata \
  -o output
```

**Combined Results:**

| Metric | Value |
|--------|------:|
| Total raw references | 36,989 |
| - From R1 (260 docs) | 25,075 |
| - From R2 (229 docs) | 11,914 |
| Total documents | 489 |
| Unique references | 17,171 |
| - In-dataset | 193 |
| - External | 16,978 |
| Dedup ratio | 2.15x |
| Graph nodes | 17,471 |
| Graph edges | 33,653 |
| Inter-dataset citations | 896 |

**Match Method Breakdown:**

| Method | Count |
|--------|------:|
| DOI | 5,627 |
| Composite key | 9,778 |
| arXiv ID | 1,194 |
| Hash (fallback) | 572 |

**Top 10 Most-Cited External References:**

| Rank | Citations | Title (truncated) | Year |
|-----:|----------:|-------------------|-----:|
| 1 | 213 | These original works are the preliminary steps tow... | 2007 |
| 2 | 192 | Strong coupling of a single photon to a supercondu... | 2008 |
| 3 | 122 | Purification of noisy entanglement and faithful te... | 2010 |
| 4 | 119 | Blais, A., J. Gambetta, A. Wallraff, D. I. Schuste... | 2021 |
| 5 | 116 | Fabrication process and properties of fully-planar... | 2015 |
| 6 | 96 | Evidence for interacting two-level systems from th... | 2005 |
| 7 | 92 | Surface codes: towards practical large-scale quant... | 2012 |
| 8 | 81 | In current realizations of the 3D-transmon qubits,... | 2009 |
| 9 | 79 | Protocols for optimal readout of qubits using a co... | 2006 |
| 10 | 76 | High-Fidelity Readout in Circuit Quantum Electrody... | 2013 |

**Output Files:**
- `tests/inference_test/output/unique_references.jsonl` - 17,171 deduplicated references
- `tests/inference_test/output/citation_graph.json` - Graph with 17,471 nodes, 33,653 edges
- `tests/inference_test/output/dedup_stats.json` - Statistics and top 100 external references

### 2026-02-14: Top-100 Metadata Verification and Repair

After generating `top_100_external_refs_R2.md`, DOI-based verification found significant metadata corruption (wrong title/year/authors attached to valid DOI links).

**Verification artifact:**
- `tests/inference_test/output/top_100_external_refs_R2_verification.md`

**Verification summary:**

| Metric | Value |
|--------|------:|
| Total entries in list | 100 |
| DOI-checkable entries | 96 |
| Hard mismatches (title/year/authors) | 49 |
| Fully matched | 47 |
| Not auto-checkable (no DOI/arXiv link) | 4 |

**Observed failure mode:**
- `top_100_external_refs_R2.md` and `dedup_stats.json` contained rows where `doi` was correct but `title`, `year`, and/or `authors` were inherited from unrelated extracted reference text.
- This indicates canonical metadata drift in upstream deduplicated records (not only markdown rendering issues).

**Safety-first update process:**
- Created timestamped backups before any repair write:
  - `tests/inference_test/output/top_100_external_refs_R2.md.bak_20260214_114124`
  - `tests/inference_test/output/dedup_stats.json.bak_20260214_114124`
- Added repair script:
  - `tests/inference_test/correct_top100_refs_metadata.py`

**Repair actions performed:**
1. Canonicalized metadata for DOI/arXiv-linked entries in:
   - `tests/inference_test/output/top_100_external_refs_R2.md`
2. Updated `top_100_external` block in:
   - `tests/inference_test/output/dedup_stats.json`
   - Patched by rank alignment with corrected markdown to keep downstream source summary consistent.

**Repair result summary:**

| Update target | Count |
|--------------|------:|
| Markdown entries updated | 96 |
| `dedup_stats.json` top-100 rows updated | 79 |

Note: 4 entries still require manual handling because no DOI/arXiv link is available in the current list format.

---

## Next Steps

- [x] Run Phase 3: Merge R2 references into combined citation graph with R1
- [x] Generate updated statistics and top-cited external papers list
- [ ] Generate R2 metadata files (currently only R1 has 195 metadata files)
- [ ] Analyze citation patterns between R1 and R2 documents

---

## Reproducibility Pointers

- Collection logs and state:
  - `semiconductor_processing_dataset/raw_documents_R2/collection_attempts_r2.jsonl`
  - `semiconductor_processing_dataset/raw_documents_R2/manifest_documents_r2.jsonl`
- Phase 1 extraction result:
  - `tests/inference_test/output/reference_sections_r2.jsonl`
- Phase 2 extraction result:
  - `tests/inference_test/output/phase2_extracted_refs_r2.jsonl`
- Phase 3 combined outputs:
  - `tests/inference_test/output/unique_references.jsonl`
  - `tests/inference_test/output/citation_graph.json`
  - `tests/inference_test/output/dedup_stats.json`
- Key scripts:
  - `tests/inference_test/collect_external_refs_r2.py`
  - `tests/inference_test/manual_retry_unresolved_r2.py`
  - `tests/inference_test/browser_retry_unresolved_r2.py`
  - `tests/inference_test/extract_reference_sections.py`
  - `tests/inference_test/batch_extract_refs_phase2.py`
  - `tests/inference_test/build_citation_graph.py`
