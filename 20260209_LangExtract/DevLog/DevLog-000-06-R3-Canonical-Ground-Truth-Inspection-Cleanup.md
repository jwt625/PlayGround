# DevLog-000-06: R3 Canonical Ground Truth, Inspection, and Cleanup

## Objective

Prepare Round 3 (R3) sourcing on a reliable canonical baseline by:
1. Diagnosing graph/classification and metadata inconsistencies
2. Building a canonical reference registry with sourcing status across rounds
3. Regenerating R3 target lists and running clean collection retries
4. Performing manual inspection/cleanup (including Downloads ingest)

---

## Why This Work Was Needed

During R3 prep, several issues were observed:
- Top external lists contained metadata corruption (title/year/authors not matching DOI metadata)
- The graph "external" label and collection "downloaded" status were being interpreted as the same thing
- R3 retries were mixing old run state and stale input files
- Some manually downloaded PDFs in `~/Downloads` were not being ingested into R3 manifest state

---

## Canonical and Graph Fixes

### 1) Citation graph source identity coverage fix

**Script updated:** `tests/inference_test/build_citation_graph.py`

Added support for collection manifest inputs so in-dataset/source matching is not limited to metadata JSON directories only.

New CLI argument:
- `--manifest` / `-c` (repeatable)

Used for rebuild:

```bash
cd tests/inference_test && python3 build_citation_graph.py \
  -i output/phase2_extracted_refs.jsonl \
  -i output/phase2_extracted_refs_r2.jsonl \
  -m ../../data/collection_r1/metadata \
  -c ../../data/collection_r1/manifest_documents.jsonl \
  -c ../../data/collection_r2/manifest_documents_r2.jsonl \
  -o output
```

Result after rebuild:
- `unique_references`: 17,297
- `in_dataset_references`: 505
- `external_references`: 16,792
- `documents_in_dataset_with_metadata`: 519

---

### 2) Canonical registry for cross-round truth

**Script created:** `tests/inference_test/build_canonical_reference_registry.py`

Outputs:
- `tests/inference_test/output/canonical_reference_registry.jsonl`
- `tests/inference_test/output/canonical_reference_registry_summary.json`

Registry fields include:
- canonical reference identity and metadata
- graph labels: `in_dataset`, `external`
- round-level sourcing flags and statuses for R1/R2/R3/R3new
- strict vs relaxed matching provenance
- `downloaded_any_round` and `downloaded_any_round_strict`

Important clarification encoded in summary:
- `sourced_*` counts are **reference-level counts**, not document counts.

---

### 3) DOI enrichment for relaxed-only unmatched references

**Script created:** `tests/inference_test/enrich_relaxed_refs_doi.py`

Outputs:
- `tests/inference_test/output/relaxed_only_doi_candidates.jsonl`
- `tests/inference_test/output/relaxed_only_doi_candidates_summary.json`
- `tests/inference_test/output/canonical_reference_registry_enriched.jsonl`

Resolution result (relaxed-only missing DOI subset):
- target: 269
- resolved with confident DOI: 141
- unresolved: 128

Global DOI coverage after enrichment snapshot:
- total refs: 17,297
- refs with DOI: 5,842
- refs still missing DOI: 11,455

---

## R3 Target List and Collection Cleanup

### 1) New R3 processing list (strict not-downloaded external queue)

Generated:
- `tests/inference_test/output/top_external_refs_for_r3_processing.md`

A fresh collector input was regenerated from this file:
- `tests/inference_test/output/r3_processing_input.json`

---

### 2) Clean rerun (state reset)

Before rerun, stale state was removed:
- `semiconductor_processing_dataset/raw_documents_R3/manifest_documents_r3.jsonl`
- `semiconductor_processing_dataset/raw_documents_R3/collection_attempts_r3.jsonl`
- `semiconductor_processing_dataset/raw_documents_R3/collection_summary_r3.json`
- all PDFs under `semiconductor_processing_dataset/raw_documents_R3/papers/`

Then reran collection from current 100-entry list.

Rerun summary:
- total refs: 100
- succeeded: 48
- failed: 52
- skipped: 0

---

### 3) Manual Downloads inspection and ingest

After OA retries, manual filename-to-failed-reference matching was performed against `~/Downloads` and corresponding refs were marked complete in R3 manifest.

Current R3 status (`collection_summary_r3.json`):
- total refs: 100
- succeeded: 63
- failed: 37
- skipped: 0

Key R3 artifacts:
- `semiconductor_processing_dataset/raw_documents_R3/manifest_documents_r3.jsonl`
- `semiconductor_processing_dataset/raw_documents_R3/collection_attempts_r3.jsonl`
- `semiconductor_processing_dataset/raw_documents_R3/collection_summary_r3.json`
- `semiconductor_processing_dataset/raw_documents_R3/failed_external_refs_r3.md`
- `semiconductor_processing_dataset/raw_documents_R3/failed_external_refs_r3_validation.md`
- `semiconductor_processing_dataset/raw_documents_R3/downloads_match_report_r3.md`

---

## Metadata Quality Cleanup

- Revalidated failed list DOI/title coherence and rewrote failed list with cleaner entries.
- Segregated DOI-unresolvable and missing-DOI entries in validation output.
- Removed stale `top_100_external_refs_*` markdown working files to avoid confusion with the new R3 processing list.

---

## Current State

- Graph and dedup now include manifest-based source identity coverage.
- Canonical registry exists with explicit strict/relaxed provenance.
- R3 list processing has been rerun cleanly from current input.
- R3 recovery from manual downloads has been applied and reflected in manifest/summary.

---

## Update (2026-02-15): R3 Processing and Graph Rebuild Completed

### 1) R3 marker processing completed

- R3 marker outputs were ingested from `marker_all_outputs_R3.zip`.
- Phase 1 extraction completed:
  - input: `semiconductor_processing_dataset/processed_documents/text_extracted/marker_r3/`
  - output: `tests/inference_test/output/reference_sections_r3.jsonl`
  - files found: 63
  - success: 62
  - no references found: 1
- Phase 2 extraction completed:
  - output: `tests/inference_test/output/phase2_extracted_refs_r3.jsonl`
  - progress: `tests/inference_test/output/phase2_progress_r3.json`
  - completed documents: 62/62

### 2) Citation graph rebuilt with R1 + R2 + R3

Rebuilt using:
- `tests/inference_test/output/phase2_extracted_refs.jsonl`
- `tests/inference_test/output/phase2_extracted_refs_r2.jsonl`
- `tests/inference_test/output/phase2_extracted_refs_r3.jsonl`
- manifests: R1/R2/R3

Current results:
- `unique_references`: 18,429
- strict in-dataset: 600
- strict external: 17,829
- effective in-dataset/seen: 770
- effective external: 17,659

Outputs refreshed:
- `tests/inference_test/output/unique_references.jsonl`
- `tests/inference_test/output/citation_graph.json`
- `tests/inference_test/output/dedup_stats.json`

### 3) Canonical registry rebuilt from refreshed unique references

Outputs refreshed:
- `tests/inference_test/output/canonical_reference_registry.jsonl`
- `tests/inference_test/output/canonical_reference_registry_summary.json`

Current summary totals:
- total refs: 18,429
- strict in-dataset: 600
- strict external: 17,829
- effective in-dataset: 808
- effective external: 17,621

### 4) Visualization/classification fix for resurfaced refs

Issue:
- Highly cited items already attempted/processed in R1-R3 were still being interpreted as "truly external" in analysis flow.

Fix implemented:
- Added effective classification fields in graph/registry outputs:
  - `in_dataset_effective`
  - `in_dataset_effective_strict`
  - manifest seen flags and status in graph nodes
- Updated `tests/inference_test/output/citation_graph_viewer.html` to render:
  - strict in-dataset
  - seen/processed in manifests
  - truly external

Note:
- Browser deep refresh is required to ensure the viewer loads the latest `citation_graph.json`.

---

## Next Steps

1. Add/standardize DOI correction rules for known metadata-conflict references directly in canonical build flow.
2. Generate and validate the next true-external top list from effective classification only.
3. Run targeted collection retry on true-external high-citation unresolved refs.
4. Add a regression check that prevents previously seen/processed refs from being reported as truly external.
