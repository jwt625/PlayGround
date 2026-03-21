# DevLog-001-00: Fabrication Knowledge Extraction and Process Graph Build Plan

## Date
2026-03-01

## Objective

Transition from citation-focused data prep to fabrication-knowledge extraction so the project can produce a high-quality, auditable process graph from the collected corpus.

Primary extraction goals:
1. Process steps
2. Recipes and parameters
3. Equipment and materials
4. Site/facility attribution (institution, cleanroom/fab)
5. Cross-document normalization and graph construction

---

## Why This Stage Now

R1-R3 collection and citation graphing are now mature enough to support knowledge extraction directly from the corpus.

Current state already in place:
- Stable corpus in `data/` and `semiconductor_processing_dataset/`
- Maintained citation pipeline in `tests/inference_test/pipeline/phase1..phase3e`
- Canonical registry and integrity checks for external-reference flow

Remaining gap:
- No standardized extraction pipeline yet for fabrication entities/relations
- Knowledge-base files exist but are mostly unpopulated and need systematic filling from grounded extractions

---

## Scope and Non-Goals

In scope:
- Implement a maintained Phase 4 pipeline for fab extraction, normalization, graph build, and validation
- Keep provenance on every extraction (document ID + source text + character span)
- Build deterministic normalization for common aliases (tools, materials, process naming)

Out of scope for this phase:
- New data collection rounds
- Expansive ontology redesign beyond practical extraction needs
- Fine-tuning custom models

---

## Phase 4 Pipeline Design

### Phase 4a: Extract Fabrication Entities (LLM, grounded)

Script:
- `tests/inference_test/pipeline/phase4a_extract_fab_entities.py`

Inputs:
- Marker markdown corpus (default):
  - `semiconductor_processing_dataset/processed_documents/text_extracted/marker/`
- Optional metadata hints:
  - `semiconductor_processing_dataset/processed_documents/metadata/`

Outputs:
- `tests/inference_test/output/phase4a_fab_entities_raw.jsonl`
- `tests/inference_test/output/phase4a_fab_entities_summary.json`

Extraction classes:
- `process_step`
- `recipe`
- `equipment`
- `material`
- `parameter`
- `site`
- `cleanroom`
- `facility`
- `sample_context`

Required provenance fields per extraction:
- `document_id`
- `source_text`
- `char_start`
- `char_end`
- `section_hint`

---

### Phase 4b: Normalize and Canonicalize

Script:
- `tests/inference_test/pipeline/phase4b_normalize_fab_entities.py`

Inputs:
- `phase4a_fab_entities_raw.jsonl`

Outputs:
- `tests/inference_test/output/phase4b_fab_entities_normalized.jsonl`
- `tests/inference_test/output/phase4b_fab_entities_summary.json`

Responsibilities:
- Normalize aliases/synonyms (equipment, materials, process terms)
- Generate canonical IDs
- Preserve relation hints from attributes
- Flag low-confidence or weak-provenance records

---

### Phase 4c: Build Fabrication Process Graph

Script:
- `tests/inference_test/pipeline/phase4c_build_fab_process_graph.py`

Inputs:
- `phase4b_fab_entities_normalized.jsonl`

Outputs:
- `tests/inference_test/output/phase4c_fab_process_graph.json`
- `tests/inference_test/output/phase4c_fab_process_graph_summary.json`

Graph components:
- Nodes: process steps, recipes, equipment, materials, parameters, sites/facilities
- Edges:
  - `step_uses_equipment`
  - `step_uses_material`
  - `step_has_parameter`
  - `step_occurs_at_site`
  - `step_precedes_step`
  - `document_mentions`

---

### Phase 4d: Validate Integrity and Readiness

Script:
- `tests/inference_test/pipeline/phase4d_validate_fab_integrity.py`

Inputs:
- Phase 4b normalized entities
- Phase 4c graph

Outputs:
- `tests/inference_test/output/phase4d_fab_validation_summary.json`

Checks:
- Required provenance completeness
- Character span validity
- Edge-node referential integrity
- Required-attribute coverage for relation-bearing entities

---

## Prompting and Extraction Strategy

Model framework:
- LangExtract with schema-constrained extraction
- Multi-pass extraction for recall on long docs
- Chunked inference to avoid long-context degradation

Guidance principles:
- Prefer explicit textual evidence over inferred claims
- Keep site attribution strict: only mark as factual when text evidence is explicit
- Retain unresolved/inferred flags rather than forcing hard assignments

---

## Acceptance Criteria for Data-Prep Wrap-Up

Data prep for this stage is considered complete when:
1. Phase 4a-4d run successfully on target corpus slice
2. Validation reports no critical integrity violations
3. Graph is traceable to grounded spans for auditability
4. Outputs are stable enough for downstream analysis and model-building

---

## Immediate Implementation Plan

1. Create Phase 4 scripts and config files in maintained pipeline
2. Add Phase 4 steps to orchestrator and workflow docs (as opt-in)
3. Run Phase 4 on a bounded subset for first quality pass
4. Iterate normalization rules and validation gates
5. Freeze workflow and proceed to downstream graph analytics/modeling

---

## Progress Update (2026-03-01 to 2026-03-03)

### Implementation Completed

Implemented maintained Phase 4 scripts and configs:

- `tests/inference_test/pipeline/phase4a_extract_fab_entities.py`
- `tests/inference_test/pipeline/phase4b_normalize_fab_entities.py`
- `tests/inference_test/pipeline/phase4c_build_fab_process_graph.py`
- `tests/inference_test/pipeline/phase4d_validate_fab_integrity.py`
- `tests/inference_test/pipeline/config/fab_entity_schema.json`
- `tests/inference_test/pipeline/config/fab_normalization_rules.json`

Pipeline integration/docs:
- Added phase4 steps to `tests/inference_test/pipeline/run_pipeline.py` (kept default `--to-step` at `phase3e` to avoid changing routine runs)
- Updated workflow doc: `tests/inference_test/README.txt`

### Pilot Runs and Fixes

Initial pilot on 20 docs surfaced two categories of issues:
1. Inference configuration mismatch (endpoint credentials/provider loading path)
2. Ungrounded extraction records (null char spans), causing Phase 4d integrity failures

Fixes applied:
- Added `.env`-based preflight endpoint test in Phase 4a before extraction
- Added provider/endpoint fallback wiring for current project inference endpoint
- Added strict grounding behavior in Phase 4a:
  - class allowlist from schema
  - drop ungrounded records by default (`--allow-ungrounded` opt-out)
- Added per-document retry controls and deterministic temperature option

### Latest Bounded Pilot (20 docs)

Phase 4a command shape used (bounded):
- `--max-docs 20 --reset-output --extraction-passes 1 --max-workers 1 --max-char-buffer 700 --max-attempts-per-doc 1`

Results:
- Phase 4a:
  - processed: 16
  - failures: 4
  - total extractions: 1029
  - preflight: `ok`
- Phase 4b:
  - total documents: 16
  - total entities: 1029
  - missing provenance rows: 0
- Phase 4c:
  - node count: 881
  - edge count: 1229
- Phase 4d:
  - violations: 0 (integrity pass)

Key outputs produced:
- `tests/inference_test/output/phase4a_fab_entities_raw.jsonl`
- `tests/inference_test/output/phase4a_fab_entities_summary.json`
- `tests/inference_test/output/phase4b_fab_entities_normalized.jsonl`
- `tests/inference_test/output/phase4b_fab_entities_summary.json`
- `tests/inference_test/output/phase4c_fab_process_graph.json`
- `tests/inference_test/output/phase4c_fab_process_graph_summary.json`
- `tests/inference_test/output/phase4d_fab_validation_summary.json`

### Current Status

- Phase 4 pipeline is now implemented and runnable.
- Bounded pilot reaches validation pass with grounded provenance.
- Next step is scaling the run from bounded sample to full marker corpus and then tightening precision on noisy classes.
