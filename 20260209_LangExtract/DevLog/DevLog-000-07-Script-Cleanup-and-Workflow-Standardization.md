# DevLog-000-07: Script Cleanup and Workflow Standardization

## Date
2026-03-01

## Objective

Clean up and organize the inference scripts accumulated during R1-R3 work so the maintained workflow is clear, reproducible, and easier to operate.

Goals:
1. Separate maintained pipeline scripts from debug and historical one-off scripts
2. Rename core scripts with phase-based naming
3. Fix path/import assumptions after reorganization
4. Add a single orchestrator entrypoint and one workflow document

---

## Why This Cleanup Was Needed

Before cleanup:
- `tests/inference_test/` mixed active pipeline scripts, throwaway diagnostics, and historical round-specific scripts
- Several scripts had `test_*.py` names but were not tests
- Script names did not clearly indicate phase role
- Some scripts depended on relative folder assumptions that became fragile

Impact:
- Hard to identify which scripts are authoritative for the current pipeline
- Higher risk of running outdated or round-specific scripts by mistake
- Harder onboarding and handoff

---

## Changes Made

### 1) Folder Structure Standardization

`tests/inference_test/` is now organized into:

- `pipeline/`: maintained, authoritative workflow scripts
- `debug/`: manual probes and diagnostics
- `legacy/`: historical scripts retained for reproducibility
  - `legacy/r2_collection/`
  - `legacy/ofs/`
  - `legacy/evaluation/`
  - `legacy/one_off_fixes/`
- `output/`: generated artifacts

---

### 2) Core Pipeline Renaming

Active scripts were renamed to phase-based names:

- `extract_reference_sections.py` -> `pipeline/phase1_extract_reference_sections.py`
- `batch_extract_refs_phase2.py` -> `pipeline/phase2_extract_structured_refs.py`
- `build_citation_graph.py` -> `pipeline/phase3_build_citation_graph.py`
- `build_canonical_reference_registry.py` -> `pipeline/phase3_build_canonical_registry.py`
- `enrich_relaxed_refs_doi.py` -> `pipeline/phase3_enrich_registry_doi.py`
- `generate_true_external_queue.py` -> `pipeline/phase3_generate_true_external_queue.py`
- `validate_citation_integrity.py` -> `pipeline/phase3_validate_integrity.py`

---

### 3) Debug Script Clarification

Manual scripts were renamed and moved to avoid confusion with real tests:

- `test_extract_references_llm.py` -> `debug/debug_extract_references_llm.py`
- `test_paper_extraction.py` -> `debug/debug_extract_papers_from_blog.py`

These are no longer presented as pipeline dependencies.

---

### 4) Legacy Script Archiving

Historical scripts moved under `legacy/`:

- R2 collection scripts
- OFS extraction/audit scripts
- Early evaluation/benchmark scripts
- One-off repair script(s)

This keeps history available while reducing operational noise.

---

### 5) Path and Import Fixes

After moving files, script roots and output paths were updated to stay project-root stable.

Notable hardening:
- Removed phase2 dependency on debug module import by inlining tool schema and prompt in `phase2_extract_structured_refs.py`
- Updated absolute/relative root calculations for moved pipeline and legacy scripts
- Updated help text/examples to point to new script paths

---

### 6) Single Orchestrator Added

Added:
- `tests/inference_test/pipeline/run_pipeline.py`

Features:
- Runs maintained pipeline end-to-end
- Supports bounded execution via `--from-step` and `--to-step`
- Supports preview mode via `--dry-run`

---

### 7) Single Workflow Document Added

Added:
- `tests/inference_test/README.txt`

Contains:
- folder intent (`pipeline/debug/legacy/output`)
- authoritative step-by-step workflow mapping
- single-command pipeline entrypoint
- notes on what is maintained vs historical

Note:
- `README.txt` is used instead of markdown because this repo ignores `*.md`.

---

## Validation

Validation performed after cleanup:
- Python syntax compile for scripts (excluding `.venv`) passed
- `run_pipeline.py --help` works
- `run_pipeline.py --dry-run --from-step phase3 --to-step phase3e` works
- Pipeline script help commands resolve with new locations

---

## Current State

- The maintained workflow is now explicit and phase-scoped in `tests/inference_test/pipeline/`
- Debug and historical scripts are separated from operational scripts
- A single workflow doc and runner are available for reproducible execution

---

## Next Steps

1. Keep all new pipeline additions under `tests/inference_test/pipeline/` only
2. Treat `legacy/` as read-only unless backporting a fix
3. Use `pipeline/run_pipeline.py` for routine runs and reproducibility
4. If needed, add a lightweight CI job that runs `phase3_validate_integrity.py` on updated artifacts

