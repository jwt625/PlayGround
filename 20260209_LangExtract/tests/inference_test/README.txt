Inference Scripts: Workflow and Organization
=============================================

This folder is split into:

- pipeline/: maintained scripts for the citation/registry workflow
- debug/: manual probes and diagnostics
- legacy/: historical scripts kept for reference only
- output/: generated artifacts


Current Authoritative Pipeline
------------------------------

Run from project root: 20260209_LangExtract

Quick start (single command):

  python tests/inference_test/pipeline/run_pipeline.py

Useful variants:

  python tests/inference_test/pipeline/run_pipeline.py --dry-run
  python tests/inference_test/pipeline/run_pipeline.py --from-step phase3
  python tests/inference_test/pipeline/run_pipeline.py --to-step phase3d


Step-by-step mapping (script -> purpose)
----------------------------------------

1) phase1
  script: tests/inference_test/pipeline/phase1_extract_reference_sections.py
  purpose: extract bibliography/reference sections from marker markdown
  main output: tests/inference_test/output/reference_sections*.jsonl

2) phase2
  script: tests/inference_test/pipeline/phase2_extract_structured_refs.py
  purpose: parse references into structured JSON via LLM function-calling
  main output: tests/inference_test/output/phase2_extracted_refs*.jsonl

3) phase3
  script: tests/inference_test/pipeline/phase3_build_citation_graph.py
  purpose: deduplicate references, build graph, compute stats
  main output:
    - tests/inference_test/output/unique_references.jsonl
    - tests/inference_test/output/citation_graph.json
    - tests/inference_test/output/dedup_stats.json

4) phase3b
  script: tests/inference_test/pipeline/phase3_build_canonical_registry.py
  purpose: build canonical cross-round registry with strict/relaxed provenance
  main output:
    - tests/inference_test/output/canonical_reference_registry.jsonl
    - tests/inference_test/output/canonical_reference_registry_summary.json

5) phase3c
  script: tests/inference_test/pipeline/phase3_enrich_registry_doi.py
  purpose: enrich relaxed-only entries with DOI candidates (Crossref)
  main output:
    - tests/inference_test/output/relaxed_only_doi_candidates.jsonl
    - tests/inference_test/output/canonical_reference_registry_enriched.jsonl

6) phase3d
  script: tests/inference_test/pipeline/phase3_generate_true_external_queue.py
  purpose: create clean true-external processing queue from effective labels
  main output:
    - tests/inference_test/output/top_external_refs_for_r3_processing.md
    - tests/inference_test/output/r3_processing_input.json
    - tests/inference_test/output/true_external_queue_validation.json

7) phase3e
  script: tests/inference_test/pipeline/phase3_validate_integrity.py
  purpose: regression checks (ID alignment, queue invariants, effective top-list)
  pass condition: violations == 0


Debug Scripts
-------------

- tests/inference_test/debug/debug_extract_references_llm.py
- tests/inference_test/debug/debug_extract_papers_from_blog.py

These are for manual inspection and should not be used as pipeline dependencies.


Legacy Scripts
--------------

- tests/inference_test/legacy/r2_collection/*
- tests/inference_test/legacy/ofs/*
- tests/inference_test/legacy/evaluation/*
- tests/inference_test/legacy/one_off_fixes/*

These are retained for reproducibility/history and are not part of the maintained workflow.

