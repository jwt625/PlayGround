#!/usr/bin/env python3
"""Run the maintained citation pipeline end-to-end.

Usage examples:
  python tests/inference_test/pipeline/run_pipeline.py
  python tests/inference_test/pipeline/run_pipeline.py --from-step phase3b
  python tests/inference_test/pipeline/run_pipeline.py --to-step phase3d
  python tests/inference_test/pipeline/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable


def cmd_phase1() -> list[str]:
    return [PY, "tests/inference_test/pipeline/phase1_extract_reference_sections.py"]


def cmd_phase2() -> list[str]:
    return [PY, "tests/inference_test/pipeline/phase2_extract_structured_refs.py"]


def cmd_phase3() -> list[str]:
    return [
        PY,
        "tests/inference_test/pipeline/phase3_build_citation_graph.py",
        "-i",
        "tests/inference_test/output/phase2_extracted_refs.jsonl",
        "-i",
        "tests/inference_test/output/phase2_extracted_refs_r2.jsonl",
        "-i",
        "tests/inference_test/output/phase2_extracted_refs_r3.jsonl",
        "-m",
        "data/collection_r1/metadata",
        "-c",
        "data/collection_r1/manifest_documents.jsonl",
        "-c",
        "data/collection_r2/manifest_documents_r2.jsonl",
        "-c",
        "semiconductor_processing_dataset/raw_documents_R3/manifest_documents_r3.jsonl",
        "-c",
        "semiconductor_processing_dataset/raw_documents_R3_new/manifest_documents_r3new.jsonl",
        "-o",
        "tests/inference_test/output",
    ]


def cmd_phase3b() -> list[str]:
    return [PY, "tests/inference_test/pipeline/phase3_build_canonical_registry.py"]


def cmd_phase3c() -> list[str]:
    return [PY, "tests/inference_test/pipeline/phase3_enrich_registry_doi.py"]


def cmd_phase3d() -> list[str]:
    return [
        PY,
        "tests/inference_test/pipeline/phase3_generate_true_external_queue.py",
        "--input",
        "tests/inference_test/output/canonical_reference_registry_enriched.jsonl",
        "--top-n",
        "100",
    ]


def cmd_phase3e() -> list[str]:
    return [PY, "tests/inference_test/pipeline/phase3_validate_integrity.py"]


STEPS = [
    ("phase1", "Extract reference sections", cmd_phase1),
    ("phase2", "Extract structured references", cmd_phase2),
    ("phase3", "Build citation graph", cmd_phase3),
    ("phase3b", "Build canonical registry", cmd_phase3b),
    ("phase3c", "Enrich DOI coverage", cmd_phase3c),
    ("phase3d", "Generate true-external queue", cmd_phase3d),
    ("phase3e", "Validate integrity", cmd_phase3e),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run maintained citation pipeline")
    parser.add_argument("--from-step", choices=[s[0] for s in STEPS], default=STEPS[0][0])
    parser.add_argument("--to-step", choices=[s[0] for s in STEPS], default=STEPS[-1][0])
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    keys = [s[0] for s in STEPS]
    i0 = keys.index(args.from_step)
    i1 = keys.index(args.to_step)
    if i0 > i1:
        raise SystemExit("--from-step must be before or equal to --to-step")

    selected = STEPS[i0 : i1 + 1]
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Running steps: {[s[0] for s in selected]}")

    for key, desc, fn in selected:
        cmd = fn()
        print(f"\n[{key}] {desc}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

