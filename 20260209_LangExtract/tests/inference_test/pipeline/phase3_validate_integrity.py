#!/usr/bin/env python3
"""Regression checks for citation graph and canonical registry integrity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def ndoi(v):
    if not v:
        return None
    s = str(v).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or None


def narxiv(v):
    if not v:
        return None
    s = str(v).strip().lower()
    s = re.sub(r"^arxiv:\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate citation data integrity")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("tests/inference_test/output/canonical_reference_registry_enriched.jsonl"),
    )
    parser.add_argument(
        "--dedup-stats",
        type=Path,
        default=Path("tests/inference_test/output/dedup_stats.json"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("tests/inference_test/output/r3_processing_input.json"),
    )
    args = parser.parse_args()

    rows = []
    by_ref = {}
    for line in args.registry.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append(r)
        by_ref[r.get("ref_id")] = r

    violations = []

    # 1) Strict ID alignment checks.
    for r in rows:
        ref_id = r.get("ref_id") or ""
        c = r.get("canonical_enriched") or r.get("canonical") or {}
        if ref_id.startswith("doi:"):
            expected = ref_id[4:]
            if ndoi(c.get("doi")) != expected:
                violations.append({"type": "doi_mismatch", "ref_id": ref_id, "doi": c.get("doi")})
        if ref_id.startswith("arxiv:"):
            expected = ref_id[6:]
            if narxiv(c.get("arxiv_id")) != expected:
                violations.append({"type": "arxiv_mismatch", "ref_id": ref_id, "arxiv_id": c.get("arxiv_id")})

    # 2) Queue entries must remain truly external (effective).
    queue = json.loads(args.queue.read_text())
    for q in queue:
        rid = q.get("ref_id")
        r = by_ref.get(rid)
        if not r:
            violations.append({"type": "queue_ref_missing_in_registry", "ref_id": rid})
            continue
        if not bool(r.get("external_effective")):
            violations.append({"type": "queue_not_external_effective", "ref_id": rid})
        if bool(r.get("seen_in_any_round")) or bool(r.get("downloaded_any_round")):
            violations.append(
                {
                    "type": "queue_seen_or_downloaded",
                    "ref_id": rid,
                    "seen_in_any_round": r.get("seen_in_any_round"),
                    "downloaded_any_round": r.get("downloaded_any_round"),
                }
            )

    # 3) top_100 effective list should not contain already-seen references.
    dedup = json.loads(args.dedup_stats.read_text())
    for x in dedup.get("top_100_external_effective", []):
        if x.get("seen_in_manifest"):
            violations.append({"type": "top100_effective_seen_in_manifest", "ref_id": x.get("ref_id")})

    summary = {
        "registry_rows": len(rows),
        "queue_rows": len(queue),
        "violations": len(violations),
    }
    if violations:
        summary["sample_violations"] = violations[:50]
    print(json.dumps(summary, indent=2))

    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())

