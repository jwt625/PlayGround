#!/usr/bin/env python3
"""Phase 4b: Normalize fabrication entities and assign canonical IDs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4a_fab_entities_raw.jsonl"
DEFAULT_RULES_FILE = PROJECT_ROOT / "tests" / "inference_test" / "pipeline" / "config" / "fab_normalization_rules.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4b_fab_entities_normalized.jsonl"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4b_fab_entities_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4b: Normalize fabrication entities")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_FILE)
    return parser.parse_args()


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def normalize_text(entity_class: str, text: str, rules: dict[str, Any]) -> str:
    key = text.strip().lower()
    if entity_class == "equipment":
        return rules.get("equipment_aliases", {}).get(key, key)
    if entity_class == "material":
        return rules.get("material_aliases", {}).get(key, key)
    if entity_class in {"process_step", "recipe"}:
        return rules.get("process_aliases", {}).get(key, key)
    return key


def load_rules(fp: Path) -> dict[str, Any]:
    if not fp.exists():
        return {}
    return json.loads(fp.read_text())


def iter_raw_rows(fp: Path):
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    args = parse_args()

    rules = load_rules(args.rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_entities = 0
    missing_provenance = 0
    by_class: dict[str, int] = {}
    canonical_counts: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as out_f:
        for doc in iter_raw_rows(args.input):
            if doc.get("status") != "ok":
                continue
            total_docs += 1
            doc_id = doc.get("document_id")
            for ext in doc.get("extractions", []):
                entity_class = (ext.get("extraction_class") or "unknown").strip()
                raw_text = (ext.get("extraction_text") or "").strip()
                if not raw_text:
                    continue

                normalized_text = normalize_text(entity_class, raw_text, rules)
                canonical_id = f"{entity_class}:{slugify(normalized_text)}"
                attrs = ext.get("attributes") if isinstance(ext.get("attributes"), dict) else {}

                row = {
                    "document_id": doc_id,
                    "source_file": doc.get("source_file"),
                    "entity_class": entity_class,
                    "extraction_text": raw_text,
                    "normalized_text": normalized_text,
                    "canonical_id": canonical_id,
                    "attributes": attrs,
                    "source_text": ext.get("source_text"),
                    "char_start": ext.get("char_start"),
                    "char_end": ext.get("char_end"),
                    "section_hint": ext.get("section_hint", "unknown"),
                }

                if (
                    not row["source_text"]
                    or row["char_start"] is None
                    or row["char_end"] is None
                ):
                    missing_provenance += 1

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_entities += 1
                by_class[entity_class] = by_class.get(entity_class, 0) + 1
                canonical_counts[canonical_id] = canonical_counts.get(canonical_id, 0) + 1

    top_canonical = sorted(
        [{"canonical_id": k, "count": v} for k, v in canonical_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:200]

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_documents": total_docs,
        "total_entities": total_entities,
        "missing_provenance_rows": missing_provenance,
        "by_class": by_class,
        "top_canonical": top_canonical,
    }
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
