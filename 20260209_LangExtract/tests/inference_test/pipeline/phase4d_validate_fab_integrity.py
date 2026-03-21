#!/usr/bin/env python3
"""Phase 4d: Validate integrity for normalized fabrication entities and graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTITIES = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4b_fab_entities_normalized.jsonl"
DEFAULT_GRAPH = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4c_fab_process_graph.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4d_fab_validation_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4d: Validate fabrication integrity")
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def iter_rows(fp: Path):
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    args = parse_args()

    violations = []
    row_count = 0

    for row in iter_rows(args.entities):
        row_count += 1
        required = ["document_id", "entity_class", "extraction_text", "source_text", "section_hint"]
        for key in required:
            if not row.get(key):
                violations.append({"type": "missing_required_field", "field": key, "document_id": row.get("document_id")})

        s = row.get("char_start")
        e = row.get("char_end")
        source_text = row.get("source_text") or ""

        if s is None or e is None:
            violations.append({"type": "missing_char_span", "document_id": row.get("document_id"), "canonical_id": row.get("canonical_id")})
        elif not isinstance(s, int) or not isinstance(e, int) or s < 0 or e < s:
            violations.append({"type": "invalid_char_span", "document_id": row.get("document_id"), "canonical_id": row.get("canonical_id"), "char_start": s, "char_end": e})

        if isinstance(source_text, str) and source_text.strip() == "":
            violations.append({"type": "empty_source_text", "document_id": row.get("document_id"), "canonical_id": row.get("canonical_id")})

    graph = json.loads(args.graph.read_text())
    node_ids = {n["node_id"] for n in graph.get("nodes", []) if "node_id" in n}
    edge_count = 0

    for edge in graph.get("edges", []):
        edge_count += 1
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_ids:
            violations.append({"type": "edge_missing_source_node", "edge": edge})
        if dst not in node_ids:
            violations.append({"type": "edge_missing_target_node", "edge": edge})
        if not edge.get("type"):
            violations.append({"type": "edge_missing_type", "edge": edge})

    summary = {
        "entities_file": str(args.entities),
        "graph_file": str(args.graph),
        "entity_rows": row_count,
        "graph_nodes": len(node_ids),
        "graph_edges": edge_count,
        "violations": len(violations),
        "sample_violations": violations[:100],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
