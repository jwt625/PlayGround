#!/usr/bin/env python3
"""Phase 4c: Build fabrication process graph from normalized entities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4b_fab_entities_normalized.jsonl"
DEFAULT_GRAPH_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4c_fab_process_graph.json"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "tests" / "inference_test" / "output" / "phase4c_fab_process_graph_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4c: Build fabrication process graph")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_FILE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_FILE)
    return parser.parse_args()


def iter_rows(fp: Path):
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def main() -> int:
    args = parse_args()

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    def ensure_node(node_id: str, node_type: str, label: str) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "label": label,
                "support_count": 0,
                "documents": set(),
            }

    relation_count: dict[str, int] = {}

    def add_edge(edge_type: str, src: str, dst: str, doc_id: str) -> None:
        key = (edge_type, src, dst, doc_id)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"type": edge_type, "from": src, "to": dst, "document_id": doc_id})
        relation_count[edge_type] = relation_count.get(edge_type, 0) + 1

    for row in iter_rows(args.input):
        doc_id = row["document_id"]
        doc_node = f"document:{doc_id}"
        ensure_node(doc_node, "document", doc_id)

        ent_id = row["canonical_id"]
        ent_type = row["entity_class"]
        ent_label = row["normalized_text"]
        ensure_node(ent_id, ent_type, ent_label)

        nodes[ent_id]["support_count"] += 1
        nodes[ent_id]["documents"].add(doc_id)
        nodes[doc_node]["support_count"] += 1

        add_edge("document_mentions", doc_node, ent_id, doc_id)

        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}

        if ent_type in {"process_step", "recipe"}:
            for eq in to_list(attrs.get("equipment")):
                eq_id = f"equipment:{eq.lower().replace(' ', '_')}"
                ensure_node(eq_id, "equipment", eq)
                add_edge("step_uses_equipment", ent_id, eq_id, doc_id)

            for mat in to_list(attrs.get("materials")):
                mat_id = f"material:{mat.lower().replace(' ', '_')}"
                ensure_node(mat_id, "material", mat)
                add_edge("step_uses_material", ent_id, mat_id, doc_id)

            for prm in to_list(attrs.get("parameters")):
                prm_id = f"parameter:{prm.lower().replace(' ', '_')}"
                ensure_node(prm_id, "parameter", prm)
                add_edge("step_has_parameter", ent_id, prm_id, doc_id)

            for site in to_list(attrs.get("site")):
                site_id = f"site:{site.lower().replace(' ', '_')}"
                ensure_node(site_id, "site", site)
                add_edge("step_occurs_at_site", ent_id, site_id, doc_id)

            for nxt in to_list(attrs.get("next_step")):
                nxt_id = f"process_step:{nxt.lower().replace(' ', '_')}"
                ensure_node(nxt_id, "process_step", nxt)
                add_edge("step_precedes_step", ent_id, nxt_id, doc_id)

    node_list = []
    for node in nodes.values():
        node_copy = dict(node)
        node_copy["documents"] = sorted(list(node_copy["documents"]))
        node_list.append(node_copy)

    graph = {
        "nodes": node_list,
        "edges": edges,
    }

    args.graph.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.graph.write_text(json.dumps(graph, indent=2))

    summary = {
        "input": str(args.input),
        "graph": str(args.graph),
        "node_count": len(node_list),
        "edge_count": len(edges),
        "relation_count": relation_count,
    }
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
