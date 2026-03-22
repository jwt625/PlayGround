#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard extraction indexes.")
    parser.add_argument("--index-glob", default="tmp/ofc2026_shards/paper_text_index.shard*.json")
    parser.add_argument("--output-index", default="paper_text_index.json")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    merged: dict[str, dict] = {}
    for index_path in sorted(Path(".").resolve().glob(args.index_glob)):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for pdf_name, record in payload.get("papers", {}).items():
            merged[pdf_name] = record

    output_path = Path(args.output_index).resolve()
    output_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "paper_count": len(merged),
                "papers": dict(sorted(merged.items())),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"merged_papers": len(merged), "output_index": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
