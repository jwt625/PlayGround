"""Build re-score batches cache/scoring/batch_rNNN.jsonl for kept pool rows that are either unscored or
were scored on a different abstract than the pool now holds (OpenAlex placeholder/mismatch replaced by the
Xplore backfill). Rows still lacking a full abstract are skipped unless --include-truncated.
Outputs out_rNNN.jsonl sort after out_0NN.jsonl, so their scores override.

Usage: uv run python scripts/make_rescore.py [--size 100] [--include-truncated] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402
from make_batches import FIELDS  # noqa: E402
from prefilter import clean_text  # noqa: E402

D = ROOT / "cache" / "scoring"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--include-truncated", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    pool = [json.loads(ln) for ln in (ROOT / "cache" / "pool.jsonl").read_text().splitlines()]
    batched: dict[str, str] = {}  # uid -> abstract it was (or will be) scored on
    for p in sorted(D.glob("batch_*.jsonl")):
        for ln in p.read_text().splitlines():
            r = json.loads(ln)
            batched[r["uid"]] = r.get("abstract") or ""
    todo, reasons = [], {"unscored": 0, "abstract_changed": 0}
    for r in pool:
        if not r["prefilter_keep"]:
            continue
        if r["abstract_source"] == "xplore_truncated" and not args.include_truncated:
            continue
        if r["uid"] not in batched:
            todo.append(r); reasons["unscored"] += 1
        elif clean_text(batched[r["uid"]]) != clean_text(r["abstract"]):
            todo.append(r); reasons["abstract_changed"] += 1
    start = len(list(D.glob("batch_r*.jsonl")))
    n = (len(todo) + args.size - 1) // args.size
    print(f"{len(todo)} rows {reasons} -> {n} batches from r{start:03d}")
    if args.dry_run:
        return
    for i in range(n):
        with (D / f"batch_r{start + i:03d}.jsonl").open("w") as f:
            for r in todo[i * args.size:(i + 1) * args.size]:
                f.write(json.dumps({k: r.get(k) for k in FIELDS}) + "\n")


if __name__ == "__main__":
    main()
