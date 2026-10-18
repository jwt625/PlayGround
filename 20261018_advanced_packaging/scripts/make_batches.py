"""Split prefilter-kept pool rows into scoring batches: cache/scoring/batch_NNN.jsonl (skip-existing).

Usage: uv run python scripts/make_batches.py [--size 100] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

D = ROOT / "cache" / "scoring"
FIELDS = ["uid", "venue", "year", "committee", "session", "title", "abstract", "first_affiliation", "orgs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-truncated", action="store_true", help="also batch rows with only a truncated Xplore abstract")
    args = ap.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in (ROOT / "cache" / "pool.jsonl").read_text().splitlines()]
    keep = sorted((r for r in rows if r["prefilter_keep"]), key=lambda r: (r["venue"], r["year"], r["uid"]))
    assigned = set()
    for p in D.glob("batch_*.jsonl"):
        assigned |= {json.loads(ln)["uid"] for ln in p.read_text().splitlines()}
    todo = [r for r in keep if r["uid"] not in assigned
            and (args.include_truncated or r["abstract_source"] != "xplore_truncated")]
    start = len(list(D.glob("batch_*.jsonl")))
    n = (len(todo) + args.size - 1) // args.size
    print(f"kept {len(keep)}, already batched {len(assigned)}, new {len(todo)} -> {n} batches from {start:03d}")
    if args.dry_run:
        return
    for i in range(n):
        chunk = todo[i * args.size:(i + 1) * args.size]
        with (D / f"batch_{start + i:03d}.jsonl").open("w") as f:
            for r in chunk:
                f.write(json.dumps({k: r.get(k) for k in FIELDS}) + "\n")


if __name__ == "__main__":
    main()
