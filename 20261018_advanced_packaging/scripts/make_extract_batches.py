"""Create extraction batches from cached papers (text.md present) not yet in any batch or in data/papers.csv.

Batch files: data/_staging/batches/b<NNN>.csv (+ staging dir with templates). Order: cache_list priority, then
round-robin over buckets. Usage: uv run python scripts/make_extract_batches.py [--size 3] [--max-batches N] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data" / "_staging" / "batches"
COLS = ["paper_id", "doi", "title", "year", "venue", "primary_bucket", "buckets", "priority", "triage_rationale"]


def pid_of(doi: str) -> str:
    return doi.lower().split("/", 1)[1].replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader((ROOT / "outputs" / "cache_list.csv").open()))
    assigned = set()
    for p in B.glob("*.csv"):
        assigned |= {r["paper_id"] for r in csv.DictReader(p.open())}
    dp = ROOT / "data" / "papers.csv"
    if dp.exists():
        assigned |= {r["paper_id"] for r in csv.DictReader(dp.open())}
    ready = [r for r in rows if r["doi"] and (ROOT / "references" / pid_of(r["doi"]) / "text.md").exists()
             and pid_of(r["doi"]) not in assigned]
    ready.sort(key=lambda r: int(r["priority"]))
    by_b: dict[str, list] = {}
    for r in ready:
        by_b.setdefault(r["primary_bucket"], []).append(r)
    ordered = []
    while any(by_b.values()):
        for b in sorted(by_b):
            if by_b[b]:
                ordered.append(by_b[b].pop(0))
    n0 = len([p for p in B.glob("b*.csv")])
    chunks = [ordered[i:i + a.size] for i in range(0, len(ordered), a.size)]
    if a.max_batches:
        chunks = chunks[: a.max_batches]
    print(f"{len(ready)} ready papers -> {len(chunks)} batches from b{n0:03d}")
    if a.dry_run:
        return 0
    for k, ch in enumerate(chunks):
        name = f"b{n0 + k:03d}"
        with (B / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in ch:
                w.writerow([pid_of(r["doi"]), r["doi"], r["title"], r["year"], r["venue"], r["primary_bucket"],
                            r["buckets"], r["priority"], r["rationale"]])
        st = ROOT / "data" / "_staging" / name
        (st / "evidence").mkdir(parents=True, exist_ok=True)
        for t in (ROOT / "data" / "_staging" / "TEMPLATE").glob("*.csv"):
            if not (st / t.name).exists():
                shutil.copyfile(t, st / t.name)
        print(name, [pid_of(r["doi"]) for r in ch])
    return 0


if __name__ == "__main__":
    sys.exit(main())
