"""Merge a validated staging batch into data/ (coordinator only). Dry run by default.

Conflicts: a staged key already in data/<table>.csv (re-distilling a paper requires --replace, which drops all
existing rows of the staged paper_ids first). Validation must report 0 errors before --apply writes anything.

Usage: uv run python scripts/merge_staging.py data/_staging/<batch> [--apply] [--replace]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_db import ROOT, TABLES, read, validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    st = ROOT / a.staging
    data = ROOT / "data"
    for table, ts in TABLES.items():  # make sure data/ tables exist
        p = data / f"{table}.csv"
        if not p.exists():
            p.write_text(",".join(ts["columns"]) + "\n")
    errs, warns = validate(st, base=data)
    staged_papers = {r["paper_id"] for r in read(st, "papers")[1]}
    existing_tv = {r["tv_id"]: r["paper_id"] for r in read(data, "test_vehicles")[1]}

    def paper_of(table: str, r: dict) -> str:
        if "paper_id" in r and r["paper_id"]:
            return r["paper_id"]
        return existing_tv.get(r.get("tv_id", ""), "") or next(
            (t["paper_id"] for t in read(st, "test_vehicles")[1] if t["tv_id"] == r.get("tv_id")), "")

    conflicts = []
    for table, ts in TABLES.items():
        _, base_rows = read(data, table)
        base_keys = {r[ts["key"]] for r in base_rows if not (a.replace and paper_of(table, r) in staged_papers)}
        for r in read(st, table)[1]:
            if r[ts["key"]] in base_keys:
                conflicts.append(f"{table}: {r[ts['key']]} already in data/")
    for c in conflicts:
        print("CONFLICT", c)
    for e in errs:
        print("ERROR", e)
    print(f"{len(errs)} validation errors, {len(warns)} warnings, {len(conflicts)} conflicts; papers: {sorted(staged_papers)}")
    if errs or conflicts or not a.apply:
        return 1 if (errs or conflicts) else 0
    for table, ts in TABLES.items():
        hdr, base_rows = read(data, table)
        if a.replace:
            base_rows = [r for r in base_rows if paper_of(table, r) not in staged_papers]
        new = read(st, table)[1]
        with (data / f"{table}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ts["columns"]))
            w.writeheader()
            for r in base_rows + new:
                w.writerow({k: r.get(k, "") for k in ts["columns"]})
    (data / "evidence").mkdir(exist_ok=True)
    for ev in (st / "evidence").glob("*.yaml"):
        shutil.copyfile(ev, data / "evidence" / ev.name)
    print(f"applied {a.staging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
