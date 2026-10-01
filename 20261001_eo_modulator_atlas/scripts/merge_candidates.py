"""Merge staged candidate lists into data/candidates.csv: dedupe by DOI / arXiv id / normalized title, resolve id collisions.

Usage: uv run python scripts/merge_candidates.py data/_staging/candidates_seed.csv data/_staging/candidates_landmark.csv
Existing rows in data/candidates.csv are kept (first writer wins); tags (discovered_via) are unioned; collisions of paper_id
between different papers get a letter suffix on the later row. Prints a summary and every rename.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "candidates.csv"


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def norm_doi(d: str) -> str:
    return d.lower().replace("https://doi.org/", "").strip()


def main(paths: list[str]) -> None:
    rows: list[dict[str, str]] = []
    header: list[str] = []
    if OUT.exists():
        with OUT.open(newline="") as f:
            r = csv.DictReader(f)
            header = list(r.fieldnames or [])
            rows = list(r)
    for p in paths:
        with Path(p).open(newline="") as f:
            r = csv.DictReader(f)
            for h in r.fieldnames or []:
                if h not in header:
                    header.append(h)
            for row in r:
                key_doi, key_ax, key_t = norm_doi(row.get("doi", "")), row.get("arxiv_id", "").strip(), norm_title(row["title"])
                match = None
                for ex in rows:
                    if (
                        (key_doi and norm_doi(ex.get("doi", "")) == key_doi)
                        or (key_ax and ex.get("arxiv_id", "").strip() == key_ax)
                        or (key_t and norm_title(ex["title"]) == key_t)
                    ):
                        match = ex
                        break
                if match is not None:
                    tags = sorted(set(filter(None, match["discovered_via"].split(";") + row["discovered_via"].split(";"))))
                    match["discovered_via"] = ";".join(tags)
                    for k in ("doi", "arxiv_id", "license", "published_on"):
                        if not match.get(k) and row.get(k):
                            match[k] = row[k]
                    match["notes"] = (match.get("notes", "") + " | also in " + Path(p).stem).strip(" |")
                    print(f"dup: {row['paper_id']} -> {match['paper_id']}")
                    continue
                ids = {x["paper_id"] for x in rows}
                pid = row["paper_id"]
                if pid in ids:
                    for suf in "abcdefghij":
                        if f"{pid}{suf}" not in ids:
                            print(f"rename: {pid} -> {pid}{suf} ({row['title'][:60]})")
                            row["notes"] = (row.get("notes", "") + f" | id renamed from {pid}").strip(" |")
                            row["paper_id"] = f"{pid}{suf}"
                            break
                rows.append(row)
    for row in rows:
        for h in header:
            row.setdefault(h, "")
    rows.sort(key=lambda x: (x.get("priority", "9"), x["paper_id"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} candidates -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
