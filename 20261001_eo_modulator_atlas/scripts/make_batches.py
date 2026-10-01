"""Split data/candidates.csv into distillation batches and a manual-download list.

Usage: uv run python scripts/make_batches.py --max-priority 1 [--size 5] [--tag p1]
Writes data/_staging/batches/<tag>_NN.csv (tracked; contains only candidate metadata and the `@corpus_*` alias placeholders,
never absolute local paths; scripts/prefetch_batch.py resolves aliases through the git-ignored private mapping) and
appends to data/manual_downloads.md (tracked) with paywalled papers that have no local copy and are not cached.
Adds a `source_state` column: cached (references/<id>/source.pdf exists) | local_alias | open_fetch.
Skips paper_ids already in data/papers.csv or already in an existing batch file, unless --rebuild (rewrites all <tag>_* files).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "data" / "_staging"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-priority", type=int, default=1)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--rebuild", action="store_true", help="delete and regenerate all <tag>_NN.csv batch files")
    a = ap.parse_args()
    cands = list(csv.DictReader((ROOT / "data/candidates.csv").open(newline="")))
    done = {r["paper_id"] for r in csv.DictReader((ROOT / "data/papers.csv").open(newline=""))}
    bdir = STAGING / "batches"
    bdir.mkdir(exist_ok=True)
    queued: set[str] = set()
    if a.rebuild:
        for old in bdir.glob(f"{a.tag}_*.csv"):
            old.unlink()
    for batch_file in bdir.glob("*.csv"):
        queued |= {r["paper_id"] for r in csv.DictReader(batch_file.open(newline=""))}
    private_ids: set[str] = set()
    pp = STAGING / "candidates_seed_local_paths_private.csv"
    if pp.exists():
        private_ids = {r["paper_id"] for r in csv.DictReader(pp.open(newline=""))}
    accessible: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    for r in cands:
        if int(r["priority"] or 9) > a.max_priority or r["paper_id"] in done or r["paper_id"] in queued:
            continue
        cached = (ROOT / "references" / r["paper_id"] / "source.pdf").exists()
        alias = r["local_source_path"].startswith("@") and r["paper_id"] in private_ids
        # absolute paths from any producer must never reach tracked batch files
        if r["local_source_path"].startswith("/"):
            r["local_source_path"] = "@local_corpus" if r["paper_id"] in private_ids else ""
        r["source_state"] = "cached" if cached else ("local_alias" if alias else "open_fetch")
        if cached or alias or r["access_guess"] in ("open_access", "arxiv"):
            accessible.append(r)
        else:
            manual.append(r)
    accessible.sort(key=lambda r: (r["platform_guess"], r["year"], r["paper_id"]))
    header = list(accessible[0].keys()) if accessible else []
    start = len(list(bdir.glob(f"{a.tag}_*.csv")))
    for i in range(0, len(accessible), a.size):
        chunk = accessible[i : i + a.size]
        with (bdir / f"{a.tag}_{start + i // a.size + 1:02d}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(chunk)
    md = ROOT / "data" / "manual_downloads.md"
    existing = (
        md.read_text()
        if md.exists()
        else "# Manual downloads (paywalled or not accessible)\n\nDrop each file into references/_inbox/ using the exact save_as filename.\n\n"
    )
    add = ""
    for r in manual:
        if f"paper_id: {r['paper_id']}\n" in existing:
            continue
        add += (
            f"- paper_id: {r['paper_id']}\n  title: {r['title']}\n  doi: {r['doi']}\n  publisher_url: {r['url']}\n"
            f"  save_as: {r['paper_id']}.pdf\n  drop_folder: references/_inbox/\n  priority: {r['priority']}\n"
            f"  why_needed: reports Vpi/bandwidth/loss metrics per abstract (not yet read)\n"
        )
    md.write_text(existing + add)
    print(f"accessible={len(accessible)} in {-(-len(accessible) // a.size)} batches; manual_added={len(manual)}")


if __name__ == "__main__":
    main()
