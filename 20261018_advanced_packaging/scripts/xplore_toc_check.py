"""Completeness check: compare search-kept counts against each proceedings/issue TOC totalRecords,
and backfill missing records from the TOC endpoint.

Writes cache/xplore_toc_check.json; appends backfilled records to cache/xplore/<venue>_<year>.jsonl
(rewrites the trailing _done line with toc totals). Resumable via the toc_checked flag in the _done line.
Usage: uv run python scripts/xplore_toc_check.py [--venues X,Y] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT, AntiBotError, xplore_session  # noqa: E402
from xplore_fetch import slim  # noqa: E402


def toc_all(x, pn: str, isn: str) -> tuple[int, list[dict]]:
    recs, page, total_pages, total = [], 1, 1, 0
    while page <= total_pages:
        r = x.post(f"/rest/search/pub/{pn}/issue/{isn}/toc",
                   {"punumber": pn, "isnumber": isn, "sortType": "vol-only-seq",
                    "pageNumber": str(page), "rowsPerPage": "100"})
        total = int(r.get("totalRecords") or 0)
        total_pages = int(r.get("totalPages") or 1)
        recs += r.get("records") or []
        page += 1
    return total, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sel = set(args.venues.split(",")) if args.venues else None
    files = []
    for p in sorted((ROOT / "cache" / "xplore").glob("*.jsonl")):
        venue, year = p.stem.rsplit("_", 1)
        if sel and venue not in sel:
            continue
        lines = p.read_text().strip().splitlines()
        done = json.loads(lines[-1])
        if done.get("_done") and not done.get("toc_checked") and done.get("pubs"):
            files.append((p, venue, int(year), lines, done))
    print(f"{len(files)} files to check, {sum(len(f[4]['pubs']) for f in files)} TOCs")
    if args.dry_run or not files:
        return
    report_p = ROOT / "cache" / "xplore_toc_check.json"
    report = json.loads(report_p.read_text()) if report_p.exists() else {}
    with xplore_session("xplore_toc_check") as x:
        for p, venue, year, lines, done in files:
            recs = [json.loads(ln) for ln in lines[:-1]]
            have = {r["articleNumber"] for r in recs}
            added, tocs = [], {}
            try:
                for key in done["pubs"]:
                    pn, isn = key.split("/")
                    total, trecs = toc_all(x, pn, isn)
                    new = [slim(t) for t in trecs if str(t.get("articleNumber")) not in have]
                    for n in new:
                        have.add(n["articleNumber"])
                    added += new
                    tocs[key] = {"toc_total": total, "search_kept": done["pubs"][key]["n"], "added": len(new)}
            except AntiBotError as e:
                print(f"STOP: {e}")
                break
            with p.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                for r in added:
                    f.write(json.dumps({"venue": venue, "year": year, **r, "via": "toc"}) + "\n")
                done.update(toc_checked=True, n=len(recs) + len(added), tocs=tocs)
                f.write(json.dumps(done) + "\n")
            report[p.stem] = tocs
            report_p.write_text(json.dumps(report, indent=1))
            print(f"{p.stem}: +{len(added)} from TOC; {tocs if len(tocs) < 4 else str(len(tocs)) + ' issues'}")
        print(f"requests: {x.n_requests}")


if __name__ == "__main__":
    main()
