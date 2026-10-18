"""Enumerate every Xplore record per venue-year (search by publication title + year range).

Resumable: a venue-year whose JSONL has a trailing {"_done": ...} line is skipped.
Usage: uv run python scripts/xplore_fetch.py [--venues ECTC,TCPMT] [--years 2025] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT, AntiBotError, xplore_session  # noqa: E402

OUT = ROOT / "cache" / "xplore"
KEEP = [
    "articleNumber", "doi", "articleTitle", "authors", "abstract", "accessType", "publicationTitle",
    "publicationNumber", "isNumber", "publicationYear", "publicationDate", "startPage", "endPage",
    "contentType", "displayContentType", "documentLink", "pdfLink", "citationCount", "isEarlyAccess",
]


def out_path(venue: str, year: int) -> Path:
    return OUT / f"{venue}_{year}.jsonl"


def is_done(p: Path) -> bool:
    if not p.exists():
        return False
    lines = p.read_text().strip().splitlines()
    return bool(lines) and '"_done"' in lines[-1]


def slim(rec: dict) -> dict:
    d = {k: rec.get(k) for k in KEEP}
    d["authors"] = [
        {"name": a.get("preferredName") or a.get("normalizedName"), "id": a.get("id")}
        for a in (rec.get("authors") or [])
    ]
    d["accessType"] = (rec.get("accessType") or {}).get("type")
    return d


def fetch_venue_year(x, venue: str, cfg: dict, year: int) -> dict:
    match = re.compile(cfg["match"], re.I)
    exclude = re.compile(cfg["exclude"], re.I) if cfg.get("exclude") else None
    body = {
        "queryText": f'("Publication Title":"{cfg["query"]}")',
        "ranges": [f"{year}_{year}_Year"],
        "rowsPerPage": 100,
        "newsearch": True,
    }
    kept, other_pubs, page, total_pages, total = [], {}, 1, 1, None
    while page <= total_pages:
        r = x.post("/rest/search", {**body, "pageNumber": page})
        total_pages = int(r.get("totalPages") or 1)
        total = r.get("totalRecords")
        for rec in r.get("records") or []:
            title = rec.get("publicationTitle") or ""
            if match.search(title) and not (exclude and exclude.search(title)):
                kept.append(slim(rec))
            else:
                other_pubs[title] = other_pubs.get(title, 0) + 1
        page += 1
    pubs: dict[str, dict] = {}
    for k in kept:
        key = f'{k["publicationNumber"]}/{k["isNumber"]}'
        pubs.setdefault(key, {"title": k["publicationTitle"], "n": 0})["n"] += 1
    return {"records": kept, "search_total": total, "pages": total_pages, "pubs": pubs, "other_pubs": other_pubs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues")
    ap.add_argument("--years")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    venues = yaml.safe_load((ROOT / "config" / "venues.yaml").read_text())["venues"]
    sel_v = set(args.venues.split(",")) if args.venues else set(venues)
    sel_y = {int(y) for y in args.years.split(",")} if args.years else None
    jobs = [
        (v, c, y) for v, c in venues.items() if v in sel_v and c.get("source", "xplore") == "xplore"
        for y in c["years"] if (sel_y is None or y in sel_y) and not is_done(out_path(v, y))
    ]
    print(f"{len(jobs)} venue-years to fetch: {[f'{v}{y}' for v, _, y in jobs]}")
    if args.dry_run or not jobs:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with xplore_session("xplore_fetch") as x:
        for v, c, y in jobs:
            try:
                res = fetch_venue_year(x, v, c, y)
            except AntiBotError as e:
                print(f"STOP: {e}")
                break
            p = out_path(v, y)
            with p.open("w") as f:
                for rec in res["records"]:
                    f.write(json.dumps({"venue": v, "year": y, **rec}) + "\n")
                f.write(json.dumps({"_done": True, "n": len(res["records"]), "search_total": res["search_total"],
                                    "pubs": res["pubs"], "other_pubs": res["other_pubs"]}) + "\n")
            print(f"{v} {y}: kept {len(res['records'])} / search {res['search_total']} pubs={res['pubs']} "
                  f"other={sum(res['other_pubs'].values())}")
        print(f"requests: {x.n_requests}")


if __name__ == "__main__":
    main()
