"""Enrich Xplore records with OpenAlex: full abstract, raw affiliations, institutions, OA status, preprint URLs.

Batches of 50 DOIs per request, ~1 req/s. Resumable: DOIs already in cache/openalex.jsonl are skipped
(a DOI with no OpenAlex hit is stored as {"doi": ..., "found": false}).
Usage: uv run python scripts/openalex_enrich.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT, log_path  # noqa: E402

OUT = ROOT / "cache" / "openalex.jsonl"
SELECT = "id,doi,title,type,abstract_inverted_index,authorships,open_access,best_oa_location,locations,cited_by_count"


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def abstract_from_inv(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def slim(w: dict) -> dict:
    auths = []
    for a in w.get("authorships") or []:
        auths.append({
            "name": (a.get("author") or {}).get("display_name"),
            "position": a.get("author_position"),
            "raw_affiliations": a.get("raw_affiliation_strings") or [],
            "institutions": [i.get("display_name") for i in a.get("institutions") or []],
            "countries": a.get("countries") or [],
        })
    oa = w.get("open_access") or {}
    best = w.get("best_oa_location") or {}
    repos = []
    for loc in w.get("locations") or []:
        src = loc.get("source") or {}
        if src.get("type") == "repository" or "arxiv" in (loc.get("landing_page_url") or ""):
            repos.append(loc.get("pdf_url") or loc.get("landing_page_url"))
    return {
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
        "found": True,
        "openalex_id": w.get("id"),
        "type": w.get("type"),
        "abstract": abstract_from_inv(w.get("abstract_inverted_index")),
        "authorships": auths,
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "oa_url": oa.get("oa_url") or best.get("pdf_url") or best.get("landing_page_url"),
        "repository_urls": [u for u in repos if u],
        "cited_by_count": w.get("cited_by_count"),
    }


def all_dois() -> list[str]:
    dois = []
    for p in sorted((ROOT / "cache" / "xplore").glob("*.jsonl")):
        for ln in p.read_text().splitlines():
            r = json.loads(ln)
            if r.get("doi"):
                dois.append(r["doi"].lower())
    return list(dict.fromkeys(dois))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    load_env()
    key = os.environ.get("OPENALEX_API_KEY")
    done = set()
    if OUT.exists():
        done = {json.loads(ln)["doi"] for ln in OUT.read_text().splitlines()}
    todo = [d for d in all_dois() if d not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} cached, {len(todo)} to fetch, key={'yes' if key else 'no'}")
    if args.dry_run or not todo:
        return
    log = log_path("openalex_enrich")
    with httpx.Client(timeout=60) as client, OUT.open("a") as fo:
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            params = {"filter": "doi:" + "|".join(batch), "per-page": 50, "select": SELECT}
            if key:
                params["api_key"] = key
            for attempt in range(4):
                r = client.get("https://api.openalex.org/works", params=params)
                if r.status_code == 200:
                    break
                time.sleep(5 * (attempt + 1))
            r.raise_for_status()
            data = r.json()
            found = {}
            for w in data.get("results") or []:
                s = slim(w)
                found[s["doi"]] = s
            for d in batch:
                fo.write(json.dumps(found.get(d, {"doi": d, "found": False})) + "\n")
            fo.flush()
            msg = (f"{datetime.now().isoformat(timespec='seconds')} batch {i // 50 + 1}: {len(found)}/{len(batch)} "
                   f"found cost={data.get('meta', {}).get('cost_usd')}")
            print(msg, flush=True)
            with log.open("a") as fl:
                fl.write(msg + "\n")
            time.sleep(1.0 + random.uniform(0, 0.5))


if __name__ == "__main__":
    main()
