"""Enumerate non-IEEE venues via Crossref (cursor paging, polite pool, ~1 req/2 s).

IMAPS: prefix 10.4071, container "IMAPSource Proceedings" (all IMAPS events; Symposium filtered later by volume).
SMTA:  prefix 10.37665, container "Wafer-Level Packaging Symposium" (IWLPC candidate; identity checked in report).
Usage: uv run python scripts/crossref_fetch.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

OUT = ROOT / "cache" / "crossref"
JOBS = {
    "IMAPSource": ("10.4071", "IMAPSource Proceedings"),
    "SMTA_WLP": ("10.37665", "Wafer-Level Packaging Symposium"),
}
from openalex_enrich import load_env  # noqa: E402

load_env()
MAILTO = os.environ.get("CONTACT_EMAIL", "")  # Crossref polite pool; set in .env (git-ignored)


def strip_jats(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def slim(i: dict) -> dict:
    return {
        "doi": i["DOI"].lower(),
        "title": (i.get("title") or [""])[0],
        "authors": [
            {"name": " ".join(x for x in (a.get("given"), a.get("family")) if x),
             "affiliations": [af.get("name") for af in a.get("affiliation") or []]}
            for a in i.get("author") or []
        ],
        "container": (i.get("container-title") or [None])[0],
        "volume": i.get("volume"),
        "issue": i.get("issue"),
        "event": (i.get("event") or {}).get("name"),
        "published": "-".join(str(x) for x in ((i.get("published") or {}).get("date-parts") or [[None]])[0] if x),
        "abstract": strip_jats(i.get("abstract")),
        "url": i.get("URL"),
        "type": i.get("type"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as client:
        for name, (prefix, container) in JOBS.items():
            p = OUT / f"{name}.jsonl"
            if p.exists():
                print(f"{name}: exists, skipped")
                continue
            cursor, rows = "*", []
            while True:
                r = client.get(
                    f"https://api.crossref.org/prefixes/{prefix}/works",
                    params={"filter": f"from-pub-date:2022-01-01,until-pub-date:2026-12-31,container-title:{container}",
                            "rows": 500, "cursor": cursor, "mailto": MAILTO},
                )
                r.raise_for_status()
                m = r.json()["message"]
                items = m.get("items") or []
                rows += [slim(i) for i in items]
                print(f"{name}: +{len(items)} (total {len(rows)} / {m.get('total-results')})", flush=True)
                cursor = m.get("next-cursor")
                time.sleep(2)
                if not items or not cursor:
                    break
            with p.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
