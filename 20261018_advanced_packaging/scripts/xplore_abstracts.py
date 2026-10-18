"""Backfill full abstracts + author affiliations from Xplore document pages for kept pool rows whose
abstract is only the truncated Xplore snippet. Resumable: cache/xplore_docs.jsonl (skip-existing).

Usage: uv run python scripts/xplore_abstracts.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT, AntiBotError, xplore_session  # noqa: E402

OUT = ROOT / "cache" / "xplore_docs.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    done = {json.loads(ln)["article_number"] for ln in OUT.read_text().splitlines()} if OUT.exists() else set()
    pool = [json.loads(ln) for ln in (ROOT / "cache" / "pool.jsonl").read_text().splitlines()]
    todo = [r["article_number"] for r in pool if r["abstract_source"] in ("xplore_truncated", "none")
            and r["prefilter_reason"] != "front matter / non-paper"
            and r["article_number"] and r["article_number"] not in done]
    # kept rows first so scoring can start on them early
    keep = {r["article_number"] for r in pool if r["prefilter_keep"]}
    todo.sort(key=lambda a: a not in keep)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} cached, {len(todo)} to fetch")
    if args.dry_run or not todo:
        return
    with xplore_session("xplore_abstracts") as x, OUT.open("a") as f:
        for an in todo:
            try:
                md = x.get(f"/rest/document/{an}/abstract")
            except AntiBotError as e:
                print(f"STOP: {e}")
                break
            except RuntimeError as e:
                f.write(json.dumps({"article_number": an, "found": False, "error": str(e)[:200]}) + "\n")
                continue
            auths = md.get("authors") or []
            row = {
                "article_number": an, "found": bool(md.get("abstract")),
                "abstract": re.sub(r"<[^>]+>", "", md.get("abstract") or "").strip() or None,
                "affiliations": [a for au in auths for a in (au.get("affiliation") or [])],
                "first_affiliation": ((auths or [{}])[0].get("affiliation") or [None])[0],
                "is_open_access": md.get("isOpenAccess"),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
        print(f"requests: {x.n_requests}")


if __name__ == "__main__":
    main()
