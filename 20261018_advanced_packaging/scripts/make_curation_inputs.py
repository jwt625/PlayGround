"""Build per-bucket curation inputs for the selection subagents: cache/curation/input_<bucket>.jsonl
plus cache/curation/first_disclosure_flags.jsonl (all LLM flags, any bucket).

Usage: uv run python scripts/make_curation_inputs.py [--min-score 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

BUCKETS = list(yaml.safe_load((ROOT / "config" / "buckets.yaml").read_text())["buckets"])
CONDITIONAL = {"cpo", "thermal_pdn"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=4)
    args = ap.parse_args()
    pool = {json.loads(ln)["uid"]: json.loads(ln) for ln in (ROOT / "cache" / "pool.jsonl").read_text().splitlines()}
    scores = {}
    for p in sorted((ROOT / "cache" / "scoring").glob("out_*.jsonl")):
        for ln in p.read_text().splitlines():
            if ln.strip():
                s = json.loads(ln)
                scores[s["uid"]] = s  # later files (rescore_*) override earlier
    d = ROOT / "cache" / "curation"
    d.mkdir(parents=True, exist_ok=True)
    flags = []
    counts = {}
    for b in BUCKETS:
        rows = []
        for uid, s in scores.items():
            if b not in (s.get("buckets") or []) or uid not in pool:
                continue
            if b in CONDITIONAL and not s.get("stack_detailed"):
                continue
            if s["score"] < args.min_score and not s.get("first_disclosure"):
                continue
            p = pool[uid]
            rows.append({
                "uid": uid, "doi": p["doi"], "title": p["title"], "year": p["year"], "venue": p["venue"],
                "session": " | ".join(x for x in (p.get("committee"), p.get("session")) if x) or None,
                "first_author": p["first_author"], "first_affiliation": p["first_affiliation"], "orgs": p["orgs"],
                "score": s["score"], "rationale": s.get("rationale"), "buckets": s.get("buckets"),
                "first_disclosure": s.get("first_disclosure"), "lineage": s.get("lineage"),
                "is_review": s.get("is_review"), "access": p["access"],
            })
        rows.sort(key=lambda r: (-r["score"], r["year"]))
        with (d / f"input_{b}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[b] = (len(rows), sum(1 for r in rows if r["score"] >= 7))
    for uid, s in scores.items():
        if s.get("first_disclosure") and uid in pool:
            p = pool[uid]
            flags.append({"uid": uid, "title": p["title"], "year": p["year"], "venue": p["venue"],
                          "first_affiliation": p["first_affiliation"], "flag": s["first_disclosure"],
                          "buckets": s.get("buckets"), "score": s["score"]})
    with (d / "first_disclosure_flags.jsonl").open("w") as f:
        for r in sorted(flags, key=lambda r: r["year"]):
            f.write(json.dumps(r) + "\n")
    reviews = [{"uid": u, "title": pool[u]["title"], "year": pool[u]["year"], "venue": pool[u]["venue"],
                "first_affiliation": pool[u]["first_affiliation"], "buckets": s.get("buckets")}
               for u, s in scores.items() if s.get("is_review") and u in pool]
    with (d / "reviews.jsonl").open("w") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")
    print({b: f"{n} cand / {n7} >=7" for b, (n, n7) in counts.items()})
    print(f"{len(flags)} first-disclosure flags, {len(reviews)} reviews")


if __name__ == "__main__":
    main()
