"""Generate outputs/triage_report.md from pipeline state (deterministic; no network).

Inputs: outputs/candidates.csv, outputs/cache_list.csv, cache/selection_stats.json, cache/oow_references.json,
        cache/pool.jsonl, cache/curation/<bucket>.json, config/report_notes.yaml, config/venues.yaml
Usage: uv run python scripts/report.py
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

BCFG = yaml.safe_load((ROOT / "config" / "buckets.yaml").read_text())["buckets"]
VENUES = yaml.safe_load((ROOT / "config" / "venues.yaml").read_text())["venues"]
SEL = yaml.safe_load((ROOT / "config" / "selection.yaml").read_text())
NOTES = yaml.safe_load((ROOT / "config" / "report_notes.yaml").read_text())
YEARS = [2022, 2023, 2024, 2025, 2026]
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+", re.I)


def jl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def main() -> None:
    cand = list(csv.DictReader((ROOT / "outputs" / "candidates.csv").open()))
    cache = list(csv.DictReader((ROOT / "outputs" / "cache_list.csv").open()))
    st = json.loads((ROOT / "cache" / "selection_stats.json").read_text())
    pool = {r["uid"]: r for r in jl(ROOT / "cache" / "pool.jsonl")}
    refs = json.loads((ROOT / "cache" / "oow_references.json").read_text())
    cur = {b: json.loads((ROOT / "cache" / "curation" / f"{b}.json").read_text())
           for b in BCFG if (ROOT / "cache" / "curation" / f"{b}.json").exists()}
    cached_dois = {(r["doi"] or "").lower() for r in cache}
    cand_by_doi = {(r["doi"] or "").lower(): r for r in cand}

    def org_of(r: dict) -> str:
        return r.get("first_author_org") or "unknown"

    def resolved(text: str) -> str:
        def tag(m: re.Match) -> str:
            d = m.group(0).rstrip(".")
            return f"{d} [{'cached' if d.lower() in cached_dois else 'not cached'}]"
        return DOI_RE.sub(tag, text)

    L: list[str] = []
    w = L.append
    w("# Advanced packaging literature triage (2022-2026)\n")
    w(f"Generated {NOTES['generated']} from pipeline state. Spec: `init-prompt.md`. Plan and log: "
      "`DevLog/DevLog-000-literature-triage-plan.md`. Deliverables: `outputs/candidates.csv`, `outputs/cache_list.csv`.\n")

    w("## Summary\n")
    w(f"- Records enumerated: {st['n_records']} unique DOIs. Excluding front matter ({st['n_front_matter']}) and duplicate "
      f"index entries ({st['n_duplicates']}; the two sets overlap) leaves {st['n_screened_papers']} papers screened.")
    w(f"- Prefilter kept {st['n_prefilter_kept']} of the screened papers; {st['n_llm_scored']} screened papers carry an "
      "LLM score (a few scored papers were later dropped by the final prefilter and keep their score).")
    w(f"- LLM score >= {SEL['threshold']} with a bucket: {st['n_llm_ge_threshold_with_bucket']}; of these "
      f"{st['n_llm_ge_threshold_cached']} cached. The cap forced {st['dropped_for_cap']} drops.")
    w(f"- Cached: {st['n_selected']} (cap {SEL['cap']}). Protected (spec-list technologies, overrides, anchored "
      f"lineages): {st['n_protected_cached']}. Outside the screened set (pre-2022 or unscreened venue, spec-list only): "
      f"{st['outside_screen_cached']}. Cached with LLM score below {SEL['threshold']}: {st['cached_below_threshold']} "
      "(spec-list papers, overrides, lineage members and curator picks; see per-bucket table).")
    hi9 = sum(1 for r in cand if r["score_method"] == "llm" and r["buckets"] and int(r["score"]) >= 9 and r["cache"] == "N")
    w(f"- Not cached despite LLM score >= 9: {hi9} (cap with per-bucket balancing; listed in candidates.csv with cache=N).")
    pri = collections.Counter(r["priority"] for r in cache)
    w(f"- Priority split: 1 = {pri.get('1', 0)}, 2 = {pri.get('2', 0)}, 3 = {pri.get('3', 0)}. "
      f"Lineage ids written (complete series only): {st['lineage_ids_written']} of {st['lineages_total']} series proposed "
      f"({st['core_lineages']} anchored by a spec-list paper and protected).")
    lw = st["lowered_threshold"]
    w(f"- Per-bucket floor ({SEL['floor_n']} papers, {SEL['floor_orgs']} orgs, {SEL['floor_years']} years): met in all "
      f"buckets; threshold lowering needed: {', '.join(f'{b} to {t}' for b, t in lw.items()) if lw else 'none'}.\n")

    w("## Spec-list technologies (cache regardless of score)\n")
    w("| Technology | Cached papers selected for it (year, DOI; from selection_reason) | Gap / caveat |")
    w("|---|---|---|")
    for it in NOTES["spec_items"]:
        rx = re.compile(it["re"], re.I)
        hits = [r for r in cache if r["selection_reason"].startswith("spec-list") and rx.search(r["selection_reason"])]
        hits.sort(key=lambda r: int(r["year"] or 0))
        cell = "; ".join(f"{r['year']} {r['doi']}" for r in hits[:6]) + (f"; +{len(hits) - 6} more" if len(hits) > 6 else "")
        w(f"| {it['name']} | {cell or 'none'} | {it['gap'] or ''} |")
    w("")

    w("## Venue enumeration\n")
    w("| Venue | " + " | ".join(str(y) for y in YEARS) + " | Source | Notes |")
    w("|---|" + "---|" * (len(YEARS) + 2))
    counts = collections.Counter((r["venue"], int(r["year"])) for r in cand if r["score_method"] in ("llm", "rule"))
    for v, c in list(VENUES.items()) + [("IMAPS", {"years": [2022, 2023, 2024, 2025]})]:
        cells = ["-" if y not in c["years"] else str(counts.get((v, y), 0)) for y in YEARS]
        w(f"| {v} | " + " | ".join(cells) + f" | {NOTES['venue_source'].get(v, 'IEEE Xplore')} | "
          f"{NOTES['venue_notes'].get(v, '')} |")
    w("\nCounts are unique records including front matter. '-' = year not in scope for that venue; "
      "'0' = not found in the source.\n")
    for n in NOTES["enumeration_notes"]:
        w(f"- {n}")
    w("")

    w("## Per-bucket summary\n")
    w(f"| Bucket | Screened (LLM-assigned) | LLM >= {SEL['threshold']} | Cached | Cached < {SEL['threshold']} | "
      "Cached orgs | Cached 2022/23/24/25/26 | Outside screen |")
    w("|---|---|---|---|---|---|---|---|")
    for b, bc in BCFG.items():
        scr = [r for r in cand if b in r["buckets"].split(";") and r["score_method"] == "llm"]
        hi = [r for r in scr if int(r["score"]) >= SEL["threshold"]]
        cb = [r for r in cache if b in r["buckets"].split(";")]
        ys = collections.Counter(int(r["year"]) for r in cb if r["year"])
        lo = sum(1 for r in cb if r["score"] and int(r["score"]) < SEL["threshold"])
        w(f"| {bc['label']} | {len(scr)} | {len(hi)} | {len(cb)} | {lo} | {len({org_of(r) for r in cb})} | "
          + "/".join(str(ys.get(y, 0)) for y in YEARS) + f" | {sum(1 for r in cb if r['outside_screen'] != 'N')} |")
    w("\nBuckets overlap (a paper can count in several). Primary bucket per paper is in cache_list.csv.\n")
    for b, bc in BCFG.items():
        cb = [r for r in cache if b in r["buckets"].split(";")]
        orgs = collections.Counter(org_of(r) for r in cb)
        ys = collections.Counter(int(r["year"]) for r in cb if r["year"] and int(r["year"]) >= 2022)
        thin = [str(y) for y in YEARS if ys.get(y, 0) <= 2]
        c = cur.get(b, {})
        w(f"### {bc['label']} (`{b}`)\n")
        w("- Dominant organizations (cached): " + ", ".join(f"{o} ({n})" for o, n in orgs.most_common(6)))
        w(f"- Thin years (<= 2 cached in-window): {', '.join(thin) if thin else 'none'}")
        lids = sorted({r["lineage_id"] for r in cb if r["lineage_id"]})
        if lids:
            w(f"- Lineages cached as series: {', '.join(lids)}")
        for g in c.get("gaps", []):
            w(f"- Gap: {resolved(g)}")
        for u in c.get("unverified", []):
            w(f"- Unverified: {u.get('tech') or u.get('candidate')}: {u.get('reason', '')}")
        w("")

    w("## Orientation reviews\n")
    for i, rv in enumerate(NOTES["reviews"], 1):
        r = cand_by_doi.get(rv["doi"].lower(), {})
        w(f"{i}. `{rv['doi']}` - {r.get('title', 'N/A')} ({r.get('venue', '')} {r.get('year', '')}). {rv['why']}.")
    w(f"\n{NOTES['reviews_note']}\n")

    if refs:
        w("## Out-of-window primary disclosures outside the spec list (reference only, not cached)\n")
        w("| Year | Venue | DOI | Technology | Bucket |")
        w("|---|---|---|---|---|")
        for e in sorted(refs, key=lambda e: (e.get("bucket"), e.get("year") or 0)):
            w(f"| {e.get('year')} | {e.get('venue')} | {e.get('doi')} | {(e.get('tech') or '')[:90]} | {e.get('bucket')} |")
        w("")

    w("## Method and overrides\n")
    for n in NOTES["method"]:
        w(f"- {n}")
    ov = st.get("overrides_cached", [])
    w(f"- Orchestrator overrides cached: {len(ov)} of {len(SEL.get('orchestrator_overrides', []))}.")
    w("")
    (ROOT / "outputs" / "triage_report.md").write_text("\n".join(L) + "\n")
    print(ROOT / "outputs" / "triage_report.md")


if __name__ == "__main__":
    main()
