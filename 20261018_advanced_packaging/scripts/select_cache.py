"""Merge LLM scores + per-bucket curation into outputs/candidates.csv and outputs/cache_list.csv.

Inputs: cache/pool.jsonl, cache/scoring/out_*.jsonl (out_r* override, out_rebucket.jsonl re-buckets),
        cache/curation/<bucket>.json, config/selection.yaml, config/buckets.yaml, config/orgs.yaml
Outputs: outputs/candidates.csv, outputs/cache_list.csv, cache/selection_stats.json, cache/oow_references.json

Rules (init-prompt.md + decisions in DevLog-000):
- Protected: spec-list technologies (curator mandatory/out-of-window entries whose tech text matches
  spec_mandatory_re), orchestrator overrides, and core lineages (series containing a protected paper), completed.
- Candidates: protected + curator picks + LLM score >= threshold (with a bucket).
- Per-bucket floor: >= floor_n papers, >= floor_orgs orgs, >= floor_years years, lowering the threshold to 5.
- Cap: drop from the most over-represented bucket first, lowest effective score first
  (score + pick_bonus if curator pick + priority1_bonus if curator priority 1), never breaking another bucket's floor.
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

SEL = yaml.safe_load((ROOT / "config" / "selection.yaml").read_text())
BUCKETS = list(yaml.safe_load((ROOT / "config" / "buckets.yaml").read_text())["buckets"])
CONDITIONAL = {"cpo", "thermal_pdn"}
ORGS = yaml.safe_load((ROOT / "config" / "orgs.yaml").read_text())["orgs"]
SPEC_RE = re.compile(SEL["spec_mandatory_re"], re.I)
SPEC_EX = re.compile(SEL["spec_mandatory_exclude_re"], re.I)
ORG_WORDS = re.compile(r"univ|institut|inc\b|corp|ltd|co\.|company|gmbh|laborator|center|centre|research|"
                       r"technolog|semiconductor|electronics|foundry|imec|fraunhofer|a\*star", re.I)


def is_spec(tech: str | None) -> bool:
    return bool(tech) and bool(SPEC_RE.search(tech)) and not SPEC_EX.search(tech)


def jl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


SUFFIX = re.compile(r"\b(inc|corp|corporation|co|ltd|limited|gmbh|ag|sa|llc|kk|company)\b\.?", re.I)


def clean_aff(a: str | None) -> str | None:
    return re.sub(r"\s*\((OpenAlex|Crossref)[^)]*\)", "", a).strip() if a else a


def org_key(r: dict) -> str:
    if r.get("first_org"):
        return r["first_org"]
    aff = clean_aff(r.get("first_affiliation")) or ""
    segs = [s.strip() for s in aff.split(",") if s.strip()]
    for s in segs:
        if ORG_WORDS.search(s) and not re.match(r"^(school|department|dept|faculty|college|division)\b", s, re.I):
            return s[:60]
    return (segs[0] if segs else "unknown")[:60]


def norm_org(o: str) -> str:
    return re.sub(r"[^a-z0-9]", "", SUFFIX.sub("", o.lower()))


def ntitle(t: str) -> str:
    return "".join(c.lower() for c in t if c.isalnum())


def main() -> None:
    pool = {r["uid"]: r for r in jl(ROOT / "cache" / "pool.jsonl")}
    scores: dict[str, dict] = {}
    for p in sorted((ROOT / "cache" / "scoring").glob("out_*.jsonl")):
        if p.name.startswith("out_rebucket"):
            continue
        for s in jl(p):
            scores[s["uid"]] = s
    for rb in jl(ROOT / "cache" / "scoring" / "out_rebucket.jsonl"):
        if rb["uid"] in scores:
            scores[rb["uid"]] = {**scores[rb["uid"]], "buckets": rb.get("buckets") or [],
                                 "stack_detailed": rb.get("stack_detailed", scores[rb["uid"]].get("stack_detailed"))}
    cur = {b: json.loads((ROOT / "cache" / "curation" / f"{b}.json").read_text())
           for b in BUCKETS if (ROOT / "cache" / "curation" / f"{b}.json").exists()}

    rows = []
    for uid, r in pool.items():
        s = scores.get(uid)
        if s:
            bks = [b for b in s.get("buckets") or [] if b in BUCKETS]
            bks = [b for b in bks if b not in CONDITIONAL or s.get("stack_detailed")]
            row = {**r, "buckets": bks, "score": int(s["score"]), "score_method": "llm",
                   "rationale": s.get("rationale") or ""}
        else:
            why = (f"prefilter drop ({r['prefilter_reason']}); not LLM-scored" if not r["prefilter_keep"]
                   else "kept by prefilter but abstract unavailable; not LLM-scored")
            row = {**r, "buckets": [], "score": None, "score_method": "rule", "rationale": why}
        row["org"] = org_key(r)
        rows.append(row)

    # dedup: same normalized title within venue-year; keep canonical (shorter) DOI, inherit best score
    groups: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        groups[(r["venue"], r["year"], ntitle(r["title"]))].append(r)
    dup_dropped = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        keeper = min(g, key=lambda r: len(r["doi"] or r["uid"]))
        top = max(g, key=lambda r: r["score"] if r["score"] is not None else -1)
        if top is not keeper and (top["score"] if top["score"] is not None else -1) > \
                (keeper["score"] if keeper["score"] is not None else -1):
            for f in ("score", "score_method", "rationale"):
                keeper[f] = top[f]
            keeper["buckets"] = list(top["buckets"])
        for r in g:
            if r is not keeper:
                r["duplicate"] = True
                r["rationale"] = f"duplicate index entry of {keeper['doi'] or keeper['uid']}; {r['rationale']}"
                dup_dropped += 1
    by_uid = {r["uid"]: r for r in rows}
    dup_of = {r["uid"]: (r["rationale"].split(";")[0].replace("duplicate index entry of ", "")) for r in rows
              if r.get("duplicate")}

    def canon(u: str) -> str:
        return dup_of.get(u, u)

    # curation
    protected, reason, cur_pri, cur_rank, home = set(), {}, {}, {}, {}
    home_rank: dict[str, int] = {}
    home_fixed: set = set()  # mandatory/out-of-window papers keep the bucket of the curator that mandated them
    lineage_sets: list[tuple[str, list[str]]] = []
    rejected_in: dict[str, set] = collections.defaultdict(set)
    stripped: list[str] = []
    oow_refs, notes = [], {}

    def add_bucket(uid: str, b: str) -> None:
        if b not in by_uid[uid]["buckets"]:
            by_uid[uid]["buckets"].append(b)

    for b, c in cur.items():
        for e in c.get("out_of_window", []):
            uid = (e.get("doi") or e["title"]).lower()
            if not is_spec(e.get("tech")):
                oow_refs.append({**e, "bucket": b})
                continue
            if uid not in by_uid:
                yr = e.get("year")
                row = {"uid": uid, "doi": e.get("doi"), "title": e["title"], "first_author": e.get("first_author"),
                       "first_affiliation": e.get("affiliation"), "venue": e.get("venue"), "year": yr,
                       "session": None, "committee": None, "buckets": [b], "score": None,
                       "score_method": "curator", "rationale": f"curator-verified primary disclosure: {e.get('tech')}",
                       "access": e.get("access") or "N/A",
                       "url": e.get("url") or (f"https://doi.org/{e['doi']}" if e.get("doi") else None),
                       "orgs": [], "industrial": True,
                       "outside_screen": "pre-2022" if (yr or 0) < 2022 else "unscreened venue"}
                row["org"] = org_key(row)
                rows.append(row)
                by_uid[uid] = row
            protected.add(uid)
            reason.setdefault(uid, f"spec-list primary disclosure ({e.get('tech')})")
            home.setdefault(uid, b)
        for m in c.get("mandatory", []):
            u = canon(m["uid"])
            if u not in by_uid:
                continue
            add_bucket(u, b)
            if u not in home_fixed:
                home[u] = b
                home_fixed.add(u)
            if is_spec(m.get("tech")):
                protected.add(u)
                reason.setdefault(u, f"spec-list technology ({m.get('tech')})")
            else:
                cur_pri[u], cur_rank[u] = 1, 0
                reason.setdefault(u, f"curator first-party disclosure ({m.get('tech')})")
        for i, p in enumerate(c.get("picks", [])):
            u = canon(p["uid"])
            if u not in by_uid:
                continue
            add_bucket(u, b)
            cur_pri[u] = min(cur_pri.get(u, 9), int(p.get("priority", 2)))
            rk = int(p.get("rank", i + 1))
            if rk < cur_rank.get(u, 999):
                cur_rank[u] = rk
            if rk < home_rank.get(u, 999) and u not in home_fixed:
                home[u], home_rank[u] = b, rk
        for lin in c.get("lineages", []):
            uids = list(dict.fromkeys(canon(u) for u in lin["uids"] if canon(u) in by_uid))
            if len(uids) >= 2:
                lineage_sets.append((lin["lineage_id"], uids))
        for u in c.get("reject", []):
            rejected_in[b].add(u)
        if c.get("notes"):
            notes[b] = c["notes"]

    for o in SEL.get("orchestrator_overrides", []):
        if o["uid"] in by_uid:
            add_bucket(o["uid"], o["bucket"])
            protected.add(o["uid"])
            reason[o["uid"]] = f"orchestrator override ({o['why']})"
            home.setdefault(o["uid"], o["bucket"])

    # bucket-scoped rejects (a reject of a duplicate entry does not touch its canonical record)
    for b, uids in rejected_in.items():
        for u in uids:
            r = by_uid.get(u)
            if r and not r.get("duplicate") and b in r["buckets"] and u not in protected:
                r["buckets"].remove(b)
                if not r["buckets"]:
                    stripped.append(u)
    rb2 = {x["uid"]: x for x in jl(ROOT / "cache" / "scoring" / "out_rebucket2.jsonl")}
    for u in stripped:
        if u in rb2:
            by_uid[u]["buckets"] = [b for b in rb2[u].get("buckets") or [] if b in BUCKETS
                                    and (b not in CONDITIONAL or rb2[u].get("stack_detailed"))]

    # org display names: merge spelling variants (case, punctuation, corporate suffixes)
    disp: dict[str, str] = {}
    for r in rows:
        r["org"] = disp.setdefault(norm_org(r["org"]) or r["org"], r["org"])

    # lineages: identical or subset series collapse into the larger one; overlapping-but-different series stay separate
    uniq: list[tuple[str, frozenset]] = []
    for name, uids in sorted(lineage_sets, key=lambda x: -len(x[1])):
        fs = frozenset(uids)
        if any(fs <= other for _, other in uniq):
            continue
        uniq.append((name, fs))
    lineages = {name: sorted(fs) for name, fs in uniq}
    lineage_ids: dict[str, list[str]] = collections.defaultdict(list)
    for name, mem in lineages.items():
        for u in mem:
            lineage_ids[u].append(name)
    core = {n for n, mem in lineages.items() if any(u in protected for u in mem) and len(mem) <= SEL["lineage_max"]}

    selected: dict[str, str] = {}
    for r in rows:
        u = r["uid"]
        if r.get("duplicate"):
            continue
        if u in protected:
            selected[u] = reason[u]
        elif u in cur_pri and r["buckets"] and (r["score"] or 0) >= SEL["pick_min_score"]:
            selected[u] = reason.get(u, "curator pick")
        elif r["score"] is not None and r["score"] >= SEL["threshold"] and r["buckets"]:
            selected[u] = f"LLM score >= {SEL['threshold']}"

    def eligible(u: str) -> bool:
        r = by_uid[u]
        return bool(r["buckets"]) and not r.get("duplicate") and (r["score"] is None or r["score"] >= SEL["floor_min_score"])
    # complete series: core (anchored by a protected paper) and near-complete ones (>= half cached, <= 2 missing)
    kept_series = set()
    for n, mem in lineages.items():
        missing = [u for u in mem if u not in selected]
        near = (len(mem) - len(missing)) * 2 >= len(mem) and len(missing) <= SEL["lineage_complete_missing_max"]
        if (n in core or near) and len(mem) <= SEL["lineage_max"] and all(eligible(u) for u in missing):
            for u in mem:
                selected.setdefault(u, f"lineage member ({n})")
                if n in core:  # only series anchored by a spec-list paper are protected from the cap
                    protected.add(u)
            if n in core:
                kept_series.add(n)

    def bucket_stats(sel) -> dict:
        st = {}
        for b in BUCKETS:
            m = [by_uid[u] for u in sel if b in by_uid[u]["buckets"]]
            st[b] = {"n": len(m), "orgs": len({x["org"] for x in m}), "years": len({x["year"] for x in m}),
                     "below_threshold": sum(1 for x in m if x["score"] is not None and x["score"] < SEL["threshold"])}
        return st

    def ok(st: dict) -> bool:
        return st["n"] >= SEL["floor_n"] and st["orgs"] >= SEL["floor_orgs"] and st["years"] >= SEL["floor_years"]

    lowered = {}
    for b in BUCKETS:
        for thr in range(SEL["threshold"] - 1, SEL["floor_min_score"] - 1, -1):
            if ok(bucket_stats(selected)[b]):
                break
            cand = [r for r in rows if b in r["buckets"] and r["uid"] not in selected and not r.get("duplicate")
                    and r["score"] == thr]
            have = {by_uid[u]["org"] for u in selected if b in by_uid[u]["buckets"]}
            cand.sort(key=lambda r: (r["org"] in have, not r.get("industrial", False)))
            for r in cand:
                if ok(bucket_stats(selected)[b]):
                    break
                selected[r["uid"]] = f"floor fill {b} (score {thr})"
                lowered[b] = thr

    def eff(r: dict) -> float:
        bonus = SEL["pick_bonus"] if r["uid"] in cur_pri else 0
        if cur_pri.get(r["uid"]) == 1:
            bonus += SEL["priority1_bonus"]
        return (r["score"] or 0) + bonus

    dropped = []
    while len(selected) > SEL["cap"]:
        st = bucket_stats(selected)
        done = False
        # spec: drop from the most over-represented bucket first (count of cached papers in the bucket)
        for b in sorted(BUCKETS, key=lambda b: -st[b]["n"]):
            cands = [by_uid[u] for u in selected if b in by_uid[u]["buckets"] and u not in protected
                     and all(st[b2]["n"] > SEL["floor_n"] for b2 in by_uid[u]["buckets"])]
            if not cands:
                continue
            # lowest effective score; ties: lower raw score, bucket-exclusive before multi-bucket, worse curator rank
            worst = min(cands, key=lambda r: (eff(r), r["score"] or 0, len(r["buckets"]) > 1,
                                              -cur_rank.get(r["uid"], 999), r["year"] or 0))
            dropped.append(worst["uid"])
            del selected[worst["uid"]]
            done = True
            break
        if not done:
            break

    def priority(r: dict) -> int:
        u = r["uid"]
        if u in protected and not selected.get(u, "").startswith("lineage"):
            return 1
        if u in cur_pri:
            return cur_pri[u]
        kinds = {ORGS[o]["kind"] for o in r.get("orgs") or [] if o in ORGS}
        if kinds & {"vendor", "foundry_osat", "idm", "memory"} and (r["score"] or 0) >= SEL["threshold"]:
            return 1
        return 2 if (r["score"] or 0) >= SEL["threshold"] else 3

    def primary_bucket(r: dict) -> str:
        h = home.get(r["uid"])
        return h if h in r["buckets"] else (r["buckets"][0] if r["buckets"] else "none")

    def complete(n: str) -> bool:
        return all(u in selected for u in lineages[n])

    def lineage_cell(u: str) -> str:
        ids = [n for n in lineage_ids.get(u, []) if complete(n)]
        return ";".join(ids)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    with (out / "candidates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doi", "title", "first_author", "affiliation", "venue", "year", "session", "buckets", "score",
                    "score_method", "rule_score", "rationale", "access", "cache"])
        for r in sorted(rows, key=lambda r: (r["venue"] or "", r["year"] or 0, r["title"])):
            sess = " | ".join(x for x in (r.get("committee"), r.get("session")) if x)
            w.writerow([r["doi"] or r["url"], r["title"], r["first_author"], clean_aff(r["first_affiliation"]), r["venue"],
                        r["year"], sess, ";".join(r["buckets"]), "" if r["score"] is None else r["score"],
                        r["score_method"], r.get("rule_score", ""), r["rationale"], r["access"],
                        "Y" if r["uid"] in selected else "N"])
    sel_rows = [by_uid[u] for u in selected]
    sel_rows.sort(key=lambda r: (BUCKETS.index(primary_bucket(r)) if primary_bucket(r) in BUCKETS else 99,
                                 r["year"] or 0, -(r["score"] or 0)))
    lin_written: set = set()
    with (out / "cache_list.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["primary_bucket", "buckets", "year", "doi", "title", "first_author", "affiliation",
                    "first_author_org", "venue", "session", "score", "priority", "lineage_id", "outside_screen",
                    "selection_reason", "rationale", "access", "url"])
        for r in sel_rows:
            lid = lineage_cell(r["uid"])
            lin_written |= set(filter(None, lid.split(";")))
            sess = " | ".join(x for x in (r.get("committee"), r.get("session")) if x)
            w.writerow([primary_bucket(r), ";".join(r["buckets"]), r["year"], r["doi"], r["title"], r["first_author"],
                        clean_aff(r["first_affiliation"]), r["org"], r["venue"], sess, "" if r["score"] is None else r["score"],
                        priority(r), lid, r.get("outside_screen", "N"), selected[r["uid"]], r["rationale"],
                        r["access"], r["url"]])

    screened = [r for r in rows if r["score_method"] in ("llm", "rule") and not r.get("duplicate")
                and r.get("prefilter_reason") != "front matter / non-paper"]
    llm_hi = [r for r in screened if r["score_method"] == "llm" and r["score"] >= SEL["threshold"] and r["buckets"]]
    stats = {
        "n_records": sum(1 for r in rows if r["score_method"] in ("llm", "rule")),
        "n_front_matter": sum(1 for r in rows if r.get("prefilter_reason") == "front matter / non-paper"),
        "n_duplicates": dup_dropped,
        "n_screened_papers": len(screened),
        "n_llm_scored": sum(1 for r in screened if r["score_method"] == "llm"),
        "n_prefilter_kept": sum(1 for r in screened if r.get("prefilter_keep")),
        "n_llm_ge_threshold_with_bucket": len(llm_hi),
        "n_llm_ge_threshold_cached": sum(1 for r in llm_hi if r["uid"] in selected),
        "n_selected": len(selected), "dropped_for_cap": len(dropped),
        "n_protected_cached": len(protected & set(selected)),
        "lowered_threshold": lowered, "lineages_total": len(lineages), "core_lineages": len(core), "series_kept": len(kept_series),
        "lineage_ids_written": len(lin_written),
        "outside_screen_cached": sum(1 for r in sel_rows if r.get("outside_screen")),
        "oow_references": len(oow_refs), "bucket_stats": bucket_stats(selected), "curation_notes": notes,
        "overrides_cached": [o["uid"] for o in SEL.get("orchestrator_overrides", []) if o["uid"] in selected],
        "cached_below_threshold": sum(1 for r in sel_rows if r["score"] is not None and r["score"] < SEL["threshold"]),
        "selected_uids": sorted(selected),
    }
    with (ROOT / "cache" / "scoring" / "rebucket2_input.jsonl").open("w") as f:
        for u in stripped:
            r = by_uid[u]
            if (r["score"] or 0) >= SEL["threshold"]:
                f.write(json.dumps({"uid": u, "venue": r["venue"], "year": r["year"], "title": r["title"],
                                    "abstract": r.get("abstract"), "first_affiliation": r["first_affiliation"],
                                    "score": r["score"], "rationale": r["rationale"],
                                    "rejected_from": [b for b, us in rejected_in.items() if u in us]}) + "\n")
    (ROOT / "cache" / "oow_references.json").write_text(json.dumps(oow_refs, indent=1))
    (ROOT / "cache" / "selection_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("bucket_stats", "curation_notes", "selected_uids")},
                     indent=1))
    print({b: (v["n"], v["orgs"], v["years"], v["below_threshold"]) for b, v in stats["bucket_stats"].items()})


if __name__ == "__main__":
    main()
