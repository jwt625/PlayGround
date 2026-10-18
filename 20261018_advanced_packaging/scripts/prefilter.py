"""Merge all sources into one screening pool, assign regex buckets, prefilter, compute a rule-based provisional score.

Inputs: cache/xplore/*.jsonl, cache/openalex.jsonl, cache/ectc_sessions.jsonl, cache/crossref/*.jsonl,
        config/venues.yaml, config/buckets.yaml, config/orgs.yaml
Output: cache/pool.jsonl (one row per unique paper), printed summary.
Usage: uv run python scripts/prefilter.py
"""

from __future__ import annotations

import collections
import html
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

C = ROOT / "config"
VENUES = yaml.safe_load((C / "venues.yaml").read_text())["venues"]
BCFG = yaml.safe_load((C / "buckets.yaml").read_text())
ORGS = yaml.safe_load((C / "orgs.yaml").read_text())["orgs"]
BUCKET_RE = {b: re.compile("|".join(f"(?:{p})" for p in v["patterns"]), re.I) for b, v in BCFG["buckets"].items()}
CONTEXT_RE = re.compile("|".join(BCFG["packaging_context"]), re.I)
CORE_RE = re.compile("|".join(BCFG["packaging_core"]), re.I)
ORG_RE = {o: re.compile(v["re"], re.I if not v["re"].isupper() else 0) for o, v in ORGS.items()}
NON_PAPER = re.compile(
    r"^(front cover|back cover|title page|copyright|table of contents|author index|index|welcome|foreword|"
    r"message from|committee|organizing committee|program committee|sponsors|keynote|plenary|cover|"
    r"[^a-z]*errata|reviewers|blank page|conference program|session \d+ overview|\[?front cover|"
    r"ieee transactions on components|ieee components, packaging|techrxiv|ieee open access|member get-a-member|"
    r"information for authors|.*abstract page$|.*cover page$|20\d\d index ieee|index$)", re.I)

# Rule-based rubric (provisional; LLM pass is authoritative)
R_DIM = re.compile(r"\d+(\.\d+)?\s?(µm|μm|um|nm|mm|micron)\b|\b\d+\s?(-| )?layers?\b|\bL/S\b|line/space|"
                   r"\d+\s?/\s?\d+\s?(µm|μm|um)|pitch of \d|\d+\s?(µm|μm|um)\s?pitch|\d{3}\s?[x×]\s?\d{3}", re.I)
R_PROC = re.compile(r"process flow|process sequence|via-(first|middle|last)|reveal|semi-?additive|\bSAP\b|"
                    r"damascene|electroplat|\bCMP\b|plasma activat|anneal|thermo-?compression|reflow|"
                    r"lamination|laser drill|carrier (bond|release|debond)|debond|temporary bond|mold(ing)?|"
                    r"backgrind|thinning|sputter|lithograph|seed layer|underfill|\bPECVD\b|wet etch|DRIE|"
                    r"bonding at \d|at \d+\s?°\s?C|°C", re.I)
R_XSEC = re.compile(r"cross[- ]section|\bSEM\b|\bFIB\b|\bTEM\b|test vehicle|process flow|\bTV\b|demonstrat|"
                    r"fabricat|prototype", re.I)
R_MEAS = re.compile(r"warpage|yield|\bCTE\b|contact resistance|daisy[- ]chain|shear strength|measured|"
                    r"measurement|reliability test|thermal cycl|\bTCT\b|\bHAST\b|resistance of", re.I)
R_FIRST = re.compile(r"\bfirst\b|novel|new generation|next[- ]generation|world'?s|for the first time|"
                     r"introduc", re.I)
R_SIM = re.compile(r"simulat|finite element|\bFEA\b|\bFEM\b|model(l)?ing|numerical|machine learning|neural", re.I)
R_FAB = re.compile(r"fabricat|measured|experiment|demonstrat|test vehicle|prototype|cross[- ]section|"
                   r"characteriz", re.I)
R_REVIEW = re.compile(r"\breview\b|overview|tutorial|survey|perspective|roadmap|outlook|trends", re.I)
R_REL = re.compile(r"reliabilit|electromigration|fatigue|drop test|thermal cycl|failure analysis|lifetime", re.I)
R_ELEC = re.compile(r"signal integrity|power integrity|\bSI\b|\bPI\b|S-parameter|insertion loss|crosstalk|"
                    r"eye diagram|impedance|channel model|electromagnetic", re.I)


def _n(s: str | None) -> str:
    return "".join(c.lower() for c in (s or "") if c.isalnum())


def oa_abstract_ok(oa_abs: str | None, xp_snip: str | None) -> bool:
    """Reject OpenAlex placeholders ("International audience", page numbers) and swapped abstracts."""
    if not oa_abs or len(oa_abs) < 200:
        return False
    if xp_snip:
        head = _n(xp_snip.rstrip(". "))[:50]
        return head in _n(oa_abs)
    return True


def clean_text(t: str | None) -> str:
    """Unescape HTML entities (Xplore titles carry e.g. R&#x0026;D) and repair the garbled micro sign."""
    t = html.unescape(t or "")
    t = re.sub("(\\d)\\s?\u00bcm", "\\1 \u00b5m", t)
    if "$" in t:  # Xplore LaTeX in titles, e.g. "$2 \\times 4 \\mu\\mathrm{m}$"
        t = re.sub(r"\\(mathrm|text|mathbf|mathit)\s*\{([^}]*)\}", r"\2", t)
        for a, b in ((r"\times", " x "), (r"\mu", "\u00b5"), (r"\sim", "~"), (r"\pm", "+/-"),
                     (r"\leq", "<="), (r"\geq", ">="), (r"\ ", " "), ("~", " ")):
            t = t.replace(a, b)
        t = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\u00b5\s+", "\u00b5", m.group(1)), t)
        t = t.replace("\\", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def jl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def canon_orgs(affs: list[str]) -> list[str]:
    found = []
    for a in affs:
        for o, rx in ORG_RE.items():
            if rx.search(a) and o not in found:
                found.append(o)
    return found


def rule_score(title: str, abstract: str, industrial: bool) -> tuple[int, list[str]]:
    t = f"{title}. {abstract}"
    s, why = 0, []
    if R_DIM.search(t):
        s += 3; why.append("dims")
    if R_PROC.search(t):
        s += 2; why.append("process")
    if R_XSEC.search(t):
        s += 2; why.append("xsec/TV")
    if R_MEAS.search(t):
        s += 1; why.append("measured")
    if industrial:
        s += 1; why.append("industrial")
    if R_FIRST.search(t):
        s += 1; why.append("first/new")
    fab = bool(R_FAB.search(t))
    if R_SIM.search(t) and not fab:
        s -= 3; why.append("-sim only")
    if R_REVIEW.search(title):
        s -= 2; why.append("-review")
    if R_REL.search(t) and not (R_DIM.search(t) or R_PROC.search(t)):
        s -= 2; why.append("-rel only")
    if R_ELEC.search(t) and not fab:
        s -= 3; why.append("-elec model")
    return max(0, min(10, s)), why


def access_status(xp_access: str | None, oa: dict | None, crossref: bool = False) -> str:
    parts = []
    if oa and oa.get("is_oa") and oa.get("oa_url"):
        parts.append(f"OA ({oa.get('oa_status')}): {oa['oa_url']}")
    elif xp_access in ("open-access", "free", "ephemera"):
        parts.append(f"OA (Xplore {xp_access})")
    else:
        parts.append("subscription" if not crossref else "subscription/member (publisher)")
    for u in (oa or {}).get("repository_urls") or []:
        if "doi.org" not in u:
            parts.append(f"preprint: {u}")
            break
    return "; ".join(parts)


def main() -> None:
    oa = {r["doi"]: r for r in jl(ROOT / "cache" / "openalex.jsonl")}
    sess = {r["articleNumber"]: r for r in jl(ROOT / "cache" / "ectc_sessions.jsonl")}
    xdocs = {r["article_number"]: r for r in jl(ROOT / "cache" / "xplore_docs.jsonl") if r.get("found")}
    pool: dict[str, dict] = {}
    out_of_scope: collections.Counter = collections.Counter()

    for p in sorted((ROOT / "cache" / "xplore").glob("*.jsonl")):
        for r in jl(p):
            if r.get("_done"):
                continue
            doi = (r.get("doi") or "").lower()
            uid = doi or f"xplore:{r['articleNumber']}"
            if uid in pool:
                continue
            # year from Xplore publicationYear (TOC backfill of early-access issues mixes years)
            try:
                year = int(r.get("publicationYear") or r["year"])
            except (TypeError, ValueError):
                year = r["year"]
            if year not in VENUES.get(r["venue"], {}).get("years", [year]):
                out_of_scope[r["venue"]] += 1
                continue
            o = oa.get(doi) if doi else None
            auths = (o or {}).get("authorships") or []
            first_aff = (auths[0]["raw_affiliations"][:1] if auths else []) or []
            all_aff = [a for au in auths for a in au["raw_affiliations"]] + \
                      [i for au in auths for i in au["institutions"]]
            xd = xdocs.get(r["articleNumber"])
            if xd:
                all_aff += xd.get("affiliations") or []
                if not first_aff and xd.get("first_affiliation"):
                    first_aff = [xd["first_affiliation"]]
            s = sess.get(r["articleNumber"], {})
            full_abs = (o or {}).get("abstract")
            if full_abs and not oa_abstract_ok(full_abs, r.get("abstract")):
                full_abs = None
            abs_src = "openalex" if full_abs else None
            if not full_abs and xd and xd.get("abstract"):
                full_abs, abs_src = xd["abstract"], "xplore_full"
            pool[uid] = {
                "uid": uid, "doi": doi or None, "article_number": r["articleNumber"], "venue": r["venue"],
                "year": year, "title": r["articleTitle"] or "",
                "abstract": full_abs or r.get("abstract") or "",
                "abstract_source": abs_src or ("xplore_truncated" if r.get("abstract") else "none"),
                "first_author": (r.get("authors") or [{}])[0].get("name") if r.get("authors") else None,
                "first_affiliation": first_aff[0] if first_aff else None,
                "affiliations_all": list(dict.fromkeys(all_aff)),
                "session": s.get("session_title"), "session_no": s.get("session_no"),
                "committee": s.get("committee"),
                "access": access_status(r.get("accessType"), o),
                "url": f"https://ieeexplore.ieee.org/document/{r['articleNumber']}/",
                "content_type": r.get("displayContentType"), "publication": r.get("publicationTitle"),
            }

    # Crossref venues
    for r in jl(ROOT / "cache" / "crossref" / "IMAPSource.jsonl"):
        if r.get("issue") not in ("IMAPS Symposium", "Symposium") or r.get("volume") not in ("2022", "2023", "2024", "2025"):
            continue
        affs = [a for au in r["authors"] for a in au["affiliations"]]
        pool[r["doi"]] = {
            "uid": r["doi"], "doi": r["doi"], "article_number": None, "venue": "IMAPS", "year": int(r["volume"]),
            "title": r["title"], "abstract": r["abstract"] or "", "abstract_source": "crossref" if r["abstract"] else "none",
            "first_author": r["authors"][0]["name"] if r["authors"] else None,
            "first_affiliation": (r["authors"][0]["affiliations"] or [None])[0] if r["authors"] else None,
            "affiliations_all": list(dict.fromkeys(affs)), "session": None, "session_no": None, "committee": None,
            "access": "subscription/member (IMAPS)", "url": r["url"], "content_type": "Conference Paper",
            "publication": f"IMAPS International Symposium on Microelectronics {r['volume']}",
        }
    for r in jl(ROOT / "cache" / "crossref" / "SMTA_WLP.jsonl"):
        try:
            year = int(r["volume"]) + 2003  # IWLPC edition N held in year N+2003 (14th = 2017)
        except (TypeError, ValueError):
            continue
        if not 2022 <= year <= 2025:
            continue
        affs = [a for au in r["authors"] for a in au["affiliations"]]
        pool[r["doi"]] = {
            "uid": r["doi"], "doi": r["doi"], "article_number": None, "venue": "IWLPC", "year": year,
            "title": r["title"], "abstract": r["abstract"] or "", "abstract_source": "crossref" if r["abstract"] else "none",
            "first_author": r["authors"][0]["name"] if r["authors"] else None,
            "first_affiliation": (r["authors"][0]["affiliations"] or [None])[0] if r["authors"] else None,
            "affiliations_all": list(dict.fromkeys(affs)), "session": None, "session_no": None, "committee": None,
            "access": "subscription/member (SMTA)", "url": r["url"], "content_type": "Conference Paper",
            "publication": f"IWLPC {year} (SMTA 'Wafer-Level Packaging Symposium' vol {r['volume']})",
        }

    stats = collections.Counter()
    for row in pool.values():
        row["title"], row["abstract"] = clean_text(row["title"]), clean_text(row["abstract"])
        row["first_affiliation"] = clean_text(row["first_affiliation"]) or None
        row["affiliations_all"] = [clean_text(a) for a in row["affiliations_all"]]
        text = f"{row['title']}. {row['abstract']}"
        # first author's affiliation first, so orgs[0] is the first author's organization
        row["orgs"] = canon_orgs(([row["first_affiliation"]] if row["first_affiliation"] else []) + row["affiliations_all"])
        fo = canon_orgs([row["first_affiliation"]]) if row["first_affiliation"] else []
        row["first_org"] = fo[0] if fo else None  # first author's org only (no co-author fallback)
        row["industrial"] = any(ORGS[o]["kind"] not in ("university_center",) for o in row["orgs"])
        row["regex_buckets"] = [b for b, rx in BUCKET_RE.items() if rx.search(text)]
        mode = VENUES.get(row["venue"], {}).get("mode", "full")
        if NON_PAPER.search(row["title"]) or len(row["title"]) < 8:
            keep, reason = False, "front matter / non-paper"
        elif not row["abstract"]:
            keep, reason = bool(row["regex_buckets"]), "no abstract; title regex only"
        elif mode == "filtered":
            keep = bool(row["regex_buckets"]) and bool(CONTEXT_RE.search(text))
            reason = "filtered venue: bucket+packaging context" if keep else "filtered venue: no packaging bucket/context"
        else:
            core = bool(CORE_RE.search(text))
            keep = bool(row["regex_buckets"]) or core
            reason = ("bucket regex match" if row["regex_buckets"] else
                      "packaging-core term, no bucket regex" if core else "no bucket or packaging-core match")
        row["prefilter_keep"], row["prefilter_reason"] = keep, reason
        row["rule_score"], row["rule_why"] = rule_score(row["title"], row["abstract"], row["industrial"])
        stats[(row["venue"], keep)] += 1

    print("skipped (publicationYear outside venue scope):", dict(out_of_scope))
    out = ROOT / "cache" / "pool.jsonl"
    with out.open("w") as f:
        for row in pool.values():
            f.write(json.dumps(row) + "\n")
    venues = sorted({v for v, _ in stats})
    print(f"{'venue':8} {'total':>6} {'keep':>6}")
    for v in venues:
        print(f"{v:8} {stats[(v, True)] + stats[(v, False)]:6d} {stats[(v, True)]:6d}")
    keep = [r for r in pool.values() if r["prefilter_keep"]]
    print(f"TOTAL {len(pool)} keep {len(keep)}; abstract sources (kept): "
          f"{collections.Counter(r['abstract_source'] for r in keep)}")
    print("bucket regex hits (kept):", collections.Counter(b for r in keep for b in r["regex_buckets"]))
    print(out)


if __name__ == "__main__":
    main()
