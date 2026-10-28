"""Stage 6: build the self-contained gallery + viewer, index.html.

Usage: python3 -I pipeline/build.py [--out index.html]
Reads data/<pn>/{meta,text,figures,notes}.json, data/<pn>/traces/fig-<id>.json,
work/<pn>/figs/fig-<id>.png and work/<pn>/pages/p-NNN.png for every patent directory in data/.
Cross-links are computed here once: numeral spans and FIG. spans per paragraph, and per-part
mention sentences, so the page only wraps precomputed spans.
"""
import argparse
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

UNIT_WORDS = {
    "bit", "bits", "element", "elements", "clock", "clocks", "period", "periods", "word", "words", "parcel",
    "parcels", "times", "per", "percent", "degrees", "ns", "nanoseconds", "nanosecond", "inches", "inch", "cm",
    "mm", "gpm", "cfm", "psi", "watts", "kw", "minutes", "seconds", "octal", "registers", "register", "%",
    "x", "microns", "feet", "°",
}
PREV_BLOCK = {"fig.", "figs.", "fig", "figs", "claim", "claims", "column", "columns", "page",
              "no.", "nos.", "ser.", "patent", "about", "approximately", "of", "than", "from", "by", "only"}
ABBREV = ["FIGS.", "FIG.", "Figs.", "Fig.", "No.", "Nos.", "et al.", "e.g.", "i.e.", "U.S.", "approx.", "Inc.", "vs."]
FIG_RE = re.compile(r"\b(FIGS?|Figs?)\.?\s*(\d+[A-Z]?)((?:\s*(?:,|and|through|to|-)\s*\d+[A-Z]?)*)")
NUM_RE = re.compile(r"(?<![\w.,/-])(\d{1,3}[a-z]?)(?![\w/]|\.\d|,\d)")


def b64png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def as_bilevel(path: Path, scale: float = 1.0) -> Image.Image:
    im = Image.open(path).convert("L")
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        return im.convert("L")  # keep grey when downscaled, 1-bit would alias
    return im.point(lambda v: 255 if v > 140 else 0).convert("1")


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans [start, end) that respect common abbreviations."""
    masked = text
    for a in ABBREV:
        masked = masked.replace(a, a.replace(".", "․"))
    out, start = [], 0
    for m in re.finditer(r"[.;:?!](?=\s+[A-Z(\"])", masked):
        out.append((start, m.end()))
        start = m.end()
        while start < len(text) and text[start] == " ":
            start += 1
    if start < len(text):
        out.append((start, len(text)))
    return out


def link_spans(text: str, refs: set[str], fig_ids: set[str]) -> list[list]:
    """[start, end, 'fig'|'ref', key] spans, non-overlapping, sorted."""
    spans, taken = [], []
    for m in FIG_RE.finditer(text):
        # whole "FIG. 1" phrase links to the first figure; each later number links to its own figure
        first_end = m.start(2) + len(m.group(2))
        if m.group(2) not in fig_ids:  # e.g. the misprint "curved FIG. 95": leave the numeral linkable
            continue
        spans.append([m.start(), first_end, "fig", m.group(2)])
        for sm in re.finditer(r"\d+[A-Z]?", m.group(3) or ""):
            s = m.start(3) + sm.start()
            if sm.group() in fig_ids:
                spans.append([s, s + len(sm.group()), "fig", sm.group()])
        taken.append((m.start(), m.end()))
    for m in NUM_RE.finditer(text):
        ref = m.group(1)
        if ref not in refs or any(a <= m.start() < b for a, b in taken):
            continue
        before = text[: m.start()].rstrip()
        prev = before.split()[-1].lower() if before.split() else ""
        after = text[m.end():].lstrip(" ")
        nxt = re.match(r"[\w%°]+", after)
        nxt = nxt.group().lower() if nxt else ""
        if prev in PREV_BLOCK or nxt in UNIT_WORDS or not before:
            continue
        if prev == "to" and re.search(r"\d\s+to$", before):  # numeric range "0 to 63"
            continue
        if not re.search(r"[A-Za-z(),]$", before):  # must follow a word, a paren or a list comma
            continue
        spans.append([m.start(), m.end(), "ref", ref])
    spans.sort()
    return spans


def load_patent(pdir: Path) -> dict:
    pn = pdir.name
    meta = json.loads((pdir / "meta.json").read_text())
    text = json.loads((pdir / "text.json").read_text())
    figs = json.loads((pdir / "figures.json").read_text())
    notes = json.loads((pdir / "notes.json").read_text()) if (pdir / "notes.json").exists() else {}
    parts = {p["ref"]: dict(p) for p in figs["parts"]}
    refs, fig_ids = set(parts), {f["id"] for f in figs["figures"]}

    paras, sections = [], []
    for si, sec in enumerate(text["sections"]):
        sections.append({"heading": sec["heading"], "first": len(paras)})
        for p in sec["paras"]:
            q = {"id": len(paras), "sec": si, "col": p.get("col"), "line": p.get("line")}
            if "table" in p:
                q["table"] = p["table"]
                q["text"] = ""
                q["spans"] = []
            else:
                q["text"] = p["text"]
                q["spans"] = link_spans(p["text"], refs, fig_ids)
            paras.append(q)
    claims = []
    for c in text["claims"]:
        claims.append({**c, "spans": link_spans(c["text"], refs, fig_ids)})

    # per-part mention sentences, per-figure paragraphs
    for ref in parts:
        parts[ref]["mentions"] = []
    fig_paras = {f: [] for f in fig_ids}
    for q in paras:
        if not q["text"]:
            continue
        sents = sentences(q["text"])
        for s, e, typ, key in q["spans"]:
            if typ == "fig":
                if q["id"] not in fig_paras[key]:
                    fig_paras[key].append(q["id"])
                continue
            for a, b in sents:
                if a <= s < b:
                    ms = parts[key]["mentions"]
                    if not ms or ms[-1][0] != q["id"] or ms[-1][1] != a:
                        ms.append([q["id"], a, b])
                    break

    brief = {b["fig"]: b for b in text.get("brief_drawings", [])}
    out_figs = []
    for f in figs["figures"]:
        png = ROOT / "work" / pn / "figs" / f"fig-{f['id']}.png"
        im = as_bilevel(png)
        tpath = pdir / "traces" / f"fig-{f['id']}.json"
        trace = json.loads(tpath.read_text()) if tpath.exists() else None
        if trace and (trace["w"], trace["h"]) != im.size:
            raise SystemExit(f"{pn} fig {f['id']}: trace {trace['w']}x{trace['h']} != crop {im.size}")
        fig_refs = [r for r, p in parts.items() if f["id"] in p.get("figs", [])]
        if trace:
            fig_refs += [e["ref"] for e in trace["elements"] if e.get("ref") and e["ref"] not in fig_refs]
        first = {r: (parts[r]["mentions"][0][0] if r in parts and parts[r]["mentions"] else 1e9) for r in fig_refs}
        fig_refs.sort(key=lambda r: (first[r], int(re.sub(r"\D", "", r) or 0)))
        # sub-figures (3A..3D) share the parent's brief description when the patent gives only one
        b = brief.get(f["id"]) or brief.get(re.sub(r"[A-Z]+$", "", f["id"]), {})
        out_figs.append({
            "id": f["id"], "label": f.get("label", f"FIG. {f['id']}"), "sheet": f.get("sheet"), "page": f["page"],
            "front": bool(f.get("front_page")), "w": im.width, "h": im.height, "img": b64png(im),
            "thumb": b64png(as_bilevel(png, 220 / im.width)),
            "brief": {"text": b.get("text", ""), "col": b.get("col"), "line": b.get("line")},
            "trace": trace, "paras": fig_paras[f["id"]], "refs": fig_refs,
        })
    pages = [b64png(as_bilevel(p)) for p in sorted((ROOT / "work" / pn / "pages").glob("p-*.png"))]
    keep = ["number", "title", "inventors", "assignee", "filed", "issued", "appl_no", "int_cl", "us_cl",
            "refs_patents", "refs_other", "examiners", "claims_count", "drawing_figures", "abstract"]
    return {
        "pn": pn, **{k: text.get(k) for k in keep}, "google": meta.get("google"), "col1_page": meta["col1_page"],
        "sections": sections, "paras": paras, "claims": claims, "figures": out_figs, "parts": parts,
        "notes": notes, "pages": pages, "ocr_fixes": len(text.get("ocr_fixes", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="index.html")
    a = ap.parse_args()
    pats = [load_patent(d) for d in sorted((ROOT / "data").iterdir()) if (d / "text.json").exists() and (d / "figures.json").exists()]
    v = ROOT / "viewer"
    html = (v / "template.html").read_text()
    data = json.dumps(pats, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (html.replace("/*__CSS__*/", (v / "app.css").read_text())
                .replace("/*__RENDER__*/", (v / "trace-render.js").read_text())
                .replace("/*__APP__*/", (v / "app.js").read_text())
                .replace("/*__DATA__*/", f"window.PATENTS = {data};"))
    out = ROOT / a.out
    out.write_text(html)
    for p in pats:
        nt = sum(1 for f in p["figures"] if f["trace"])
        nm = sum(len(x["mentions"]) for x in p["parts"].values())
        print(f"{p['pn']}: {len(p['paras'])} paras, {len(p['claims'])} claims, {len(p['figures'])} figs ({nt} traced), "
              f"{len(p['parts'])} parts, {nm} part mentions")
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
