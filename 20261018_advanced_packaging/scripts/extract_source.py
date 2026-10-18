"""Extract page-delimited text and figure images from a cached source PDF (adapted from eo-modulator-atlas).

Outputs in references/<paper_id>/ (git-ignored): text.md ("<!-- page N -->" markers + metadata header),
figures/page_NN.png (150 dpi render of every page), figures/img_pNN_k.png (embedded
rasters >= 150 px), figures/figures.json; updates source.json with sha256/pages/text_chars.

Usage: uv run python scripts/extract_source.py --paper-id <id> [--pdf <path>]   (default: references/<id>/source.pdf)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "references"
CAPTION_RE = re.compile(r"^\s*(Fig\.?|Figure|FIG\.?|Table|TABLE)\s*(S?[\dIVX]+)", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


COMMON = re.compile(r"\b(the|and|of|with|for|is|are|was|were|in|to)\b", re.I)


def page_quality(t: str) -> tuple[float, float] | None:
    ns = [c for c in t if not c.isspace()]
    if len(ns) < 300:
        return None
    alpha = sum(c.isalpha() for c in ns) / len(ns)
    common = len(COMMON.findall(t)) / max(1, len(t.split()))
    return alpha, common


def text_quality(body: str, pages: list[str] | None = None) -> dict[str, Any]:
    """Flag font-encoded/garbled text layers. Per page: alphabetic ratio < 0.55 or common-word ratio < 0.03.
    A document is garbled if any page is (b013: one paper had 3 of 8 pages shifted; whole-doc averages hid it)."""
    n = max(1, len(body))
    digits = sum(c.isdigit() for c in body) / n
    words = max(1, len(body.split()))
    common = len(COMMON.findall(body)) / words
    bad = [i + 1 for i, t in enumerate(pages or [body]) if (q := page_quality(t)) and (q[0] < 0.55 or q[1] < 0.03)]
    return {"digit_ratio": round(digits, 4), "common_word_ratio": round(common, 3), "garbled_pages": bad,
            "garbled_text_layer": bool(bad)}


NUM = re.compile(r"\d+(?:\.\d+)?")


def number_check(doc: Any, out: Path) -> list[int]:
    """Pages whose numbers are missing from the text layer (symbol-font shifts that keep letters readable;
    b017: ectc51909.2023.00307). OCR each page at 200 dpi; flag if >= 5 OCR numbers (>= 2 chars) and >= 25% of them
    are absent from the PDF text of that page."""
    bad = []
    for pno, page in enumerate(doc, start=1):
        txt = set(NUM.findall(page.get_text("text")))
        img = out / "figures" / f".ocr_tmp_{pno:02d}.png"
        page.get_pixmap(dpi=200).save(img)
        try:
            ocr = subprocess.run(["tesseract", str(img), "stdout"], capture_output=True, text=True, timeout=180).stdout
        except (OSError, subprocess.TimeoutExpired):
            ocr = ""
        img.unlink(missing_ok=True)
        o = {x for x in NUM.findall(ocr) if len(x) >= 2}
        miss = [x for x in o if x not in txt]
        if len(miss) >= 5 and len(miss) >= 0.25 * len(o):
            bad.append(pno)
    return bad


def extract(paper_id: str, pdf: Path | None = None) -> dict[str, Any]:
    out = REFS / paper_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    dest = out / "source.pdf"
    if pdf is not None and pdf.resolve() != dest.resolve():
        shutil.copyfile(pdf, dest)
    meta: dict[str, Any] = {}
    if (out / "source.json").exists():
        meta = json.loads((out / "source.json").read_text())
    doc = fitz.open(dest)
    header = (
        "---\n"
        f"paper_id: {paper_id}\n"
        f"doi: {meta.get('doi', '')}\n"
        f"source_url: {meta.get('url', '')}\n"
        f"sha256: {sha256(dest)}\n"
        f"pages: {doc.page_count}\n"
        f"extracted_on: {dt.date.today().isoformat()}\n"
        "extraction_method: pymupdf get_text('text'); tables/equations may be garbled; read figures from page renders\n"
        "---\n\n"
    )
    parts = [header]
    figures: list[dict[str, Any]] = []
    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text")
        parts.append(f"<!-- page {pno} -->\n{text}\n")
        captions = [ln.strip() for ln in text.splitlines() if CAPTION_RE.match(ln)]
        name = f"page_{pno:02d}.png"  # every page rendered (pilot feedback: figures on caption-less pages were missed)
        page.get_pixmap(dpi=150).save(out / "figures" / name)
        figures.append({"kind": "page_render", "file": name, "page": pno, "captions": captions[:12]})
        for n, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                pm = fitz.Pixmap(doc, xref)
                if pm.width < 150 or pm.height < 150:
                    continue
                if pm.alpha:
                    pm = fitz.Pixmap(pm, 0)
                if pm.colorspace is None or pm.colorspace.n not in (1, 3):
                    pm = fitz.Pixmap(fitz.csRGB, pm)
                name = f"img_p{pno:02d}_{n}.png"
                pm.save(out / "figures" / name)
                figures.append({"kind": "embedded", "file": name, "page": pno, "width": pm.width, "height": pm.height})
            except Exception as exc:  # noqa: BLE001  pymupdf raises its own error types
                figures.append({"kind": "embedded_failed", "page": pno, "xref": xref, "error": str(exc)[:120]})
    q = text_quality("".join(parts[1:]), [pg.get_text("text") for pg in doc])
    # number_check() is NOT run automatically: it false-flagged ~19 clean papers (OCR reads numbers inside raster
    # figures/tables that a text layer never contains). Confirmed number-only garbling is forced via
    # config/force_ocr.txt (one paper_id per line), set after a human/agent check.
    force = ROOT / "config" / "force_ocr.txt"
    if not q["garbled_text_layer"] and force.exists() and paper_id in force.read_text().split():
        q["garbled_text_layer"] = True
        q["garbled_pages"] = ["forced"]
    if q["garbled_text_layer"]:
        # unusable text layer: hi-res renders for reading numbers off the page images, and a loud warning
        for pno, page in enumerate(doc, start=1):
            page.get_pixmap(dpi=220).save(out / "figures" / f"page_{pno:02d}_220dpi.png")
        ocr_parts = []
        for pno in range(1, doc.page_count + 1):
            img = out / "figures" / f"page_{pno:02d}_220dpi.png"
            try:
                txt = subprocess.run(["tesseract", str(img), "stdout"], capture_output=True, text=True,
                                     timeout=180).stdout
            except (OSError, subprocess.TimeoutExpired):
                txt = ""
            ocr_parts.append(f"<!-- page {pno} -->\n{txt}\n")
        oq = text_quality("".join(ocr_parts), [x.split("-->", 1)[-1] for x in ocr_parts])
        q["ocr_digit_ratio"], q["ocr_common_word_ratio"] = oq["digit_ratio"], oq["common_word_ratio"]
        warn = ("<!-- WARNING: the PDF text layer is garbled (font-encoded). The text below is tesseract OCR of the "
                "220 dpi page renders: verify every number against figures/page_NN_220dpi.png -->\n")
        parts = [parts[0], warn] + ocr_parts
        (out / "TEXT_LAYER_WARNING.md").write_text(
            f"text layer unusable: digit_ratio={q['digit_ratio']}, common_word_ratio={q['common_word_ratio']}; "
            "text.md now holds tesseract OCR of figures/page_NN_220dpi.png; verify numbers against the images\n")
    (out / "text.md").write_text("".join(parts))
    (out / "figures" / "figures.json").write_text(json.dumps(figures, indent=1))
    text_chars = sum(len(p) for p in parts)
    stamped = bool(re.search(r"Authorized licensed use limited to|Downloaded .{0,80} by ", "".join(parts)))
    meta.update(sha256=sha256(dest), pages=doc.page_count, text_chars=text_chars,
                n_figure_files=len([f for f in figures if "file" in f]), extracted_on=dt.date.today().isoformat(),
                **q,
                institutional_stamp=stamped)
    if stamped:
        meta["redistribution"] = "restricted_local_only"
    (out / "source.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", required=True)
    ap.add_argument("--pdf")
    a = ap.parse_args()
    info = extract(a.paper_id, Path(a.pdf) if a.pdf else None)
    print(json.dumps({k: info[k] for k in ("pages", "text_chars", "n_figure_files", "institutional_stamp")}))
    if info["text_chars"] < 2000:
        print("WARNING: very little text extracted (scanned PDF?)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
