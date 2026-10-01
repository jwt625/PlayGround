"""Extract page-delimited text and figure images from a source PDF into references/<paper_id>/.

Outputs (all inside the git-ignored cache):
  source.pdf (copied if --pdf is outside the cache), source.json, text.md, figures/*.png, figures/figures.json

Figures: embedded raster images >= 150 px on both sides, plus a 150 dpi render of every page that contains a
figure caption ("Fig." / "Figure"), so vector plots can be inspected and digitized.

Usage: uv run python scripts/extract_source.py --paper-id kohli2025 --pdf /path/file.pdf \
         --url https://... --doi 10.xxxx --license CC-BY-4.0 [--local-origin "path or corpus name"]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
CAPTION_RE = re.compile(r"^\s*(Fig\.?|Figure|FIG\.?)\s*(S?\d+)", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(pdf: Path, out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    dest_pdf = out / "source.pdf"
    if pdf.resolve() != dest_pdf.resolve():
        shutil.copyfile(pdf, dest_pdf)
    doc = fitz.open(dest_pdf)
    header = (
        "---\n"
        f"paper_id: {meta['paper_id']}\n"
        f"source_url: {meta.get('url', '')}\n"
        f"doi: {meta.get('doi', '')}\n"
        f"license: {meta.get('license', '')}\n"
        f"sha256: {sha256(dest_pdf)}\n"
        f"pages: {doc.page_count}\n"
        f"extracted_on: {dt.date.today().isoformat()}\n"
        "extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized\n"
        "---\n\n"
    )
    parts = [header]
    figures: list[dict[str, Any]] = []
    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text")
        parts.append(f"<!-- page {pno} -->\n{text}\n")
        captions = [ln.strip() for ln in text.splitlines() if CAPTION_RE.match(ln)]
        if captions:
            pix = page.get_pixmap(dpi=150)
            name = f"page_{pno:02d}.png"
            pix.save(out / "figures" / name)
            figures.append({"kind": "page_render", "file": name, "page": pno, "captions": captions})
        for n, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                pm = fitz.Pixmap(doc, xref)
                if pm.width < 150 or pm.height < 150:
                    continue
                if pm.n - pm.alpha >= 4:
                    pm = fitz.Pixmap(fitz.csRGB, pm)
                name = f"img_p{pno:02d}_{n}.png"
                pm.save(out / "figures" / name)
                figures.append({"kind": "embedded", "file": name, "page": pno, "width": pm.width, "height": pm.height})
            except (RuntimeError, ValueError) as exc:  # unsupported colorspace etc.
                figures.append({"kind": "embedded_failed", "page": pno, "xref": xref, "error": str(exc)})
    (out / "text.md").write_text("".join(parts))
    (out / "figures" / "figures.json").write_text(json.dumps(figures, indent=1))
    info = {
        **meta,
        "sha256": sha256(dest_pdf),
        "pages": doc.page_count,
        "text_chars": sum(len(p) for p in parts),
        "n_figure_files": len([f for f in figures if "file" in f]),
        "extracted_on": dt.date.today().isoformat(),
    }
    (out / "source.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--doi", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--local-origin", default="")
    a = ap.parse_args()
    out = ROOT / "references" / a.paper_id
    info = extract(
        Path(a.pdf),
        out,
        {"paper_id": a.paper_id, "url": a.url, "doi": a.doi, "license": a.license, "local_origin": a.local_origin},
    )
    print(json.dumps(info))
    if info["text_chars"] < 2000:
        print("WARNING: very little text extracted (scanned PDF?)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
