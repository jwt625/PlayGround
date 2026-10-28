"""Stage 3: upright figure crops + numeral OCR, driven by data/<pn>/figures.json.

Usage: python3 -I pipeline/crop.py <pn> [fig-id ...]
For each figure: crop page `page` (0-based index of work/<pn>/pages/p-NNN.png) to `crop`
[x0, y0, x1, y1] in native page pixels, rotate by `rotate` degrees counter-clockwise
(0/90/180/270) so the figure reads upright, then write
  work/<pn>/figs/fig-<id>.png        upright crop (all trace coordinates use this pixel space)
  work/<pn>/figs/fig-<id>.words.tsv  sparse OCR words: text, x0, y0, x1, y1, conf
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def crop_figure(pn: str, fig: dict) -> Path:
    page = ROOT / "work" / pn / "pages" / f"p-{fig['page']:03d}.png"
    out = ROOT / "work" / pn / "figs"
    out.mkdir(parents=True, exist_ok=True)
    im = Image.open(page).convert("L").crop(tuple(fig["crop"]))
    if fig.get("rotate", 0):
        im = im.rotate(fig["rotate"], expand=True, fillcolor=255)
    dst = out / f"fig-{fig['id']}.png"
    im.save(dst, optimize=True)
    base = out / f"fig-{fig['id']}.ocr"
    subprocess.run(["tesseract", str(dst), str(base), "--psm", "11", "tsv"], check=True, capture_output=True)
    with open(f"{base}.tsv") as f, open(out / f"fig-{fig['id']}.words.tsv", "w") as g:
        g.write("text\tx0\ty0\tx1\ty1\tconf\n")
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            t = (r.get("text") or "").strip()
            if not t or float(r["conf"]) < 0:
                continue
            x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
            g.write(f"{t}\t{x}\t{y}\t{x + w}\t{y + h}\t{float(r['conf']):.0f}\n")
    Path(f"{base}.tsv").unlink()
    print(f"fig {fig['id']}: {im.size[0]}x{im.size[1]} -> {dst.relative_to(ROOT)}")
    return dst


def main() -> None:
    pn, only = sys.argv[1], set(sys.argv[2:])
    figs = json.loads((ROOT / "data" / pn / "figures.json").read_text())["figures"]
    for fig in figs:
        if not only or fig["id"] in only:
            crop_figure(pn, fig)


if __name__ == "__main__":
    main()
