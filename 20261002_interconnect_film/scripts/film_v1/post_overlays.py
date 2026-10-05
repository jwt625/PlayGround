"""Burn post-production overlays (FX notes, timecodes, source footnote cards) onto a clean film render.

Scenes built with FILM_HUD=0 render clean and write scenes/v1/<scene>.blend.overlays.json (asm.finalize). This script
draws those texts with Pillow and overlays them with ffmpeg at their film times (scene n starts at 10*(n-1) s).
Run: uv run --no-project --with pillow python scripts/film_v1/post_overlays.py <clean.mp4> <out.mp4> [--kinds CARD,FX,TC]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
# kind: (rgb, size as fraction of frame height, anchor corner, wrap chars)
STYLE = {"FX": ((255, 217, 51), 0.024, "tl", 48), "CARD": ((217, 217, 217), 0.021, "bl", 62), "TC": ((153, 153, 153), 0.021, "br", 20)}

ap = argparse.ArgumentParser()
ap.add_argument("clean")
ap.add_argument("out")
ap.add_argument("--kinds", default="CARD,FX,TC")
ap.add_argument("--glob", default=os.path.join(PROJ, "scenes", "v1", "s0*_*.blend.overlays.json"))
a = ap.parse_args()
kinds = set(a.kinds.split(","))
w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                          "-of", "csv=p=0", a.clean]).decode().strip().split(","))
items = []
for jp in sorted(glob.glob(a.glob)):
    n = int(re.match(r"s0(\d)_", os.path.basename(jp)).group(1))
    for o in json.load(open(jp))["overlays"]:
        if o["kind"] in kinds:
            items.append((n, o))
tmp = tempfile.mkdtemp(prefix="post_ovl_")
inputs, chain, last = [], [], "0:v"
for i, (n, o) in enumerate(items):
    rgb, frac, corner, wrap = STYLE[o["kind"]]
    size = max(9, int(frac * h))
    font = ImageFont.truetype(FONT, size)
    text = "\n".join(sum((textwrap.wrap(ln, wrap) or [""] for ln in o["text"].split("\n")), []))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t_, r, b = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=2)
    im = Image.new("RGBA", (r - l + 4, b - t_ + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).multiline_text((2 - l, 2 - t_), text, font=font, fill=rgb + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    png = os.path.join(tmp, "o%03d.png" % i)
    im.save(png)
    m = int(0.02 * w)
    x = m if corner[1] == "l" else w - im.width - m
    y = m if corner[0] == "t" else h - im.height - m
    t0, t1 = 10 * (n - 1) + o["t0"], 10 * (n - 1) + o["t1"]
    inputs += ["-i", png]
    lab = "v%d" % i
    chain.append("[%s][%d:v]overlay=%d:%d:enable='between(t,%.3f,%.3f)'[%s]" % (last, i + 1, x, y, t0, t1 - 1e-3, lab))
    last = lab
if not items:
    raise SystemExit("no overlays found (build scenes with asm.finalize to write <blend>.overlays.json)")
fg = os.path.join(tmp, "graph.txt")
open(fg, "w").write(";".join(chain))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.clean] + inputs + ["-filter_complex_script", fg, "-map", "[%s]" % last, "-map", "0:a?",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "copy", "-movflags", "+faststart", a.out], check=True)
print("wrote", a.out, "with", len(items), "overlays")
