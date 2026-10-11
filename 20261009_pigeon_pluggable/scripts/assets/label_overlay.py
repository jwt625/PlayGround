"""Draw small text labels with leader lines on the exploded render.

Run from the project folder:
  uv run --with pillow python scripts/assets/label_overlay.py outputs/hero/work/pigeon_osfp_dr8_exploded_raw.png \
      outputs/hero/work/pigeon_osfp_dr8_exploded_labels.json outputs/hero/pigeon_osfp_dr8_exploded.png
Anchors come from render_hero_pigeon.py (3D bbox points projected to pixels). Label boxes start at an offset
away from the anchor cloud centroid (or a manual direction below) and are relaxed apart to avoid overlaps.
"""
import json
import math
import sys

from PIL import Image, ImageDraw, ImageFont

RAW, JS, OUT = sys.argv[1:4]
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
SIZE = 21
LINE = (150, 210, 230, 255)
TEXT = (235, 240, 245, 255)
BOX = (8, 12, 18, 200)
# manual initial directions (unit-ish vectors in pixels) for crowded labels
DIRS = {
    "DSP": (-1.0, -0.9), "thermal pad": (-1.0, -1.3), "MCU": (0.6, -1.0), "EEPROM": (0.2, -1.2),
    "PMIC + buck": (1.0, -0.6), "inductors": (1.2, -0.2), "crystal": (1.2, 0.5), "TX PIC (SiPh)": (-1.2, -0.5),
    "MZM x8": (-1.3, 0.2), "CW DFB x2": (-0.6, -1.3), "PD array (8 ch)": (1.0, 1.0), "TIA (octal)": (1.3, 0.6),
    "TX lenslet array 2x4": (-1.2, -0.4), "RX lenslet array 2x4": (-1.0, 0.8), "prisms 1x8 -> 2x4": (-0.8, -1.2),
    "glass relays": (-0.6, -1.4), "PCB": (1.2, 0.9), "60-pad edge / feet": (1.0, 0.4), "shell (bottom)": (1.0, 1.0),
    "shell (top)": (1.1, -0.3), "fins = feathers": (0.8, -1.0), "beak latch": (-0.6, 1.0), "leash (pull tab)": (0.8, 1.0),
    "EMI fingers = neck ruff": (0.6, 1.2),
}
DIST = 120

d = json.load(open(JS))
img = Image.open(RAW).convert("RGBA")
W, H = img.size
font = ImageFont.truetype(FONT, SIZE)
labs = []
cx = sum(r["anchor"][0] for r in d["labels"]) / len(d["labels"])
cy = sum(r["anchor"][1] for r in d["labels"]) / len(d["labels"])
for r in d["labels"]:
    ax, ay = r["anchor"]
    dx, dy = DIRS.get(r["text"], (ax - cx, ay - cy))
    n = math.hypot(dx, dy) or 1.0
    tw = font.getlength(r["text"])
    bw, bh = tw + 14, SIZE + 10
    x = ax + dx / n * DIST - bw / 2
    y = ay + dy / n * DIST - bh / 2
    labs.append(dict(text=r["text"], ax=ax, ay=ay, x=x, y=y, w=bw, h=bh))


def clamp(L):
    L["x"] = min(max(L["x"], 12), W - L["w"] - 12)
    L["y"] = min(max(L["y"], 12), H - L["h"] - 12)


for it in range(400):
    moved = False
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]) + 6
            oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]) + 4
            if ox > 0 and oy > 0:
                moved = True
                if oy < ox:
                    s = 1 if a["y"] < b["y"] else -1
                    a["y"] -= s * oy / 2
                    b["y"] += s * oy / 2
                else:
                    s = 1 if a["x"] < b["x"] else -1
                    a["x"] -= s * ox / 2
                    b["x"] += s * ox / 2
        clamp(a)
    if not moved:
        break

COLUMNS = len(sys.argv) > 4 and sys.argv[4] == "columns"
if COLUMNS:
    # classic callout columns: labels sorted by anchor height in a left and a right column, elbow leaders
    xs = sorted(L["ax"] for L in labs)
    split = xs[len(xs) // 2]                          # median: balanced columns
    for side_ in ("L", "R"):
        col = sorted([L for L in labs if (L["ax"] < split) == (side_ == "L")], key=lambda L: L["ay"])
        if not col:
            continue
        wmax = max(L["w"] for L in col)
        x_edge = max(wmax + 20, min(L["ax"] for L in labs) - 120) if side_ == "L" else min(W - 20 - wmax, max(L["ax"] for L in labs) + 120)
        gap = 8
        ys = [L["ay"] - L["h"] / 2 for L in col]
        for i in range(1, len(col)):                       # push down to remove overlaps
            ys[i] = max(ys[i], ys[i - 1] + col[i - 1]["h"] + gap)
        shift = max(0, ys[-1] + col[-1]["h"] + 12 - H) if col else 0
        for i in range(len(col)):
            ys[i] -= shift
        for i in range(len(col) - 2, -1, -1):              # then pull up from the bottom if needed
            ys[i] = min(ys[i], ys[i + 1] - col[i]["h"] - gap)
        for L, y in zip(col, ys):
            L["y"] = max(12, y)
            L["x"] = x_edge - L["w"] if side_ == "L" else x_edge
            L["side"] = side_

ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)
for L in labs:
    # leader: anchor -> nearest point on the box edge (elbow in columns mode)
    if COLUMNS:
        ex_ = L["x"] + L["w"] if L["side"] == "L" else L["x"]
        ey_ = L["y"] + L["h"] / 2
        knee = ex_ + (18 if L["side"] == "L" else -18)
        dr.line([(L["ax"], L["ay"]), (knee, ey_), (ex_, ey_)], fill=LINE, width=2)
    else:
        px = min(max(L["ax"], L["x"]), L["x"] + L["w"])
        py = min(max(L["ay"], L["y"]), L["y"] + L["h"])
        dr.line([(L["ax"], L["ay"]), (px, py)], fill=LINE, width=2)
    dr.ellipse([L["ax"] - 4, L["ay"] - 4, L["ax"] + 4, L["ay"] + 4], fill=LINE)
for L in labs:
    dr.rectangle([L["x"], L["y"], L["x"] + L["w"], L["y"] + L["h"]], fill=BOX, outline=LINE, width=1)
    dr.text((L["x"] + 7, L["y"] + 4), L["text"], font=font, fill=TEXT)
out = Image.alpha_composite(img, ov).convert("RGB")
out.save(OUT)
print("LABELS_DONE", OUT, len(labs), "iterations", it + 1)
