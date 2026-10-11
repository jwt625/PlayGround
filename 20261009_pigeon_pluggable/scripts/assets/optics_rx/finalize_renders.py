"""Overlay labels / notes / scale bar on the RX optics raw renders -> outputs/assets/optics_rx/optics_rx_<shot>.png.

  uv run --with pillow python scripts/assets/optics_rx/finalize_renders.py
Reads outputs/assets/optics_rx/work/<shot>_raw.png, <shot>_meta.json, exploded_labels.json.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(ROOT, "outputs", "assets", "optics_rx", "work")
OUT = os.path.join(ROOT, "outputs", "assets", "optics_rx")
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
f_lab = ImageFont.truetype(FONT, 24)
f_note = ImageFont.truetype(FONT, 26)
LINE = (150, 210, 230, 255)
TEXT = (235, 240, 245, 255)
BOX = (8, 12, 18, 210)
BEAM_NOTE = "* beams invisible (1311 nm); rendered for viewing"

CAPTIONS = {
    "overview": "RX eye subassembly, true scale (carrier 6.95 x 6.40 mm)",
    "exploded": None,
    "tia_closeup": "octal TIA: spiral peaking inductors, MIM caps, guard rails, dummy fill (micro-optics hidden)",
    "bond_closeup": "PD (front) -> TIA: differential G-S-S-G bonds, 25 um Au (P->IN+, N->IN-, G to submount GND; micro-optics hidden)",
    "optics_closeup": "2x4 lenslets -> Si row interleavers -> stepped glass turning block (Au 45 deg facets) -> 1x8 PD row (Si drawn translucent)",
    "optics_top": "top view (orthographic): rows shifted -/+0.25 mm by the Si plates, then turned down onto 0.50 mm PD pitch",
}


def box_text(d, xy, txt, font, anchor="la"):
    l, t, r, b = d.textbbox(xy, txt, font=font, anchor=anchor)
    d.rectangle([l - 8, t - 6, r + 8, b + 6], fill=BOX)
    d.text(xy, txt, font=font, fill=TEXT, anchor=anchor)


def labels(img, anchors):
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    left = sorted([a for a in anchors if a["x"] < W * 0.5], key=lambda a: a["y"])
    right = sorted([a for a in anchors if a["x"] >= W * 0.5], key=lambda a: a["y"])
    for col, xs, anc in ((left, 30, "lm"), (right, W - 30, "rm")):
        n = len(col)
        if not n:
            continue
        ys = [max(60, min(H - 60, a["y"])) for a in col]
        gap = 44
        for i in range(1, n):                      # push down to keep spacing
            ys[i] = max(ys[i], ys[i - 1] + gap)
        over = ys[-1] - (H - 60)
        if over > 0:
            ys = [y - over for y in ys]
            for i in range(n - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        # untangle: swap label slots of any two leaders that cross (repeat until none cross)
        def _end(a, y):
            tw = d.textlength(a["label"], font=f_lab)
            return (xs + tw + 10 if anc == "lm" else xs - tw - 10, y)

        def _cross(p1, p2, q1, q2):
            def o(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0

        for _ in range(60):
            swapped = False
            for i in range(n):
                for j in range(i + 1, n):
                    pi, pj = (col[i]["x"], col[i]["y"]), (col[j]["x"], col[j]["y"])
                    if _cross(_end(col[i], ys[i]), pi, _end(col[j], ys[j]), pj):
                        col[i], col[j] = col[j], col[i]
                        swapped = True
            if not swapped:
                break
        placed = []
        for a, y in zip(col, ys):
            tw = d.textlength(a["label"], font=f_lab)
            x_box, x_anc, y_box = xs, anc, y
            covers = (anc == "lm" and xs + tw + 30 > a["x"]) or (anc == "rm" and xs - tw - 30 < a["x"])
            if covers:                                 # column box would hide the part: put the label below-right of it
                x_box, x_anc, y_box = min(a["x"] + 60, W - 40 - tw), "lm", min(a["y"] + 110, H - 60)
            tx_end = x_box + (tw + 10 if x_anc == "lm" else -tw - 10) if not covers else x_box - 10
            placed.append((a, x_box, x_anc, y_box, tx_end))
        for a, x_box, x_anc, y_box, tx_end in placed:   # leaders first, boxes on top
            d.line([(tx_end, y_box), (a["x"], a["y"])], fill=LINE, width=2)
            d.ellipse([a["x"] - 4, a["y"] - 4, a["x"] + 4, a["y"] + 4], fill=LINE)
        for a, x_box, x_anc, y_box, tx_end in placed:
            box_text(d, (x_box, y_box), a["label"], f_lab, anchor=x_anc)


for shot in ["overview", "exploded", "tia_closeup", "bond_closeup", "optics_closeup", "optics_top"]:
    raw = os.path.join(WORK, "%s_raw.png" % shot)
    if not os.path.exists(raw):
        continue
    img = Image.open(raw).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    meta_p = os.path.join(WORK, "%s_meta.json" % shot)
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if shot == "exploded":
        labels(img, json.load(open(os.path.join(WORK, "exploded_labels.json"))))
    cap = CAPTIONS.get(shot)
    if cap:
        box_text(d, (30, 40), cap, f_note)
    if meta.get("beams"):
        box_text(d, (30, H - 40), BEAM_NOTE, f_note, anchor="lb")
    if meta.get("px_per_mm"):
        L = meta["px_per_mm"] * 1.0
        x0, y0 = W - 60 - L, H - 50
        d.rectangle([x0 - 14, y0 - 46, x0 + L + 14, y0 + 16], fill=BOX)
        d.line([(x0, y0), (x0 + L, y0)], fill=TEXT, width=4)
        d.line([(x0, y0 - 10), (x0, y0 + 10)], fill=TEXT, width=3)
        d.line([(x0 + L, y0 - 10), (x0 + L, y0 + 10)], fill=TEXT, width=3)
        d.text((x0 + L / 2, y0 - 14), "1 mm", font=f_lab, fill=TEXT, anchor="mb")
    out = os.path.join(OUT, "optics_rx_%s.png" % shot)
    img.convert("RGB").save(out)
    print("wrote", os.path.relpath(out, ROOT))
