"""Compose final TX optics renders: callout labels, lane tags, corner note (workstream T, DevLog-008).

Run from the project folder:
  uv run --with pillow python scripts/assets/optics_tx/label_overlay_tx.py RAW.png OUT.png [--labels L.json]
      [--tags T.json] [--note TEXT] [--crop]
--labels: callouts in two columns (left / right of the content) with leader lines.
--crop:   crop the raw render to its content (non-background pixels) and add side gutters for the callout columns.
"""
import argparse
import json

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("out")
ap.add_argument("--labels")
ap.add_argument("--note")
ap.add_argument("--tags")
ap.add_argument("--crop", action="store_true")
ap.add_argument("--size", type=int, default=21)
a = ap.parse_args()

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
LINE = (150, 210, 230, 255)
TEXT = (235, 240, 245, 255)
BOX = (8, 12, 18, 215)
img = Image.open(a.raw).convert("RGBA")
font = ImageFont.truetype(FONT, a.size)
offx, offy = 0, 0

if a.crop:
    bg = img.getpixel((2, 2))
    g = img.convert("RGB").point(lambda v: v)
    px = g.load()
    W0, H0 = g.size
    thr = 18
    xs, ys = [], []
    for y in range(0, H0, 3):
        for x in range(0, W0, 3):
            p = px[x, y]
            if abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > thr:
                xs.append(x)
                ys.append(y)
    pad = 40
    x0, x1 = max(min(xs) - pad, 0), min(max(xs) + pad, W0)
    y0, y1 = max(min(ys) - pad, 0), min(max(ys) + pad, H0)
    gut = 420 if a.labels else 0
    canvas = Image.new("RGBA", (x1 - x0 + 2 * gut, y1 - y0 + 40), bg)
    canvas.paste(img.crop((x0, y0, x1, y1)), (gut, 20))
    img = canvas
    offx, offy = gut - x0, 20 - y0

W, H = img.size
ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)

if a.labels:
    rows = json.load(open(a.labels))["labels"]
    cx = sum(r["anchor"][0] + offx for r in rows) / len(rows)
    cols = {"L": [], "R": []}
    for r in rows:
        ax, ay = r["anchor"][0] + offx, r["anchor"][1] + offy
        cols["L" if ax < cx else "R"].append(dict(text=r["text"], ax=ax, ay=ay))
    bh = a.size + 10
    for side, labs in cols.items():
        labs.sort(key=lambda L: L["ay"])
        y_prev = -1e9
        for L in labs:
            L["w"] = font.getlength(L["text"]) + 14
            L["y"] = max(L["ay"] - bh / 2, y_prev + bh + 10, 10)
            y_prev = L["y"]
        over = (labs[-1]["y"] + bh + 10 - H) if labs else 0
        if over > 0:
            for L in labs:
                L["y"] -= over
        for L in labs:
            L["x"] = 14 if side == "L" else W - L["w"] - 14
            ex = L["x"] + L["w"] if side == "L" else L["x"]
            ey = L["y"] + bh / 2
            dr.line([(L["ax"], L["ay"]), (ex, ey)], fill=LINE, width=2)
            dr.ellipse([L["ax"] - 4, L["ay"] - 4, L["ax"] + 4, L["ay"] + 4], fill=LINE)
            dr.rectangle([L["x"], L["y"], L["x"] + L["w"], L["y"] + bh], fill=BOX, outline=LINE, width=1)
            dr.text((L["x"] + 7, L["y"] + 4), L["text"], fill=TEXT, font=font)

if a.tags:
    ft = ImageFont.truetype(FONT, 18)
    placed = []
    for r in json.load(open(a.tags))["labels"]:
        x, y = r["anchor"][0] + offx, r["anchor"][1] + offy
        tw = ft.getlength(r["text"])
        tx0 = min(max(x - tw / 2 - 5, 4), W - tw - 14)
        ty0 = min(max(y - 34, 4), H - 30)
        for _ in range(20):
            hit = [q for q in placed if tx0 < q[2] + 4 and q[0] < tx0 + tw + 14 and ty0 < q[3] + 3 and q[1] < ty0 + 27]
            if not hit:
                break
            ty0 += 28
        placed.append((tx0, ty0, tx0 + tw + 10, ty0 + 24))
        dr.rectangle([tx0, ty0, tx0 + tw + 10, ty0 + 24], fill=BOX, outline=LINE, width=1)
        dr.text((tx0 + 5, ty0 + 2), r["text"], fill=TEXT, font=ft)

if a.note:
    f2 = ImageFont.truetype(FONT, 20)
    tw = f2.getlength(a.note)
    dr.rectangle([W - tw - 30, H - 44, W - 10, H - 10], fill=BOX)
    dr.text((W - tw - 20, H - 39), a.note, fill=TEXT, font=f2)

Image.alpha_composite(img, ov).convert("RGB").save(a.out)
print("OVERLAY_OK", a.out, img.size)
