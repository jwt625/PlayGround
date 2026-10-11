"""Rasterize the TX SiPh PIC layout (tx_geom.py) into textures for the Blender model (workstream T, DevLog-008).

Run from the project folder:
  uv run --with pillow --with numpy python scripts/assets/optics_tx/make_pic_texture.py assets/components/optics_tx/textures

Writes pic_albedo.png (sRGB), pic_mr.png (G = roughness, B = metallic; glTF packing) and pic_layout_preview.png
(downscaled). Image x = u (along the optical edge, -3.5 -> +3.5 mm), image y = v (depth from the optical edge,
row 0 = optical edge). Markings are parody names only.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tx_geom as g  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "assets/components/optics_tx/textures"
PPM = 600                                   # px per mm (1.67 um / px)
W, H = int(g.PIC_W * PPM), int(g.PIC_L * PPM)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

# layer colours (sRGB) and (roughness, metallic)
C = dict(
    bg=(46, 54, 78), fill=(86, 96, 120), seal=(196, 170, 112), trench=(62, 82, 112), core=(150, 218, 238),
    deep=(26, 30, 44), metal=(214, 180, 104), m1=(184, 124, 86), m2=(206, 176, 116), via=(236, 236, 230),
    htr=(232, 142, 70), res=(150, 64, 56), ge=(92, 70, 118), ge_rim=(170, 150, 200), fid=(222, 226, 232),
    text=(214, 182, 112), ssc=(170, 230, 245),
)
MR = dict(bg=(0.45, 0.25), fill=(0.32, 0.9), seal=(0.3, 1.0), trench=(0.4, 0.2), core=(0.25, 0.2), deep=(0.6, 0.1),
          metal=(0.24, 1.0), m1=(0.3, 1.0), m2=(0.28, 1.0), via=(0.2, 1.0), htr=(0.35, 0.8), res=(0.45, 0.5),
          ge=(0.35, 0.4), ge_rim=(0.3, 0.6), fid=(0.2, 1.0), text=(0.25, 1.0), ssc=(0.25, 0.2))


def P(u, v):
    return ((u + g.HALF_W) * PPM, v * PPM)


def px(mm):
    return max(1, int(round(mm * PPM)))


img = Image.new("RGB", (W, H), C["bg"])
mr = Image.new("RGB", (W, H), (0, int(MR["bg"][0] * 255), int(MR["bg"][1] * 255)))
keep = Image.new("L", (W, H), 0)            # dummy-fill keep-out mask
dA, dM, dK = ImageDraw.Draw(img), ImageDraw.Draw(mr), ImageDraw.Draw(keep)


def mrc(layer):
    r, m = MR[layer]
    return (0, int(r * 255), int(m * 255))


def line(pts, w_mm, layer, ko=0.02):
    q = [P(*p) for p in pts]
    dA.line(q, fill=C[layer], width=px(w_mm), joint="curve")
    dM.line(q, fill=mrc(layer), width=px(w_mm), joint="curve")
    if ko:
        dK.line(q, fill=255, width=px(w_mm + 2 * ko), joint="curve")


def poly(pts, layer, ko=0.02):
    q = [P(*p) for p in pts]
    dA.polygon(q, fill=C[layer])
    dM.polygon(q, fill=mrc(layer))
    if ko:
        dK.polygon(q, fill=255)
        dK.line(q + [q[0]], fill=255, width=px(2 * ko))


def rect(u0, v0, u1, v1, layer, ko=0.02, outline=None):
    poly([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], layer, ko)
    if outline:
        q = [P(u0, v0), P(u1, v1)]
        dA.rectangle([min(q[0][0], q[1][0]), min(q[0][1], q[1][1]), max(q[0][0], q[1][0]), max(q[0][1], q[1][1])],
                     outline=C[outline], width=2)


# ---- dummy-fill keep-outs: die margin (seal-ring zone), SSC/facet zone
dK.rectangle([0, 0, W, px(0.42)], fill=255)
for (x0, y0, x1, y1) in ((0, 0, px(0.13), H), (W - px(0.13), 0, W, H), (0, H - px(0.13), W, H)):
    dK.rectangle([x0, y0, x1, y1], fill=255)

# ---- seal ring (double), opened across the facet zone so no metal crosses the edge couplers
for inset, w in ((0.035, 0.012), (0.060, 0.008)):
    x0, y0, x1, y1 = px(inset), px(inset), W - px(inset), H - px(inset)
    dA.rectangle([x0, y0, x1, y1], outline=C["seal"], width=px(w))
    dM.rectangle([x0, y0, x1, y1], outline=mrc("seal"), width=px(w))
gap0, gap1 = P(-2.85, 0.0), P(2.85, 0.075)
dA.rectangle([gap0[0], gap0[1], gap1[0], gap1[1]], fill=C["bg"])
dM.rectangle([gap0[0], gap0[1], gap1[0], gap1[1]], fill=mrc("bg"))

# ---- optical layer: deep-trench facet zone, waveguides (partial-etch trench + core), MMIs, SSCs
dA.rectangle([0, 0, W, px(0.02)], fill=C["deep"])
wg, mmi, ssc = g.waveguides()
for p in wg:
    line(p, 0.010, "trench", ko=0.02)
for p in wg:
    line(p, 0.0035, "core", ko=0)
for (u0, v0, u1, v1) in mmi:
    rect(u0 - 0.004, v0 - 0.004, u1 + 0.004, v1 + 0.004, "trench", ko=0.02)
    rect(u0, v0, u1, v1, "core", ko=0)
for (u, v0, v1) in ssc:                      # inverse taper: narrow tip at the facet (drawn 2.5 um -> 7 um)
    poly([(u - 0.0012, v0), (u + 0.0012, v0), (u + 0.0035, v1), (u - 0.0035, v1)], "ssc", ko=0.03)
    rect(u - 0.03, 0.0, u + 0.03, 0.012, "deep", ko=0)          # facet trench notch

# ---- RF metal (GSSG electrodes, feeds, pads), terminations
polys, res, rfpads = g.rf_metal()
for pl in polys:
    poly(pl, "metal", ko=0.015)
for (u0, v0, u1, v1) in res:
    rect(u0, v0, u1, v1, "res", ko=0.01)
for (u0, v0, u1, v1) in rfpads:
    rect(u0, v0, u1, v1, "metal", ko=0.02)
    rect(u0 + 0.012, v0 + 0.012, u1 - 0.012, v1 - 0.012, "via", ko=0)   # pad opening
    rect(u0 + 0.016, v0 + 0.016, u1 - 0.016, v1 - 0.016, "metal", ko=0)

# ---- thermal phase shifters, DC routing, Ge monitor PDs
for p in g.heaters():
    line(p, 0.006, "htr", ko=0.012)
r = g.dc_routing()
for p in r["m1"]:
    line(p, 0.006, "m1", ko=0.012)
for p in r["m2"]:
    line(p, 0.008, "m2", ko=0.012)
for (u, v) in r["vias"]:
    rect(u - 0.006, v - 0.006, u + 0.006, v + 0.006, "via", ko=0.01)
for (u0, v0, u1, v1, net) in r["pads"]:
    rect(u0, v0, u1, v1, "metal", ko=0.02)
    rect(u0 + 0.012, v0 + 0.012, u1 - 0.012, v1 - 0.012, "via", ko=0)
    rect(u0 + 0.016, v0 + 0.016, u1 - 0.016, v1 - 0.016, "metal", ko=0)
for (u0, v0, u1, v1) in r["ge"]:
    rect(u0, v0, u1, v1, "ge_rim", ko=0.015)
    rect(u0 + 0.008, v0 + 0.008, u1 - 0.008, v1 - 0.008, "ge", ko=0)

# ---- fiducials: corner crosses, circle marks, verniers, edge arrows at lane 1 / lane 8
f = g.fiducials()
for (u, v, a) in f["crosses"]:
    rect(u - a, v - 0.006, u + a, v + 0.006, "fid", ko=0.03)
    rect(u - 0.006, v - a, u + 0.006, v + a, "fid", ko=0.03)
    for sx in (-1, 1):
        for sy in (-1, 1):
            rect(u + sx * a * 0.55 - 0.012, v + sy * a * 0.55 - 0.012, u + sx * a * 0.55 + 0.012, v + sy * a * 0.55 + 0.012,
                 "fid", ko=0.01)
for (u, v, rr) in f["circles"]:
    q0, q1 = P(u - rr, v - rr), P(u + rr, v + rr)
    dA.ellipse([q0, q1], outline=C["fid"], width=px(0.008))
    dM.ellipse([q0, q1], outline=mrc("fid"), width=px(0.008))
    dK.ellipse([P(u - rr - 0.03, v - rr - 0.03), P(u + rr + 0.03, v + rr + 0.03)], fill=255)
    rect(u - rr * 0.6, v - 0.004, u + rr * 0.6, v + 0.004, "fid", ko=0)
    rect(u - 0.004, v - rr * 0.6, u + 0.004, v + rr * 0.6, "fid", ko=0)
for (u, v) in f["verniers"]:
    for i in range(11):
        rect(u + i * 0.020, v, u + i * 0.020 + 0.005, v + 0.035, "fid", ko=0.02)
        rect(u + i * 0.021, v + 0.045, u + i * 0.021 + 0.005, v + 0.080, "fid", ko=0.02)
for u in (g.OUT_U[0], g.OUT_U[-1]):
    poly([(u, 0.30), (u - 0.025, 0.36), (u + 0.025, 0.36)], "fid", ko=0.02)

# ---- markings (parody), lane labels
font_big = ImageFont.truetype(FONT, px(0.16))
font_mid = ImageFont.truetype(FONT, px(0.10))
font_small = ImageFont.truetype(FONT, px(0.040))


def text(u, v, s, font, layer="text", anchor="la"):
    """Glyphs are mirrored in image space: viewed from the PIC face (+X) with the optical edge up, screen right = -u,
    so mirrored glyphs read correctly on the model."""
    q = P(u, v)
    bb = dA.textbbox(q, s, font=font, anchor=anchor)
    pad = 4
    box = (int(bb[0]) - pad, int(bb[1]) - pad, int(bb[2]) + pad, int(bb[3]) + pad)
    w, h = box[2] - box[0], box[3] - box[1]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((q[0] - box[0], q[1] - box[1]), s, fill=255, font=font, anchor=anchor)
    m = m.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(Image.new("RGB", (w, h), C[layer]), box[:2], m)
    mr.paste(Image.new("RGB", (w, h), mrc(layer)), box[:2], m)
    dK.rectangle([bb[0] - px(0.02), bb[1] - px(0.02), bb[2] + px(0.02), bb[3] + px(0.02)], fill=255)


text(1.55, 0.98, "ROCKDOVE", font_big)
text(1.55, 1.20, "PHOTONICS", font_mid)
text(1.55, 1.36, "PGN-8TX  r2  DR8", font_mid)
for k, u in enumerate(g.OUT_U):
    text(u - 0.055, 0.47, str(k + 1), font_small, "fid", anchor="mm")
text(g.LASER_U[0], 0.47, "LD1", font_small, "fid", anchor="mm")
text(g.LASER_U[1], 0.47, "LD2", font_small, "fid", anchor="mm")
for k, uk in enumerate(g.LANE_U):
    text(uk, 7.06, "TX%d" % (k + 1), font_small, "fid", anchor="mm")
# die ID matrix (decorative 2D code, left of centre)
rng = np.random.default_rng(1311)
for i in range(14):
    for j in range(14):
        if rng.random() < 0.5 or i in (0, 13) and j % 2 == 0 or j == 0:
            u0, v0 = -2.75 + i * 0.016, 1.05 + j * 0.016
            rect(u0, v0, u0 + 0.014, v0 + 0.014, "text", ko=0.02)
text(-2.75, 1.31, "ID 2026-1311", font_small)

# ---- dummy fill: 6 um squares on a 14 um grid outside the keep-outs
pitch, sq = 0.014, 0.006
ko = np.array(keep) > 0
ys, xs = np.mgrid[0:H, 0:W]
cell = ((xs % px(pitch)) < px(sq)) & ((ys % px(pitch)) < px(sq))
# a fill square is dropped if any of its four corners touches a keep-out
sqp = px(sq) - 1
ko_any = ko | np.roll(ko, -sqp, axis=1) | np.roll(ko, -sqp, axis=0) | np.roll(np.roll(ko, -sqp, axis=0), -sqp, axis=1)
fill_mask = cell & ~ko_any
a = np.array(img)
b = np.array(mr)
a[fill_mask] = C["fill"]
b[fill_mask] = mrc("fill")
img = Image.fromarray(a)
mr = Image.fromarray(b)

os.makedirs(OUT, exist_ok=True)
img.save(os.path.join(OUT, "pic_albedo.png"), optimize=True)
mr.save(os.path.join(OUT, "pic_mr.png"), optimize=True)
# preview as seen from the PIC face (+X) with the optical edge up: mirror back so it reads like the render
img.resize((W // 4, H // 4), Image.LANCZOS).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(OUT, "pic_layout_preview.png"))
print("PIC_TEXTURE_OK", W, H, "px", PPM, "px/mm", "fill_px", int(fill_mask.sum()))
