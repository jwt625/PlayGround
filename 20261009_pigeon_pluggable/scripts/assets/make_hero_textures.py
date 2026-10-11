"""Draw the hero decal textures (silkscreen, tented vias, fiducials, test points, ENIG traces, laser markings).

Run from the project folder after a build has written the layout:
  uv run --with pillow python scripts/assets/make_hero_textures.py assets/hero/textures
Reads <dir>/decal_layout.json (decal-space mm, origin bottom-left) and writes <dir>/<decal>.png (RGBA, transparent
where there is no ink). Then re-run the build so the decals are mapped. Text is generic or parody only.
"""
import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

DIR = sys.argv[1]
LAY = json.load(open(os.path.join(DIR, "decal_layout.json")))
SANS = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
SILK = (232, 232, 228, 255)
VIA_D, VIA_R = (22, 25, 27, 255), (58, 62, 64, 255)
COPPER, GOLD, CLEAR = (205, 150, 95, 255), (226, 182, 92, 255), (6, 8, 10, 255)


def draw_pcb(name, L):
    w_mm, h_mm = L["size_mm"]
    ppm = min(80.0, 1400.0 / max(w_mm, h_mm))
    W, H = int(round(w_mm * ppm)), int(round(h_mm * ppm))
    P = lambda u, v: (u * ppm, H - v * ppm)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    rng = random.Random(hash(name) & 0xFFFF)

    def disk(u, v, d, col):
        x, y = P(u, v)
        r = d * ppm / 2
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col)

    def text(t, u, v, hmm, ang=0, col=SILK, font=SANS, anchor="mm"):
        f = ImageFont.truetype(font, max(6, int(hmm * ppm)))
        if ang == 0:
            dr.text(P(u, v), t, font=f, fill=col, anchor=anchor)
            return
        tw = int(f.getlength(t)) + 4
        th = int(hmm * ppm * 1.4) + 4
        tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((tw / 2, th / 2), t, font=f, fill=col, anchor="mm")
        tile = tile.rotate(ang, expand=True)
        x, y = P(u, v)
        img.alpha_composite(tile, (int(x - tile.width / 2), int(y - tile.height / 2)))

    for (u0, v0, u1, v1) in L.get("vias", []):
        pitch = 0.55
        nu, nv = max(1, int((u1 - u0) / pitch)), max(1, int((v1 - v0) / pitch))
        for i in range(nu):
            for j in range(nv):
                if rng.random() < 0.35:
                    continue
                u = u0 + (i + 0.5) * (u1 - u0) / nu + rng.uniform(-0.06, 0.06)
                v = v0 + (j + 0.5) * (v1 - v0) / nv + rng.uniform(-0.06, 0.06)
                disk(u, v, 0.44, VIA_R)
                disk(u, v, 0.30, VIA_D)
    if L.get("ring_vias"):
        d = L["ring_vias"]
        poly = L["outline"]
        cu = sum(p[0] for p in poly) / len(poly)
        cv = sum(p[1] for p in poly) / len(poly)
        for i in range(len(poly)):
            (a0, b0), (a1, b1) = poly[i], poly[(i + 1) % len(poly)]
            seg = math.hypot(a1 - a0, b1 - b0)
            for k in range(int(seg / 0.8)):
                t = (k + 0.5) / int(seg / 0.8)
                u, v = a0 + (a1 - a0) * t, b0 + (b1 - b0) * t
                du, dv = cu - u, cv - v
                n = math.hypot(du, dv) or 1.0
                disk(u + du / n * d, v + dv / n * d, 0.42, VIA_R)
                disk(u + du / n * d, v + dv / n * d, 0.28, VIA_D)
    for tr in L.get("enig", []):
        dr.line([P(u, v) for u, v in tr], fill=GOLD, width=max(2, int(0.10 * ppm)), joint="curve")
    for (u0, v0, u1, v1, ref) in L.get("rects", []):
        dr.rectangle([P(u0, v1), P(u1, v0)], outline=SILK, width=max(1, int(0.11 * ppm)))
        text(ref, u0 + 0.05, v1 + 0.35, 0.45, anchor="lm")
        x, y = P(u0 + 0.25, v1 - 0.25)                                      # pin-1 mark
        dr.ellipse([x - 0.09 * ppm, y - 0.09 * ppm, x + 0.09 * ppm, y + 0.09 * ppm], fill=SILK)
    for (u, v, r, ref) in L.get("circles", []):
        x, y = P(u, v)
        dr.ellipse([x - r * ppm, y - r * ppm, x + r * ppm, y + r * ppm], outline=SILK, width=max(1, int(0.11 * ppm)))
        text(ref, u + r * 0.8, v + r + 0.3, 0.45, anchor="lm")
    for (u, v) in L.get("fiducials", []):
        disk(u, v, 2.0, CLEAR)
        disk(u, v, 1.0, COPPER)
    for (u, v, lab) in L.get("tps", []):
        disk(u, v, 0.9, GOLD)
        if lab:
            text(lab, u + 0.65, v, 0.4, anchor="lm")
    for (t, u, v, hmm, ang) in L.get("texts", []):
        text(t, u, v, hmm, ang)
    # clip to the board outline
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([P(u, v) for u, v in L["outline"]], fill=255)
    a = img.getchannel("A")
    img.putalpha(Image.fromarray(__import__("numpy").minimum(__import__("numpy").array(a), __import__("numpy").array(mask))))
    img.save(os.path.join(DIR, name + ".png"))
    return W, H


def draw_marking(name, L):
    w_mm, h_mm = L["size_mm"]
    ppm = min(160.0, 1024.0 / max(w_mm, h_mm))
    W, H = int(round(w_mm * ppm)), int(round(h_mm * ppm))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    ink = tuple(L.get("ink", [150, 150, 150])) + (255,)
    lines = L.get("lines", [])
    if lines:
        hl = min(h_mm / (len(lines) + 1.2), 1.6)
        for i, t in enumerate(lines):
            f = ImageFont.truetype(MONO, max(6, int(hl * 0.62 * ppm)))
            while f.getlength(t) > W * 0.9 and f.size > 6:
                f = ImageFont.truetype(MONO, f.size - 1)
            y = H / 2 + (i - (len(lines) - 1) / 2) * hl * ppm
            dr.text((W / 2, y), t, font=f, fill=ink, anchor="mm")
    if L.get("pin1"):
        r = 0.35 * ppm
        dr.ellipse([0.8 * ppm - r, 0.8 * ppm - r, 0.8 * ppm + r, 0.8 * ppm + r], fill=ink)
    img.save(os.path.join(DIR, name + ".png"))
    return W, H


for name, L in LAY.items():
    W, H = (draw_marking if L.get("kind") == "marking" else draw_pcb)(name, L)
    print("TEX %-40s %4dx%-4d" % (name, W, H))
print("TEXTURES_DONE", len(LAY))
