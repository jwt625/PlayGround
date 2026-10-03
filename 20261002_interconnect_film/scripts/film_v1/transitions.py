"""Scope transitions for the v1.2 film: six scene boundaries, one oscilloscope family (DevLog/v1/DevLog-005-transitions.md).

Each transition is a 30-frame (1.0 s) clip that REPLACES film frames [b*30-15, b*30+15) for boundary time b = 10, 20, ..., 60 s:
clip frames 0..14 show the end of scene N (its last 0.5 s, in order), clip frames 15..29 show the first 0.5 s of scene N+1.
The scene pictures are not overlaid afterwards: they are the texture of the 3D oscilloscope screen, so the camera can pull
out of / push into the screen with real perspective. Film length and scene timing are untouched.

Modes (run from the repo root; numpy only, no downloads):
  tex     uv run --no-project --with numpy python scripts/film_v1/transitions.py tex --out DIR --scene-glob 'outputs/v1/s{n:02d}_*_v1_1_draft_p50.mp4' [--s7-glob G] [--only 1,2]
          scene frames -> RGBA screen textures DIR/t{k}_{i:04d}.png (RGB = phosphor picture, A = glow boost) + DIR/tex_info.json
  build   Blender -b --python scripts/film_v1/transitions.py -- build [scenes/v1/transitions.blend]
          scope asset + backdrop + lights + six animated cameras (timeline markers) + bezel gags
  render  Blender -b scenes/v1/transitions.blend --python scripts/film_v1/transitions.py -- render TEXDIR OUTDIR W H [k,k,...]
          renders PNG frames OUTDIR/t{k}_{i:04d}.png (i = 1..30) at W x H
scripts/film_v1/compose_film.py drives all of this and assembles the film.
"""
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PROJ = os.path.dirname(SCRIPTS)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

NF = 30                      # frames per transition clip (1.0 s at 30 fps)
HALF = NF // 2               # clip frames 0..14 = end of scene N, 15..29 = start of scene N+1
# oscilloscope asset geometry (assets/components/lab_office/bench_oscilloscope.json, metres, scope faces -Y)
SCREEN_C = (-0.062, -0.079, 0.134)
SCREEN_W, SCREEN_H = 0.256, 0.160
LENS, SENSOR = 50.0, 36.0    # camera: vertical sensor fit, so frame height at distance d is SENSOR * d / LENS
D_FULL = LENS * SCREEN_H / SENSOR * 0.996   # camera distance at which the picture rectangle fills the frame (0.4 percent inset)

# ============================================================================== per-boundary specification
# sweep1 / sweep2: (u0, u1) time window (u = i / 29) of the trace sweep that wipes picture N out / picture N+1 in
SPEC = {
    1: dict(tag="s1_s2_retime", world="eye", tint=(0.35, 1.0, 0.40), sweep1=(0.14, 0.40), sweep2=(0.56, 0.84),
            rolls=[(0.10, 0.20, 0.5)], glitch=[3, 4, 12, 13, 17], dmax=0.62, lat=0.10, elev=0.06, roll=0.05, shake=0.0008),
    2: dict(tag="s2_s3_xy_package", world="xy", tint=(0.30, 0.95, 0.55), sweep1=(0.14, 0.40), sweep2=(0.64, 0.88), m=(0.12, 0.66),
            rolls=[(0.45, 0.55, 0.35)], glitch=[2, 5, 15], dmax=0.80, lat=-0.12, elev=0.08, roll=-0.06, shake=0.0008),
    3: dict(tag="s3_s4_heat_spikes", world="heat", tint=(0.35, 1.0, 0.30), sweep1=(0.14, 0.38), sweep2=(0.60, 0.84),
            rolls=[(0.62, 0.72, 0.4)], glitch=[8, 9, 10, 11, 18, 19], dmax=0.70, lat=0.14, elev=-0.05, roll=0.04, shake=0.0035),
    4: dict(tag="s4_s5_ring_dips", world="dips", tint=(0.35, 0.90, 1.0), sweep1=(0.14, 0.40), sweep2=(0.56, 0.84),
            rolls=[(0.08, 0.16, 0.4)], glitch=[4, 22], dmax=0.78, lat=-0.10, elev=0.10, roll=0.22, shake=0.0008),
    5: dict(tag="s5_s6_dips_to_fibers", world="fibers", tint=(0.30, 0.95, 0.95), sweep1=(0.12, 0.36), sweep2=(0.60, 0.84),
            rolls=[(0.30, 0.42, 0.6), (0.48, 0.56, 0.3)], glitch=[6, 7, 8, 14, 20], dmax=0.74, lat=0.30, elev=0.04, roll=-0.10, shake=0.0015),
    6: dict(tag="s6_s7_flatline", world="flat", tint=(0.35, 1.0, 0.40), sweep1=(0.0, 0.0), sweep2=(0.60, 0.88),
            rolls=[], glitch=[], dmax=0.95, lat=0.0, elev=0.05, roll=0.0, shake=0.0005),
}

# ============================================================================== glyphs (5 x 7 bitmap font)
try:
    from tex_gen import GLYPHS as _G0
except Exception:  # pragma: no cover
    _G0 = {}
GLYPHS = dict(_G0)
GLYPHS.update({
    "A": "01110 10001 10001 11111 10001 10001 10001", "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111", "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001", "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000", "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100", "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "'": "00100 00100 00000 00000 00000 00000 00000", ":": "00000 01100 01100 00000 01100 01100 00000",
    "!": "00100 00100 00100 00100 00100 00000 00100",
})


# ============================================================================== numpy helpers
def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def _box(a, r, axis):
    if r <= 0:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    p = np.concatenate([np.zeros((r,) + a.shape[1:], a.dtype), a, np.zeros((r,) + a.shape[1:], a.dtype)], axis=0)
    c = np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), np.cumsum(p, axis=0, dtype=np.float32)], axis=0)
    out = (c[2 * r + 1:2 * r + 1 + n] - c[:n]) / (2 * r + 1)
    return np.moveaxis(out.astype(np.float32), 0, axis)


def gauss(a, sigma):
    """Gaussian blur approximated by three box passes per axis (sigma in pixels)."""
    if sigma < 0.4:
        return a
    w = math.sqrt(4 * sigma * sigma + 1)
    r = max(int(round((w - 1) / 2)), 1)
    for ax in (0, 1):
        for _ in range(3):
            a = _box(a, r, ax)
    return a


def resize(img, H, W):
    """Bilinear resize of (h, w, c) float or uint8 image."""
    h, w = img.shape[:2]
    if (h, w) == (H, W):
        return img.astype(np.float32)
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - np.floor(ys)).astype(np.float32)[:, None, None]
    wx = (xs - np.floor(xs)).astype(np.float32)[None, :, None]
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def write_png_rgba(path, rgba):
    import struct
    import zlib
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


# ============================================================================== canvas: phosphor strokes in "H units"
class Canvas:
    """Draws polylines / dots / text on the screen texture. World units: x in 0..1.6 and y in 0..1 (up), 1 unit = Ht pixels."""

    def __init__(self, Wt, Ht):
        self.W, self.H = Wt, Ht
        self.S = Ht / 1350.0
        self.groups = {}
        self.text_mask = {}

    def _acc(self, key):
        if key not in self.groups:
            self.groups[key] = np.zeros((self.H, self.W), np.float32)
        return self.groups[key]

    def splat(self, key, px, py, wgt):
        acc = self._acc(key).reshape(-1)
        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        fx, fy = px - x0, py - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx < self.W) & (yy >= 0) & (yy < self.H)
            acc += np.bincount(yy[ok] * self.W + xx[ok], weights=(w * wgt)[ok], minlength=self.H * self.W).astype(np.float32)

    def poly(self, pts, color, reveal=1.0, width=1.0, gain=1.0, head=True, closed=False):
        """pts (N,2) in world units. reveal in 0..1 draws only the first fraction (beam head gets a bright spot)."""
        pts = np.asarray(pts, np.float64)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        P = np.stack([pts[:, 0] * self.H, (1.0 - pts[:, 1]) * self.H], axis=1)
        seg = np.hypot(*np.diff(P, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        tot = cum[-1]
        if tot <= 0:
            return 0.0
        n = max(int(tot / 0.6), 2)
        s = np.linspace(0, tot * float(np.clip(reveal, 0, 1)), n)
        x = np.interp(s, cum, P[:, 0])
        y = np.interp(s, cum, P[:, 1])
        self.splat((tuple(color), float(width)), x, y, np.full(n, gain))
        if head and 0.0 < reveal < 1.0:
            self.dot(x[-1] / self.H, 1.0 - y[-1] / self.H, color, 1.0, 2.5)
        return tot

    def dot(self, xh, yh, color, rad=1.0, gain=1.0):
        px, py = xh * self.H, (1.0 - yh) * self.H
        r = int(12 * self.S * rad) + 2
        x0, x1 = max(int(px) - r, 0), min(int(px) + r + 1, self.W)
        y0, y1 = max(int(py) - r, 0), min(int(py) + r + 1, self.H)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * (4.0 * self.S * rad) ** 2)).astype(np.float32)
        self._acc(("dot", tuple(color)))[y0:y1, x0:x1] += g * gain

    def text(self, xh, yh, s, size, color, gain=1.0):
        """Bitmap text, top-left at world (xh, yh); size = glyph height in world units."""
        cell = max(int(round(size * self.H / 7.0)), 1)
        m = self.text_mask.setdefault(tuple(color), np.zeros((self.H, self.W), np.float32))
        x = int(xh * self.H)
        y = int((1.0 - yh) * self.H)
        for ch in s:
            rows = GLYPHS.get(ch, GLYPHS.get(" ", "00000 " * 7)).split()
            for ry, row in enumerate(rows):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        xa, ya = x + rx * cell, y + ry * cell
                        if 0 <= ya < self.H - cell and 0 <= xa < self.W - cell:
                            m[ya:ya + cell, xa:xa + cell] = gain
            x += 6 * cell

    def render(self, sigma=1.5):
        rgb = np.zeros((self.H, self.W, 3), np.float32)
        S = self.S
        for key, acc in self.groups.items():
            if key[0] == "dot":
                col = np.array(key[1], np.float32)
                rgb += gauss(acc, 1.0)[..., None] * col
                continue
            col, width = key
            col = np.array(col, np.float32)
            sc = sigma * S * width + 0.35
            core = gauss(acc, sc) * (sc * 2.5066 * 0.6)
            halo = gauss(acc, sc * 5.0) * (sc * 5.0 * 2.5066 * 0.6)
            rgb += (core + 0.22 * halo)[..., None] * col
        for col, m in self.text_mask.items():
            col = np.array(col, np.float32)
            rgb += (gauss(m, 0.8 * S + 0.3) * 1.2 + 0.15 * gauss(m, 6 * S))[..., None] * col
        return rgb


def tone(rgb, k=1.5):
    return 1.0 - np.exp(-k * np.clip(rgb, 0, None))


# ============================================================================== scope world content per boundary
def eye_world(Wt, Ht, i, m, flash_out):
    """T1: closed eye that opens in three retimer steps (tex_gen.eye_frame: same simulation as the S1/S2 scope screens)."""
    import tex_gen as T
    th = (0.12, 0.40, 0.68)
    lv = float(sum(smoothstep(t, t + 0.07, m) for t in th))
    fr = lv / 3.0
    rng = np.random.default_rng(900 + i)
    img, ber, q = T.eye_frame(rng, 0.45 - 0.07 * fr, noise=0.30 - 0.25 * fr, jitter=0.12 - 0.09 * fr)
    world = resize(img, Ht, Wt) / 255.0
    flash_out.append(float(sum(np.exp(-((m - t - 0.03) / 0.035) ** 2) for t in th)))
    cv = Canvas(Wt, Ht)
    k = int(round(lv))
    if k > 0:
        cv.text(1.18, 0.95, "RETIMER %d" % k, 0.045, (0.4, 1.0, 0.5))
    rgbt = tone(cv.render())
    lum = world.max(axis=2)
    boost = np.clip((lum - 0.45) * 1.4, 0, 1) * 0.5
    return np.clip(world + rgbt, 0, 1), boost + rgbt.max(axis=2) * 0.7, True


def curve_pts(f, n=240):
    xs = np.linspace(0, 1, n)
    return xs, f(xs)


def package_outline():
    """T2: XY-mode drawing of an NPO-style package: substrate, ASIC, 2 x 4 optical engine blocks, laser diode and beam. World units."""
    cx, cy = 0.80, 0.50
    R = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float) + [cx, cy]
    P = [R(-0.38, -0.34, 0.38, 0.34),                       # substrate
         R(-0.14, -0.14, 0.14, 0.14)]                       # ASIC
    for sx in (-1, 1):                                        # optical engines beside the ASIC on the substrate
        for k in range(4):
            x0 = sx * 0.22 - 0.04 + (0.0)
            y0 = -0.27 + k * 0.14
            P.append(R(x0 - 0.0, y0, x0 + 0.08, y0 + 0.10))
    # laser diode symbol (triangle + bar) left of the package with a beam into the substrate edge
    lx, ly = cx - 0.62, cy
    P.append(np.array([[lx - 0.06, ly - 0.05], [lx + 0.04, ly], [lx - 0.06, ly + 0.05], [lx - 0.06, ly - 0.05]], float))
    P.append(np.array([[lx + 0.04, ly - 0.05], [lx + 0.04, ly + 0.05]], float))
    P.append(np.array([[lx + 0.04, ly], [cx - 0.38, ly], [cx - 0.42, ly + 0.025], [cx - 0.38, ly], [cx - 0.42, ly - 0.025]], float))
    return P


def xy_world(Wt, Ht, i, m, flash_out):
    cv = Canvas(Wt, Ht)
    # curves (energy per bit, latency: same shapes as tex_gen._curves, scope channel colours) collapse to a dot, then XY mode draws the package
    c = float(smoothstep(0.30, 0.48, m))
    f = float(np.clip((m - 0.50) / 0.42, 0, 1))
    xs = np.linspace(0, 1, 240)
    ys = [15.6 * xs ** 1.7 / 17.0, 0.95 * xs ** 2.4]
    cols = [(1.0, 0.85, 0.2), (0.3, 0.9, 1.0)]
    for y, col in zip(ys, cols):
        if c < 1.0:
            px = 0.30 + 1.0 * xs
            py = 0.12 + 0.76 * y
            qx = px * (1 - c) + 0.80 * c
            qy = py * (1 - c) + 0.50 * c
            cv.poly(np.stack([qx, qy], 1), col, reveal=1.0, gain=1.0, head=False)
    P = package_outline()
    lens_ = [np.hypot(*np.diff(p, axis=0).T).sum() for p in P]
    tot = sum(lens_)
    run = 0.0
    for p, L_ in zip(P, lens_):
        rv = np.clip((f * tot - run) / L_, 0, 1)
        run += L_
        if rv > 0:
            cv.poly(p, (0.35, 1.0, 0.55), reveal=float(rv), width=1.0, gain=1.0, head=True)
    if 0.45 < m < 1.0:
        cv.text(1.28, 0.95, "XY", 0.06, (1.0, 0.85, 0.3))
    flash_out.append(float(np.exp(-((m - 0.49) / 0.03) ** 2)))
    return tone(cv.render()), None, False


def heat_world(Wt, Ht, i, m, flash_out):
    cv = Canvas(Wt, Ht)
    rng = np.random.default_rng(31)
    xs = np.linspace(0.0, 1.6, 640)
    base = 0.22 + 0.30 * m * smoothstep(0.0, 1.6, xs) ** 0.7
    sp = np.sort(rng.uniform(0.15, 1.5, 9))
    hs = np.linspace(0.18, 0.62, 9) * rng.uniform(0.85, 1.1, 9)
    y = base.copy()
    for j, (x0, h) in enumerate(zip(sp, hs)):
        a = float(smoothstep(0.08 * j, 0.08 * j + 0.22, m))
        dx = xs - x0
        sh = np.where(dx < 0, np.exp(-(dx / 0.012) ** 2), np.exp(-dx / 0.09))
        y += a * h * sh * (dx > -0.06)
    y += rng.normal(0, 0.004 + 0.012 * m, xs.size)
    y = np.clip(y, 0.02, 0.97)
    col = tuple(np.array([0.35, 1.0, 0.30]) * (1 - m) + np.array([1.0, 0.25, 0.12]) * m)
    cv.poly(np.stack([xs, y], 1), col, gain=1.0, head=False)
    cv.poly(np.array([[0.0, 0.80], [1.6, 0.80]]), (1.0, 0.2, 0.15), gain=0.0, head=False)
    # dashed limit line
    for k in range(0, 32):
        cv.poly(np.array([[0.05 * k, 0.80], [0.05 * k + 0.03, 0.80]]), (0.9, 0.2, 0.15), gain=0.8, width=0.7, head=False)
    cv.text(0.06, 0.95, "TEMP", 0.06, col)
    flash_out.append(0.0)
    return tone(cv.render()), None, False


def dips_spec(N=6, seed=5):
    rng = np.random.default_rng(seed)
    cs = np.sort(rng.uniform(0.15, 1.45, N))
    cs[2] = 0.80  # the one that lands in the pass window
    return cs, rng.uniform(0.28, 0.5, N), rng.uniform(0.010, 0.018, N)


def dips_curve(xs, cs, dep, gam, amp):
    y = 0.72 + 0.0 * xs
    for c, d, g, a in zip(cs, dep, gam, amp):
        y = y - a * d / (1.0 + ((xs - c) / g) ** 2)
    return y


def dips_world(Wt, Ht, i, m, flash_out):
    cv = Canvas(Wt, Ht)
    cs, dep, gam = dips_spec()
    rng = np.random.default_rng(77 + i)
    xs = np.linspace(0.0, 1.6, 900)
    N = len(cs)
    amp = [float(smoothstep(0.85 * k / N, 0.85 * k / N + 0.16, m)) for k in range(N)]
    drift = 0.012 * np.sin(6.0 * m + np.arange(N))
    y = dips_curve(xs, cs + drift * (1 - np.array(amp) * 0.0), dep, gam, amp) + rng.normal(0, 0.003, xs.size)
    # pass window around the centre dip (as in the S5 spectrum screen)
    xw0, xw1 = 0.76, 0.84
    band = np.zeros((Ht, Wt), np.float32)
    band[:, int(xw0 * Ht):int(xw1 * Ht)] = 1.0
    cv.poly(np.array([[xw0, 0.0], [xw0, 1.0]]), (0.2, 1.0, 0.3), gain=0.7, width=0.6, head=False)
    cv.poly(np.array([[xw1, 0.0], [xw1, 1.0]]), (0.2, 1.0, 0.3), gain=0.7, width=0.6, head=False)
    cv.poly(np.stack([xs, y], 1), (0.55, 0.95, 1.0), gain=1.0, head=False)
    k_active = int(np.clip(math.floor(m * N / 0.85), 0, N - 1))
    cv.text(0.06, 0.95, "%d/%d" % (min(N, int(sum(a > 0.5 for a in amp))), N), 0.06, (0.4, 1.0, 0.5))
    rgb = tone(cv.render()) + band[..., None] * np.array([0.0, 0.07, 0.02], np.float32)
    flash_out.append(0.0)
    return rgb, None, False


def fibers_world(Wt, Ht, i, m, flash_out):
    cv = Canvas(Wt, Ht)
    cs, dep, gam = dips_spec()
    N = len(cs)
    xs = np.linspace(0.0, 1.6, 900)
    a = float(smoothstep(0.0, 0.40, m))        # dips narrow and deepen into vertical lines
    b = float(smootherstep((m - 0.38) / 0.40))  # lines rotate to horizontal
    cc = float(smoothstep(0.66, 1.0, m))       # colour turns fibre cyan, strands start to wiggle
    colA = np.array([0.55, 0.95, 1.0])
    colB = np.array([0.1, 0.95, 1.0])
    col = tuple(colA * (1 - cc) + colB * cc)
    if b < 0.02:
        y = dips_curve(xs, cs, dep * (1 + 1.5 * a), gam * (1 - 0.8 * a), [1.0] * N)
        y = np.clip(y, 0.02, 0.97)
        cv.poly(np.stack([xs, y], 1), col, gain=1.0, head=False)
    else:
        for k in range(N):
            xk = cs[k]
            yk = 0.5 + (k - (N - 1) / 2.0) * 0.095
            th = b * math.pi / 2
            L0, L1 = 0.92, 1.05
            half = (L0 * (1 - b) + L1 * b) / 2
            cxk = xk * (1 - b) + 0.80 * b
            cyk = 0.5 * (1 - b) + yk * b
            tt = np.linspace(-1, 1, 160)
            px = cxk + tt * half * math.sin(th)
            py = cyk + tt * half * math.cos(th)
            wig = cc * 0.012 * np.sin(2 * math.pi * (tt * 1.5 + 0.8 * m * 6 + k * 0.7)) * (1 - abs(tt) ** 3)
            py = py + wig * (b > 0.5)
            cv.poly(np.stack([px, py], 1), col, gain=1.0, head=False, width=1.0 + 0.4 * cc)
    flash_out.append(0.0)
    return tone(cv.render()), None, False


def flat_world(Wt, Ht, i, m, flash_out):
    flash_out.append(0.0)
    return np.zeros((Ht, Wt, 3), np.float32), None, False


WORLDS = dict(eye=eye_world, xy=xy_world, heat=heat_world, dips=dips_world, fibers=fibers_world, flat=flat_world)


# ============================================================================== screen texture frame
def graticule(Wt, Ht, tint):
    g = np.zeros((Ht, Wt, 3), np.float32)
    S = Ht / 1350.0
    col = np.array(tint, np.float32) * 0.24
    step = Ht / 10.0
    for k in range(0, 17):
        x = int(round(k * step))
        if x < Wt:
            g[:, x] = np.maximum(g[:, x], col)
    for k in range(0, 11):
        y = min(int(round(k * step)), Ht - 1)
        g[y, :] = np.maximum(g[y, :], col)
    cx, cy = int(round(Wt / 2)), int(round(Ht / 2))
    g[:, cx] = np.maximum(g[:, cx], col * 1.6)
    g[cy, :] = np.maximum(g[cy, :], col * 1.6)
    sub = step / 5.0  # minor ticks on the centre axes
    for k in range(int(Wt / sub) + 1):
        x = int(round(k * sub))
        if x < Wt:
            g[max(cy - int(3 * S) - 1, 0):cy + int(3 * S) + 2, x] = col * 1.6
    for k in range(int(Ht / sub) + 1):
        y = int(round(k * sub))
        if y < Ht:
            g[y, max(cx - int(3 * S) - 1, 0):cx + int(3 * S) + 2] = col * 1.6
    return g


def env(u):
    """CRT effect envelope: zero on the first and last frame (so the clip meets the scenes), full in between."""
    return float(smoothstep(0.0, 0.10, u) * smoothstep(1.0, 0.90, u))


def sweep_heads(u, spec):
    def head(w):
        a, b = w
        if b <= a:
            return None
        return float(np.clip((u - a) / (b - a), 0, 1))
    return head(spec["sweep1"]), head(spec["sweep2"])


def sweep_line(Wt, Ht, xn, color):
    """Bright vertical sweep line at normalised screen x; returns (rgb, boost)."""
    S = Ht / 1350.0
    X = (np.arange(Wt, dtype=np.float32) - xn * Wt)
    p = np.exp(-0.5 * (X / (2.2 * S + 0.5)) ** 2) + 0.35 * np.exp(-0.5 * (X / (22 * S)) ** 2) + 0.12 * np.exp(-0.5 * (X / (90 * S)) ** 2)
    line = p[None, :, None] * np.array(color, np.float32)[None, None, :] * np.ones((Ht, 1, 1), np.float32)
    return line, np.broadcast_to(np.clip(p, 0, 1)[None, :], (Ht, Wt))


def make_frame(k, i, pic1, pic2, Wt, Ht, grat):
    """RGBA float frame (Ht, Wt, 4) of the scope screen for transition k, clip frame i."""
    spec = SPEC[k]
    u = i / (NF - 1.0)
    e = env(u)
    Wp = pic1.shape[1]
    x0 = (Wt - Wp) // 2
    tint = np.array(spec["tint"], np.float32)
    S = Ht / 1350.0
    rng = np.random.default_rng(1000 * k + i)
    # morph parameter of the trace world: runs over the apex window
    m0, m1 = spec.get("m", (0.24, 0.76))
    m = float(np.clip((u - m0) / (m1 - m0), 0, 1))
    flash = []
    world, wboost, opaque = WORLDS[spec["world"]](Wt, Ht, i, m, flash)
    base = grat
    trace_rgb = np.zeros_like(grat) if opaque else world
    boost = wboost if wboost is not None else (trace_rgb.max(axis=2) * 0.8)
    h1, h2 = sweep_heads(u, spec)
    X = (np.arange(Wt) + 0.5) / Wt
    img = np.empty((Ht, Wt, 3), np.float32)
    bst = np.zeros((Ht, Wt), np.float32)
    # trace world visible where picture N has been swept away and picture N+1 has not arrived yet
    hh1 = -1.0 if h1 is None else (-1.0 if h1 <= 0 else (2.0 if h1 >= 1 else h1 * 1.04 - 0.02))
    hh2 = -1.0 if h2 is None else (-1.0 if h2 <= 0 else (2.0 if h2 >= 1 else h2 * 1.04 - 0.02))
    if spec["world"] == "flat":
        hh1 = -1.0
    tr_mask = ((X < hh1) if h1 is not None and spec["sweep1"][1] > 0 else np.ones(Wt, bool)) & (X >= hh2)
    if spec["sweep1"][1] <= 0:
        tr_mask = X >= hh2
    pic1m = (X >= hh1) if spec["sweep1"][1] > 0 else np.ones(Wt, bool)
    pic2m = X < hh2
    img[:] = base
    if opaque:
        img[:, tr_mask] = world[:, tr_mask]
        bst += boost * tr_mask[None, :]
        img[:, ~tr_mask] = grat[:, ~tr_mask]
    img += trace_rgb * tr_mask[None, :, None]
    if not opaque:
        bst += boost * tr_mask[None, :]
    # picture N (distortions on the picture only)
    p1 = pic1
    if k == 3 and e > 0 and u < 0.5:      # heat shimmer on the picture before it is swept away
        amp = 0.020 * Wp * float(smoothstep(0.0, 0.35, u))
        ys = np.arange(Ht)
        dx = (amp * np.sin(2 * math.pi * ys / (0.11 * Ht) + 9.0 * u) * (0.5 + 0.5 * np.sin(2 * math.pi * ys / (0.37 * Ht) + 3 * u))).astype(int)
        cols = np.clip(np.arange(Wp)[None, :] + dx[:, None], 0, Wp - 1)
        p1 = np.take_along_axis(pic1, cols[:, :, None].repeat(3, axis=2), axis=1)
    if k == 6:
        # CRT power-off: picture squashes to a bright horizontal line, the line shrinks to a dot, the dot fades, black, card sweeps in
        sy = 1.0 - float(smootherstep((u - 0.04) / 0.22)) ** 1.5
        sx = 1.0 - float(smootherstep((u - 0.30) / 0.14))
        fade = 1.0 - float(smoothstep(0.44, 0.60, u))
        img[:] = 0.0
        if sy > 0.012 and u < 0.34:
            ys = (np.arange(Ht) + 0.5 - Ht / 2) / sy + Ht / 2
            ok = (ys >= 0) & (ys < Ht - 1)
            yi = np.clip(ys.astype(int), 0, Ht - 1)
            band = pic1[yi] * ok[:, None, None]
            heat_gain = 1.0 + 1.6 * (1 - sy)
            img[:, x0:x0 + Wp] = np.clip(band * heat_gain, 0, 1.5)
            bst[:, x0:x0 + Wp] = (1 - sy) * 0.8 * ok[:, None]
        if u >= 0.18:
            half = 0.5 * Wp * max(sx, 0.0) + 2 * S
            thick = (2.0 + 5.0 * (1 - sy)) * S
            core = np.exp(-0.5 * ((np.arange(Ht)[:, None] - Ht / 2) / thick) ** 2)
            xx = np.abs(np.arange(Wt)[None, :] - Wt / 2)
            wid = np.clip((half - xx) / (6 * S) + 0.5, 0, 1) + np.exp(-0.5 * (np.clip(xx - half, 0, None) / (14 * S)) ** 2) * 0.4
            ln = (core * wid * fade * (1.0 if u < 0.44 else 1.0))
            if sx < 0.05:
                dot_r = 10 * S * (1 + 1.5 * (1 - fade))
                yy, xxx = np.mgrid[0:Ht, 0:Wt]
                ln = ln + np.exp(-0.5 * (((yy - Ht / 2) ** 2 + (xxx - Wt / 2) ** 2) / dot_r ** 2)) * 1.4 * fade
            col = np.array([0.8, 1.0, 0.85], np.float32)
            img += (ln[..., None] * col) * (1.0 if sy < 0.05 else 0.0)
            bst += np.clip(ln, 0, 1) * (1.0 if sy < 0.05 else 0.0)
        if u > 0.45 and u < 0.60:   # dark phase: faint 'no signal' text on the blank phosphor
            cv = Canvas(Wt, Ht)
            cv.text(0.80 - 0.30, 0.52, "NO SIGNAL", 0.05, (0.4, 1.0, 0.5), gain=1.0)
            tr = tone(cv.render()) * float(smoothstep(0.45, 0.50, u) * (1 - smoothstep(0.56, 0.60, u)))
            img += tr
            bst += tr.max(axis=2) * 0.6
        # picture N+1 wiped in by the sweep
        if hh2 > -1.0:
            m2 = (np.arange(Wp) + x0 + 0.5) / Wt < hh2
            img[:, x0:x0 + Wp][:, m2] = pic2[:, m2]
            bst[:, x0:x0 + Wp][:, m2] = 0.0
    else:
        sel = pic1m[x0:x0 + Wp]
        img[:, x0:x0 + Wp][:, sel] = p1[:, sel]
        bst[:, x0:x0 + Wp][:, sel] = 0.0
        sel2 = pic2m[x0:x0 + Wp]
        img[:, x0:x0 + Wp][:, sel2] = pic2[:, sel2]
        bst[:, x0:x0 + Wp][:, sel2] = 0.0
    # sweep lines
    lcol = np.array([0.75, 1.0, 0.8], np.float32)
    for hh, w in ((h1, spec["sweep1"]), (h2, spec["sweep2"])):
        if hh is not None and 0.0 < hh < 1.0 and w[1] > w[0]:
            xn = hh * 1.04 - 0.02
            ln, lb = sweep_line(Wt, Ht, xn, lcol)
            img += ln
            bst = np.maximum(bst, lb)
    # ---- CRT artefacts (all zero at u = 0 and u = 1)
    if e > 0:
        # horizontal-sync roll with a dark sync bar
        for (a, b, amp) in spec["rolls"]:
            if a <= u <= b:
                p = (u - a) / (b - a)
                off = int(Ht * amp * math.sin(math.pi * p) ** 2 * (1 if p < 0.5 else 1))
                if off:
                    img = np.roll(img, off, axis=0)
                    bst = np.roll(bst, off, axis=0)
                    yb = off % Ht
                    h_bar = int(0.05 * Ht)
                    rows = (np.arange(yb, yb + h_bar) % Ht)
                    img[rows] *= 0.35
                    img[(rows[0] - 2) % Ht] += 0.25 * lcol
        # horizontal tear on glitch frames
        if i in spec["glitch"]:
            for _ in range(3):
                y0 = int(rng.uniform(0.05, 0.9) * Ht)
                hb = int(rng.uniform(0.01, 0.05) * Ht)
                dx = int(rng.uniform(-0.05, 0.05) * Wt)
                img[y0:y0 + hb] = np.roll(img[y0:y0 + hb], dx, axis=1)
                bst[y0:y0 + hb] = np.roll(bst[y0:y0 + hb], dx, axis=1)
        # trigger flicker / flash from morph events
        fl = 1.0 + e * (0.5 * (flash[0] if flash else 0.0) + (0.10 * rng.standard_normal() if abs(u - 0.5) < 0.30 else 0.0))
        img *= fl
        # chromatic split, scanlines, noise, vignette
        d = int(round(e * 3.5 * S * (1 + (4 if i in spec["glitch"] else 0))))
        if d:
            img[:, d:, 0] = img[:, :-d, 0].copy()
            img[:, :-d, 2] = img[:, d:, 2].copy()
        per = max(Ht / 240.0, 3.0)
        yy = np.arange(Ht, dtype=np.float32)
        scan = 1.0 - 0.28 * e * (0.5 + 0.5 * np.cos(2 * math.pi * yy / per))
        img *= scan[:, None, None]
        img += (rng.standard_normal((Ht, Wt, 1)).astype(np.float32) * 0.035 * e)
        gx = (np.arange(Wt, dtype=np.float32) / Wt - 0.5) * 1.6
        gy = (np.arange(Ht, dtype=np.float32) / Ht - 0.5)
        vig = 1.0 - 0.55 * e * np.clip(gx[None, :] ** 2 / 0.64 + gy[:, None] ** 2 / 0.25 - 0.45, 0, 1)
        img *= vig[..., None]
    rgba = np.empty((Ht, Wt, 4), np.float32)
    rgba[..., :3] = np.clip(img, 0, 1)
    rgba[..., 3] = np.clip(bst, 0, 1)
    return rgba


# ============================================================================== scene frame access
def scene_file(glob_pat, n):
    pat = glob_pat.format(n=n)
    fs = sorted(glob.glob(os.path.join(PROJ, pat) if not os.path.isabs(pat) else pat))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[-1]


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                                   "-of", "csv=p=0", path]).decode().strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def read_frames(path, start, count, W, H):
    """count RGB frames starting at frame index start, scaled to W x H; short files repeat their last frame."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf",
           "trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,scale=%d:%d:flags=lanczos" % (start, start + count, W, H),
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if len(a) < count:
        a = np.concatenate([a, np.repeat(a[-1:], count - len(a), axis=0)]) if len(a) else np.zeros((count, H, W, 3), np.uint8)
    return a


def gen_textures(out_dir, scene_glob, s7_glob=None, only=None):
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    for k in (only or range(1, 7)):
        pa = scene_file(scene_glob, k)
        pb = scene_file(s7_glob if (k == 6 and s7_glob) else scene_glob, k + 1)
        W, H, na = probe(pa)
        Wt = int(round(1.6 * H))
        Wt += (Wt - W) % 2                           # picture centred on whole pixels
        _, _, nb = probe(pb)
        fa = read_frames(pa, na - HALF, HALF, W, H).astype(np.float32) / 255.0
        fb = read_frames(pb, 0, HALF, W, H).astype(np.float32) / 255.0
        grat = graticule(Wt, H, SPEC[k]["tint"])
        for i in range(NF):
            p1 = fa[min(i, HALF - 1)]
            p2 = fb[min(max(i - HALF, 0), HALF - 1)]
            # frames before the half-way point show scene N for both layers (picture N+1 is only used after sweep2 starts)
            if i < HALF:
                p2 = fb[0]
            else:
                p1 = fa[HALF - 1]
            rgba = make_frame(k, i, p1, p2, Wt, H, grat)
            write_png_rgba(os.path.join(out_dir, "t%d_%04d.png" % (k, i + 1)), (rgba * 255 + 0.5).astype(np.uint8))
        info[k] = dict(scene_a=pa, scene_b=pb, pic=[W, H], tex=[Wt, H], scene_a_frames=na, scene_b_frames=nb)
        print("TEX boundary", k, SPEC[k]["tag"], "tex", Wt, "x", H, flush=True)
    p = os.path.join(out_dir, "tex_info.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update({str(a): b for a, b in info.items()})
    json.dump(old, open(p, "w"), indent=1)


# ============================================================================== Blender: build and render
def bump(u, k=1.25):
    v = float(np.clip((1 - abs(2 * u - 1)) * k, 0, 1))
    return float(smootherstep(v))


def cam_state(k, i):
    """Camera position and roll for transition k, clip frame i (scope space, metres, scope faces -Y)."""
    spec = SPEC[k]
    u = i / (NF - 1.0)
    b = bump(u)
    d = D_FULL * (spec["dmax"] / D_FULL) ** b
    lat = spec["lat"] * d * b * math.cos(math.pi * u) * 1.6
    ele = spec["elev"] * d * b * math.sin(math.pi * u) * 1.6
    roll = spec["roll"] * b * math.cos(math.pi * u)
    if k == 5:                                    # whip: extra lateral swing through the apex
        lat += 0.35 * d * math.sin(2 * math.pi * u) * b
    rng = np.random.default_rng(4000 + 31 * k + i)
    e = env(u)
    sh = spec["shake"] * e
    return d, lat + sh * rng.standard_normal(), ele + sh * rng.standard_normal(), roll + 0.4 * sh * rng.standard_normal(), b


def build_blend(out_blend):
    import bpy
    from mathutils import Vector
    sys.path.insert(0, os.path.join(SCRIPTS, "assets", "materials_vfx"))
    import render_presets as RP
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    RP.apply_render_preset(scn, "standard", w=1080, h=1350)
    scn.frame_start, scn.frame_end = 1, NF * 6
    scn.render.use_motion_blur = True
    scn.render.motion_blur_shutter = 0.4
    scn.render.filter_size = 0.8
    # --- oscilloscope asset
    path = os.path.join(PROJ, "assets", "components", "lab_office", "bench_oscilloscope.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [c for c in src.collections if c.startswith("ASSET_") or c.startswith("ITEM_")]
    root = None
    for c in dst.collections:
        scn.collection.children.link(c)
    obs = {o.name: o for o in bpy.data.objects}
    root = obs["ROOT_bench_oscilloscope"]
    # screen material: image sequence (RGB phosphor, alpha = glow boost -> emission strength 1 + 3.5 * alpha)
    mat = next(m for m in bpy.data.materials if m.name.startswith("MAT_lab_office_screen"))
    nt = mat.node_tree
    img_node = nt.nodes["SCREEN_IMAGE"]
    img_node.interpolation = "Linear"
    img_node.extension = "EXTEND"
    nt.nodes["SCREEN_FAC"].outputs[0].default_value = 1.0
    emis = nt.nodes["SCREEN_EMISSION"]
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    ma.inputs[1].default_value = 3.5
    ma.inputs[2].default_value = 1.0
    nt.links.new(img_node.outputs["Alpha"], ma.inputs[0])
    nt.links.new(ma.outputs[0], emis.inputs["Strength"])
    nt.nodes["SCREEN_GLASS"].inputs["Roughness"].default_value = 0.12
    # --- backdrop: bench mat and a dark wall
    def plane(name, loc, rot, size, col, rough=0.8):
        bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot)
        o = bpy.context.active_object
        o.name = name
        m = bpy.data.materials.new(name + "_mat")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = col
        b.inputs["Roughness"].default_value = rough
        o.data.materials.append(m)
        return o
    plane("BENCH", (0, -0.3, 0.0), (0, 0, 0), 4.0, (0.05, 0.16, 0.11, 1.0))
    plane("WALL", (0, 0.55, 1.0), (math.pi / 2, 0, 0), 4.0, (0.045, 0.055, 0.075, 1.0))
    # --- lights
    def area(name, loc, energy, size, col=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = energy, size, col
        o = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(o)
        o.location = loc
        d = Vector(SCREEN_C) - Vector(loc)
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return o
    area("KEY", (-0.45, -0.75, 0.75), 70, 0.6, (1.0, 0.96, 0.9))
    area("FILL", (0.6, -0.8, 0.25), 25, 0.8, (0.8, 0.9, 1.0))
    area("RIM", (0.25, 0.45, 0.65), 45, 0.5, (0.6, 0.8, 1.0))
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.016, 0.022, 1)
    scn.world = w
    # --- cameras (one per boundary, bound to timeline markers) + aim empty
    aim = bpy.data.objects.new("AIM_screen", None)
    aim.location = SCREEN_C
    scn.collection.objects.link(aim)
    for k in range(1, 7):
        cd = bpy.data.cameras.new("CAM_T%d" % k)
        cd.lens, cd.sensor_fit, cd.sensor_height = LENS, "VERTICAL", SENSOR
        cd.clip_start, cd.clip_end = 0.01, 20
        cd.dof.use_dof = True
        cd.dof.focus_object = aim
        cd.dof.aperture_fstop = 5.6
        cam = bpy.data.objects.new("CAM_T%d" % k, cd)
        scn.collection.objects.link(cam)
        base = (k - 1) * NF
        for i in range(NF):
            d, lat, ele, roll, b = cam_state(k, i)
            pos = Vector((SCREEN_C[0] + lat, SCREEN_C[1] - d, SCREEN_C[2] + ele))
            tgt = Vector(SCREEN_C) + Vector((0, 0, -0.03 * b))
            q = (tgt - pos).to_track_quat("-Z", "Y")
            from mathutils import Quaternion
            q = q @ Quaternion((0, 0, 1), roll)
            cam.location = pos
            cam.rotation_mode = "QUATERNION"
            cam.rotation_quaternion = q
            cam.keyframe_insert("location", frame=base + i + 1)
            cam.keyframe_insert("rotation_quaternion", frame=base + i + 1)
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        mk = scn.timeline_markers.new("T%d" % k, frame=base + 1)
        mk.camera = cam
    scn.camera = bpy.data.objects["CAM_T1"]
    # --- bezel gags (knob twists, button press, jolt, probe wiggle, power LED)
    def key(o, path, idx, f, v, interp="BEZIER"):
        setattr(o, path, v) if idx is None else o.__setattr__(path, _set_idx(getattr(o, path), idx, v))
        o.keyframe_insert(path, index=-1 if idx is None else idx, frame=f)
    def _set_idx(vec, idx, v):
        vv = list(vec)
        vv[idx] = v
        return tuple(vv)
    kn_s = obs["bench_oscilloscope_horizontal_scale_knob"]
    kn_p = obs["bench_oscilloscope_horizontal_position_knob"]
    # T1: horizontal scale knob clicks three times with the retimer steps (u = 0.19, 0.47, 0.75 of the clip)
    for (f_u, a) in ((0.0, 0.0), (0.40, 0.0), (0.44, 0.6), (0.62, 0.6), (0.66, 1.2), (0.84, 1.2), (0.88, 1.8), (1.0, 1.8)):
        key(kn_s, "rotation_euler", 1, 1 + int(round(f_u * 29)), a)
    key(kn_s, "rotation_euler", 1, NF + 1, 0.0)
    # T4: horizontal position knob tunes the window (two turns)
    b4 = 3 * NF
    key(kn_p, "rotation_euler", 1, b4 + 1, 0.0)
    key(kn_p, "rotation_euler", 1, b4 + 14, 0.0)
    key(kn_p, "rotation_euler", 1, b4 + 18, 4 * math.pi)
    # T2: AUTOSET button press at the apex
    bt = obs["bench_oscilloscope_autoscale_button"]
    y0 = bt.location.y
    for f, dy in ((NF + 11, 0), (NF + 13, 0.002), (NF + 16, 0.002), (NF + 18, 0)):
        key(bt, "location", 1, f, y0 + dy)
    # T3: whole scope jolt (impact of the brawl and heat spikes)
    b3 = 2 * NF
    key(root, "location", None, b3 + 1, (0.0, 0.0, 0.0))
    for j in range(8, 22):
        rng = np.random.default_rng(j)
        amp = 0.004 * (1 - abs(j - 14) / 8.0)
        root.location = (0.0, 0.0, 0.0)
        key(root, "location", None, b3 + j, tuple(amp * rng.standard_normal(3) * np.array([1, 0.3, 1])))
    key(root, "location", None, b3 + 23, (0, 0, 0))
    # T5: probe cable whip
    pr = obs["bench_oscilloscope_probe_ch1"]
    b5 = 4 * NF
    for j, a in ((0, 0.0), (8, 0.0), (12, 0.35), (16, -0.3), (20, 0.2), (24, 0.0)):
        key(pr, "rotation_euler", 2, b5 + 1 + j, a)
    # T6: power LED off after the flatline and power button press
    led = obs["bench_oscilloscope_power_led_ring"]
    b6 = 5 * NF
    led.hide_render = False
    led.keyframe_insert("hide_render", frame=b6 + 1)
    led.hide_render = True
    led.keyframe_insert("hide_render", frame=b6 + 12)
    pb = obs["bench_oscilloscope_power_button"]
    y1 = pb.location.y
    for f, dy in ((b6 + 7, 0), (b6 + 9, 0.003), (b6 + 12, 0.003), (b6 + 14, 0)):
        key(pb, "location", 1, f, y1 + dy)
    # --- compositor: bloom on the over-bright phosphor only (threshold above 1.0; the picture itself stays <= 1)
    scn.use_nodes = True
    nt = scn.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    gl = nt.nodes.new("CompositorNodeGlare")
    try:
        gl.glare_type = "BLOOM"
    except Exception:
        gl.glare_type = "FOG_GLOW"
    gl.threshold = 1.05
    gl.quality = "HIGH"
    try:
        gl.size = 7
        gl.mix = 0.0
    except Exception:
        pass
    cp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
    nt.links.new(gl.outputs["Image"], cp.inputs["Image"])
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("SAVED", out_blend)


def render(tex_dir, out_dir, W, H, ks):
    import bpy
    scn = bpy.context.scene
    scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = W, H, 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    mat = next(m for m in bpy.data.materials if m.name.startswith("MAT_lab_office_screen"))
    node = mat.node_tree.nodes["SCREEN_IMAGE"]
    os.makedirs(out_dir, exist_ok=True)
    import time
    for k in ks:
        first = os.path.join(tex_dir, "t%d_0001.png" % k)
        img = bpy.data.images.load(first, check_existing=False)
        img.source = "SEQUENCE"
        img.alpha_mode = "CHANNEL_PACKED"
        img.colorspace_settings.name = "sRGB"
        node.image = img
        node.image_user.frame_duration = NF
        node.image_user.frame_start = (k - 1) * NF + 1
        node.image_user.frame_offset = 0
        node.image_user.use_auto_refresh = True
        t0 = time.time()
        for i in range(NF):
            scn.frame_set((k - 1) * NF + i + 1)
            scn.render.filepath = os.path.join(out_dir, "t%d_%04d" % (k, i + 1))
            bpy.ops.render.render(write_still=True)
        print("RENDERED boundary %d in %.1f s (%.2f s/frame) at %dx%d" % (k, time.time() - t0, (time.time() - t0) / NF, W, H), flush=True)
        bpy.data.images.remove(img)


def main():
    argv = sys.argv
    if "--" in argv:
        a = argv[argv.index("--") + 1:]
        try:
            import bpy  # noqa: F401
        except ImportError:
            return
        if a[0] == "build":
            build_blend(a[1] if len(a) > 1 else os.path.join(PROJ, "scenes", "v1", "transitions.blend"))
        elif a[0] == "render":
            ks = [int(x) for x in a[5].split(",")] if len(a) > 5 else list(range(1, 7))
            render(a[1], a[2], int(a[3]), int(a[4]), ks)
        return
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tex"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-glob", required=True)
    ap.add_argument("--s7-glob")
    ap.add_argument("--only")
    a = ap.parse_args()
    gen_textures(a.out, a.scene_glob, a.s7_glob, [int(x) for x in a.only.split(",")] if a.only else None)


if __name__ == "__main__":
    main()
