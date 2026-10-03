"""Procedural texture sequences for the crude film (numpy only; PNGs written with zlib).

- eye sequences: simulated NRZ eye on the scope screen, new random draw every frame (so it shakes),
  with a Gaussian-estimate BER computed from the same simulated samples and printed on the screen.
- graph sequence: whiteboard curves (energy per bit and latency) revealed over time.
- spectrum sequence: ring transmission (Lorentzian dips) measured one after another with a narrow spec window.
Illustrative models: no measured data is used.
"""
import math
import os
import struct
import zlib

import numpy as np

GLYPHS = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    ".": "00000 00000 00000 00000 00000 01100 01100",
    "+": "00000 00100 00100 11111 00100 00100 00000",
    "/": "00001 00010 00010 00100 01000 01000 10000",
    "<": "00010 00100 01000 10000 01000 00100 00010",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def draw_text(img, x, y, text, scale, color):
    """Bitmap text, top-left at (x, y) in pixel coords (row 0 = top)."""
    h, w, _ = img.shape
    cx = x
    for ch in text:
        rows = GLYPHS.get(ch, GLYPHS[" "]).split()
        for ry, row in enumerate(rows):
            for rx, bit in enumerate(row):
                if bit == "1":
                    x0, y0 = cx + rx * scale, y + ry * scale
                    x1, y1 = min(x0 + scale, w), min(y0 + scale, h)
                    if 0 <= x0 < w and 0 <= y0 < h:
                        img[y0:y1, x0:x1] = color
        cx += 6 * scale


def fmt_ber(ber):
    if ber < 1e-30:
        return "<1.0E-30"
    e = int(math.floor(math.log10(ber)))
    m = ber / 10 ** e
    if round(m, 1) >= 10.0:
        m, e = 1.0, e + 1
    return "%.1fE%+03d" % (m, e)


# ------------------------------------------------------------------ eye
def rll_bits(rng, n, maxrun=2):
    """Random +-1 bits with a maximum run length (keeps the eye one coherent set of traces as it closes)."""
    b = np.empty(n, dtype=int)
    b[0] = rng.integers(0, 2) * 2 - 1
    run = 1
    for k in range(1, n):
        if run >= maxrun:
            b[k] = -b[k - 1]
            run = 1
        else:
            b[k] = rng.integers(0, 2) * 2 - 1
            run = run + 1 if b[k] == b[k - 1] else 1
    return b


def eye_frame(rng, sigma_g, noise=0.06, jitter=0.035, W=384, H=240, sps=48, n_ui=240):
    """One scope frame: random NRZ through a Gaussian-limited channel, folded to 2 UI. Returns (rgb, ber, q)."""
    n = n_ui + 12
    bits = rll_bits(rng, n, maxrun=3)
    x = np.repeat(bits.astype(float), sps)
    sg = max(sigma_g * sps, 1.0)
    kk = np.arange(-int(4 * sg), int(4 * sg) + 1)
    g = np.exp(-0.5 * (kk / sg) ** 2)
    g /= g.sum()
    y = np.convolve(x, g, mode="same")
    y = y + rng.normal(0, noise, y.size)
    ks = np.arange(4, n - 6)
    shifts = np.rint(rng.normal(0, jitter * sps, ks.size)).astype(int)
    starts = ks * sps + shifts
    idx = starts[:, None] + np.arange(2 * sps)[None, :]
    tr = y[idx]  # (m, 2*sps)
    # upsample x4 linearly for a continuous trace look
    f = np.arange(2 * sps * 4 - 4) / 4.0
    i0 = np.floor(f).astype(int)
    wgt = f - i0
    fine = tr[:, i0] * (1 - wgt) + tr[:, i0 + 1] * wgt
    px = np.clip((f / (2 * sps) * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(((1.0 - (fine + 1.5) / 3.0) * (H - 1)).astype(int), 0, H - 1)
    hist = np.bincount((py * W + px[None, :]).ravel(), minlength=W * H).reshape(H, W).astype(float)
    # glow
    hp = np.pad(hist, 1)
    hist = (hist * 4 + hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:]) / 8.0
    inten = np.clip(hist / max(hist.max() * 0.30, 1e-9), 0, 1)
    img = np.zeros((H, W, 3), dtype=np.float32)
    # grid
    for gx in range(0, W, W // 8):
        img[:, gx] = (0.10, 0.18, 0.12)
    for gy in range(0, H, H // 6):
        img[gy, :] = (0.10, 0.18, 0.12)
    img[:, W // 2] = (0.14, 0.26, 0.17)
    col = np.stack([inten ** 2.0 * 0.9, inten ** 0.55, inten ** 3.0 * 0.4], axis=-1)
    img = np.clip(img + col, 0, 1)
    # Gaussian-estimate BER from the sampled bit centers (same samples as the trace)
    cs = ks * sps + sps // 2 + shifts
    s = y[cs]
    b = bits[ks]
    s1, s0 = s[b > 0], s[b < 0]
    q = (s1.mean() - s0.mean()) / (s1.std() + s0.std() + 1e-12)
    ber = 0.5 * math.erfc(q / math.sqrt(2.0))
    out = (img * 255).astype(np.uint8)
    txt = "BER " + fmt_ber(ber)
    colr = (255, 90, 70) if ber > 1e-4 else ((255, 220, 80) if ber > 1e-9 else (120, 255, 150))
    draw_text(out, 10, 8, txt, 4, colr)
    return out, ber, q


def eye_sequence(outdir, name, params, seed=1, **kw):
    """Write frames name_0001.png ...; params = per-frame dict(sigma_g, noise, jitter). Returns list of BER."""
    os.makedirs(outdir, exist_ok=True)
    bers = []
    for k, pr in enumerate(params):
        rng = np.random.default_rng(seed * 100003 + k)
        rgb, ber, q = eye_frame(rng, pr["sigma_g"], noise=pr["noise"], jitter=pr["jitter"], **kw)
        write_png(os.path.join(outdir, "%s_%04d.png" % (name, k + 1)), rgb)
        bers.append(ber)
    return bers


# ------------------------------------------------------------------ whiteboard graph
def _disk(img, cx, cy, r, color):
    h, w, _ = img.shape
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[y0:y1, x0:x1][m] = color


GRAPH_L, GRAPH_R, GRAPH_T, GRAPH_B = 70, 640 - 30, 30, 360 - 50


def _curves(p):
    xs = np.linspace(0, 1, 240)
    return xs, [15.6 * xs ** 1.7 / 17.0, 0.95 * xs ** 2.4]   # energy per bit (0 -> ~15.6 pJ/b), latency (arbitrary)


def graph_tip(p):
    """Pixel coords (x, y) of the newest point of each curve at progress p."""
    xs, ys = _curves(p)
    n = max(int(len(xs) * max(p, 0.0)), 1)
    out = []
    for y in ys:
        out.append((int(GRAPH_L + (GRAPH_R - GRAPH_L) * xs[n - 1]), int(GRAPH_B - (GRAPH_B - GRAPH_T) * y[n - 1])))
    return out


def graph_frame(p, W=640, H=360):
    img = np.full((H, W, 3), 246, dtype=np.uint8)
    L, R, T, B = GRAPH_L, GRAPH_R, GRAPH_T, GRAPH_B
    img[T:B + 1, L - 2:L + 1] = (20, 20, 20)
    img[B:B + 3, L:R] = (20, 20, 20)
    xs, ys = _curves(p)
    for color, y in zip(((200, 30, 30), (30, 70, 200)), ys):
        n = int(len(xs) * max(p, 0.0))
        for k in range(n):
            _disk(img, int(L + (R - L) * xs[k]), int(B - (B - T) * y[k]), 3, color)
        if n > 0:
            _disk(img, int(L + (R - L) * xs[n - 1]), int(B - (B - T) * y[n - 1]), 7, color)
    return img


def graph_sequence(outdir, name, progress):
    os.makedirs(outdir, exist_ok=True)
    for k, p in enumerate(progress):
        write_png(os.path.join(outdir, "%s_%04d.png" % (name, k + 1)), graph_frame(p))


# ------------------------------------------------------------------ ring spectrum
def spectrum_frame(shown, specs, active_age, W=480, H=270, win=0.25, xr=6.0):
    img = np.zeros((H, W, 3), dtype=np.float32)
    for gx in range(0, W, W // 12):
        img[:, gx] = (0.10, 0.16, 0.20)
    for gy in range(0, H, H // 6):
        img[gy, :] = (0.10, 0.16, 0.20)
    xs = np.linspace(-xr, xr, W)
    # spec window
    x0 = int((-win + xr) / (2 * xr) * (W - 1))
    x1 = int((win + xr) / (2 * xr) * (W - 1))
    img[:, x0:x1 + 1] += np.array([0.0, 0.28, 0.08], dtype=np.float32)
    img[:, x0] = (0.2, 1.0, 0.3)
    img[:, x1] = (0.2, 1.0, 0.3)
    npass = 0
    for k in range(shown):
        c, a, gam, ok = specs[k]
        npass += 1 if ok else 0
        tr = 1.0 - a / (1.0 + ((xs - c) / gam) ** 2)
        py = np.clip(((1.0 - (0.12 + 0.68 * tr)) * (H - 1)).astype(int), 0, H - 1)
        age = shown - 1 - k
        if age <= active_age:
            color = np.array([1.0, 1.0, 1.0])
        elif ok:
            color = np.array([0.2, 1.0, 0.3])
        else:
            color = np.array([0.8, 0.2, 0.15]) * (0.3 if age > 6 else 0.75)
        for dy in (-1, 0, 1):
            img[np.clip(py + dy, 0, H - 1), np.arange(W)] = np.maximum(
                img[np.clip(py + dy, 0, H - 1), np.arange(W)], color)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    draw_text(out, 10, 8, "%d/%d" % (npass, shown), 4, (120, 255, 150))
    return out


def spectrum_sequence(outdir, name, nframes, n_traces=30, hold_from=0.7, seed=5):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    specs = []
    for k in range(n_traces):
        ok = (k % 10 == 5)  # exactly 1 in 10 pass
        if ok:
            c = rng.uniform(-0.2, 0.2)
        else:
            c = rng.choice([-1, 1]) * rng.uniform(0.6, 5.5)
        specs.append((c, rng.uniform(0.6, 0.95), rng.uniform(0.25, 0.45), ok))
    last = int(nframes * hold_from)
    for k in range(nframes):
        shown = min(n_traces, 1 + int(k * n_traces / max(last, 1)))
        age = (k % max(1, int(last / n_traces)))
        write_png(os.path.join(outdir, "%s_%04d.png" % (name, k + 1)),
                  spectrum_frame(shown, specs, active_age=0 if age < 2 else -1))
