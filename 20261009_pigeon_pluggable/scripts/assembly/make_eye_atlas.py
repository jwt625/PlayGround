"""Varying PAM4 eye-diagram atlas for the test-station monitor (workstream M, DevLog-019 v4).
Approach copied from the interconnect film's scope generator (20261002_interconnect_film/scripts/tex_gen.py, eye_frame):
random symbols through a Gaussian-limited channel + additive noise + per-UI timing jitter, folded to 2 UI, histogram with
a small glow kernel. Changed here: 4 levels (PAM4, 3 eyes), amber colour, per-frame parameter wander (noise / jitter /
amplitude breathing, periodic over N frames so the atlas loops), trace persistence (exponential decay of earlier frames,
pre-rolled one period so the loop is seamless). Synthetic display only, no measured data, no numbers on screen.
Output: one atlas PNG (COLS x ROWS cells, row-major from the top-left), cell 384 x 256.
Usage: uv run --with numpy --with pillow python scripts/assembly/make_eye_atlas.py <out.png> [n_frames=24]"""
import math
import sys

import numpy as np
from PIL import Image

W, H, SPS, N_UI = 384, 256, 48, 260
COLS = 6


def frame_hist(rng, amp, sigma_g, noise, jitter):
    n = N_UI + 12
    sym = rng.integers(0, 4, n)
    lev = np.array([-1.0, -1 / 3, 1 / 3, 1.0]) * amp
    x = np.repeat(lev[sym], SPS)
    sg = max(sigma_g * SPS, 1.0)
    kk = np.arange(-int(4 * sg), int(4 * sg) + 1)
    g = np.exp(-0.5 * (kk / sg) ** 2)
    g /= g.sum()
    y = np.convolve(x, g, mode="same") + rng.normal(0, noise, x.size)
    ks = np.arange(4, n - 6)
    shifts = np.rint(rng.normal(0, jitter * SPS, ks.size)).astype(int)
    starts = ks * SPS + shifts + SPS // 2                       # eye centred in the 2-UI window
    idx = starts[:, None] + np.arange(2 * SPS)[None, :]
    tr = y[idx]
    f = np.arange(2 * SPS * 4 - 4) / 4.0
    i0 = np.floor(f).astype(int)
    w = f - i0
    fine = tr[:, i0] * (1 - w) + tr[:, i0 + 1] * w
    px = np.clip((f / (2 * SPS) * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(((1.0 - (fine + 1.35) / 2.7) * (H - 1)).astype(int), 0, H - 1)
    hist = np.bincount((py * W + px[None, :]).ravel(), minlength=W * H).reshape(H, W).astype(float)
    hp = np.pad(hist, 1)
    return (hist * 4 + hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:]) / 8.0


def render(acc):
    inten = np.clip(acc / max(np.percentile(acc[acc > 0], 99.5) * 0.9, 1e-9), 0, 1)
    img = np.zeros((H, W, 3), dtype=np.float32)
    for gx in range(0, W, W // 8):
        img[:, gx] = (0.09, 0.08, 0.06)
    for gy in range(0, H, H // 6):
        img[gy, :] = (0.09, 0.08, 0.06)
    img[:, W // 2] = (0.13, 0.11, 0.08)
    col = np.stack([inten ** 0.6, inten ** 1.0 * 0.72, inten ** 2.2 * 0.28], axis=-1)
    return np.clip(img + col, 0, 1)


def main(out, n):
    rows = math.ceil(n / COLS)
    atlas = np.zeros((rows * H, COLS * W, 3), dtype=np.float32)
    acc = np.zeros((H, W))
    decay = 0.55                                                  # persistence: earlier frames fade by this per frame
    for k in range(2 * n):                                        # first pass = pre-roll for a seamless loop
        ph = 2 * math.pi * (k % n) / n
        amp = 1.0 + 0.035 * math.sin(ph) + 0.015 * math.sin(3 * ph + 1.0)
        noise = 0.03 + 0.014 * (0.5 + 0.5 * math.sin(2 * ph + 0.4))
        jitter = 0.022 + 0.012 * (0.5 + 0.5 * math.sin(ph + 2.1))
        rng = np.random.default_rng(1009 + k % n)
        acc = acc * decay + frame_hist(rng, amp, 0.17, noise, jitter)
        if k >= n:
            i = k - n
            r, c = divmod(i, COLS)
            atlas[r * H:(r + 1) * H, c * W:(c + 1) * W] = render(acc)
    Image.fromarray((atlas * 255).astype(np.uint8)).save(out)
    print("wrote", out, "frames", n, "grid", COLS, "x", rows, "cell", W, H)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
