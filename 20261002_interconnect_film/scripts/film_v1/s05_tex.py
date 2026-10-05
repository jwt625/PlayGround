"""S5 v1.2 monitor sequence: ring transmission traces measured one per 0.2 s plus a 10-die map strip (1 in 10 passes).

Illustrative model only (Lorentzian dips, by construction exactly 1 of 10 inside the spec window); no measured data.
Image 480 x 270 (maps 1:1 onto probe_station_cm300_style_screen). Dips point DOWN (transmission on the vertical axis).
Trace k is measured at sequence frame 6k (0.2 s spacing, matching the trace_ping SFX cues 44.8 + 0.2 k film s).
"""
import os

import numpy as np

import tex_gen as T

W, H = 480, 270
N_DIE = 10
PASS_K = 5                 # the one good ring (measured at frame 30 = scene 5.8 s)
STEP = 6                   # frames between traces
PLOT_T, PLOT_B = 40, 176   # plot rows
DIE_Y0, DIE_S, DIE_GAP, DIE_X0 = 196, 40, 6, 13


def _specs(seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(N_DIE):
        if k == PASS_K:
            c = 0.08
        else:
            c = (1 if k % 2 else -1) * rng.uniform(0.9, 4.8)
        out.append((c, rng.uniform(0.70, 0.92), rng.uniform(0.28, 0.42), k == PASS_K))
    return out


def _ring(img, cx, cy, r, th, color, alpha=1.0):
    h, w, _ = img.shape
    y0, y1, x0, x1 = max(cy - r - th, 0), min(cy + r + th + 1, h), max(cx - r - th, 0), min(cx + r + th + 1, w)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    m = np.abs(d - r) <= th / 2.0
    sub = img[y0:y1, x0:x1]
    sub[m] = sub[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha


def frame(i, specs, xr=6.0, win=0.25):
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:] = (0.02, 0.03, 0.05)
    # plot grid and spec window
    for gx in range(0, W, W // 12):
        img[PLOT_T:PLOT_B, gx] = (0.10, 0.16, 0.20)
    for gy in range(PLOT_T, PLOT_B + 1, (PLOT_B - PLOT_T) // 4):
        img[gy, :] = (0.10, 0.16, 0.20)
    xs = np.linspace(-xr, xr, W)
    x0 = int((-win + xr) / (2 * xr) * (W - 1))
    x1 = int((win + xr) / (2 * xr) * (W - 1))
    img[PLOT_T:PLOT_B, x0:x1 + 1] += np.array([0.0, 0.22, 0.06], dtype=np.float32)
    img[PLOT_T:PLOT_B, x0] = (0.2, 0.9, 0.3)
    img[PLOT_T:PLOT_B, x1] = (0.2, 0.9, 0.3)
    shown = min(N_DIE, i // STEP + 1)
    npass = 0
    for k in sorted(range(shown), key=lambda j: specs[j][3]):   # the passing trace is drawn last (on top)
        c, a, gam, ok = specs[k]
        age = i - k * STEP                      # frames since measured
        npass += 1 if ok else 0
        tr = 1.0 - a / (1.0 + ((xs - c) / gam) ** 2)
        py = np.clip((PLOT_T + (1.0 - (0.10 + 0.85 * tr)) * (PLOT_B - PLOT_T)).astype(int), PLOT_T, PLOT_B)
        if age < 2:
            col, th = np.array([1.0, 1.0, 1.0]), 1
        elif ok:
            pulse = 0.75 + 0.25 * np.cos(2 * np.pi * (age - 2) / 15.0)
            col, th = np.array([0.30, 1.0, 0.40]) * pulse, 3
        else:
            col, th = np.array([0.85, 0.18, 0.12]) * (0.85 if age < 12 else 0.45), 1
        for dy in range(-th, th + 1):
            rows = np.clip(py + dy, 0, H - 1)
            img[rows, np.arange(W)] = np.maximum(img[rows, np.arange(W)], col)
    # die strip: 10 rings on the die map
    for k in range(N_DIE):
        x = DIE_X0 + k * (DIE_S + DIE_GAP)
        age = i - k * STEP
        ok = specs[k][3]
        if age < 0:
            fill = (0.16, 0.18, 0.22)
        elif age < 2:
            fill = (0.95, 0.95, 0.95)
        elif ok:
            fill = (0.15, 0.85, 0.30)
        else:
            fill = (0.70, 0.12, 0.10)
        img[DIE_Y0:DIE_Y0 + DIE_S, x:x + DIE_S] = fill
        _ring(img, x + DIE_S // 2, DIE_Y0 + DIE_S // 2, 11, 3, (0.92, 0.92, 0.95) if age >= 0 else (0.35, 0.37, 0.42), 0.9)
        if ok and age >= 2:                    # ping: expanding ring around the good die, repeating every 15 frames
            ph = (age - 2) % 15
            r = 22 + 3 * ph
            _ring(img, x + DIE_S // 2, DIE_Y0 + DIE_S // 2, r, 3, (0.3, 1.0, 0.4), max(0.0, 1.0 - ph / 15.0))
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    T.draw_text(out, 10, 6, "%d/%d" % (npass, shown), 4, (120, 255, 150))
    return out


def monitor_sequence(outdir, name, nframes):
    os.makedirs(outdir, exist_ok=True)
    specs = _specs()
    for i in range(nframes):
        T.write_png(os.path.join(outdir, "%s_%04d.png" % (name, i + 1)), frame(i, specs))
    return specs
