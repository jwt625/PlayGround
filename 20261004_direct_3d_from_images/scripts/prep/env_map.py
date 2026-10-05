"""Room environment map from the photos' background pixels (for reflections in glossy/clear surfaces).

Every pixel whose viewing ray leaves the mat (mask value 128) shows the room. Treating the room as infinitely far
(directions taken from the object center), the pixels of all views are binned by direction into an equirectangular
map (median per bin, linear RGB). Unseen bins (mostly near the zenith and below the table) are inpainted. The map
is scaled so the cosine-weighted irradiance on an upward-facing surface equals 1, which keeps the render calibration
(photo-textured diffuse surfaces render at their photo values) while reflections show the real room.

Usage: uv run python scripts/prep/env_map.py [--w 512]
Output: data/env/room.hdr (float, linear), data/env/room_preview.jpg, data/env/room_meta.json
Limits: LDR photos clip the ceiling lights; the infinite-distance assumption mislocates near objects (the ring lamp
over the mat), so reflections of close objects are approximate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402
from tools.geom import load_views  # noqa: E402


def srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--step", type=int, default=3, help="pixel subsampling at 1/4 scale")
    a = ap.parse_args()
    W, H = a.w, a.w // 2
    V = load_views(4)
    bins = [[] for _ in range(W * H)]
    acc_idx, acc_col = [], []
    for n, v in V.items():
        pm = cv2.imread(str(DATA / "masks_s4" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        img = v.image()
        ys, xs = np.nonzero(pm[:: a.step, :: a.step] == 128)
        ys, xs = ys * a.step, xs * a.step
        if len(xs) == 0:
            continue
        d = (v.R.T @ (np.linalg.inv(v.K) @ np.stack([xs, ys, np.ones_like(xs)]).astype(float))).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arcsin(np.clip(d[:, 2], -1, 1))
        u = ((az + np.pi) / (2 * np.pi) * W).astype(int) % W
        w = np.clip(((np.pi / 2 - el) / np.pi * H).astype(int), 0, H - 1)
        acc_idx.append(w * W + u)
        acc_col.append(img[ys, xs].astype(np.float32))
    idx = np.concatenate(acc_idx)
    col = np.concatenate(acc_col)
    order = np.argsort(idx, kind="stable")
    idx, col = idx[order], col[order]
    uniq, start, cnt = np.unique(idx, return_index=True, return_counts=True)
    env = np.zeros((H * W, 3), np.float32)
    seen = np.zeros(H * W, bool)
    for k, s0, c in zip(uniq, start, cnt):
        if c >= 3:
            env[k] = np.median(col[s0:s0 + c], axis=0)
            seen[k] = True
    env = env.reshape(H, W, 3)
    seen = seen.reshape(H, W)
    filled = cv2.inpaint(np.clip(env, 0, 255).astype(np.uint8), (~seen).astype(np.uint8), 9, cv2.INPAINT_TELEA)
    lin = srgb_to_lin(filled.astype(np.float32))
    # cosine-weighted irradiance on an upward plane: sum over upper hemisphere L cos(theta) dOmega
    el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    dOm = (np.cos(el) * (np.pi / H) * (2 * np.pi / W))[:, None]
    up = (el > 0)[:, None]
    lum = lin.mean(2)
    E = float((lum * np.sin(el)[:, None] * dOm * up).sum())
    scale = np.pi / E  # uniform radiance 1 gives E = pi
    out = DATA / "env"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "room.hdr"), (lin * scale).astype(np.float32))
    prev = np.hstack([filled, cv2.cvtColor((seen * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(out / "room_preview.jpg"), prev)
    meta = {"width": W, "height": H, "seen_fraction": float(seen.mean()),
            "seen_upper_fraction": float(seen[: H // 2].mean()), "scale": scale, "n_samples": int(len(idx))}
    (out / "room_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
