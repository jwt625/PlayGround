"""Constant offset (dx, dz) of a level-frame tube polyline (default rod_l) from a photo mask (braidmask.rod_mask) in
train views: grid search +-half mm on the straight run --y Y0,Y1; cost = braidmask.cost_map at the projected
centerline points (level frame -> [tilt] dzdy shear -> "tray" frame), summed over views. Refuses holdout views.
uv run python scripts/model/lower/fit_rod.py [--key rod_l] [--views ...] [--y 110,270] [--half 8]"""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.geom import load_views  # noqa: E402

from braidmask import cost_map, rod_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--key", default="rod_l")
ap.add_argument("--views", default="IMG_5725,IMG_5735,IMG_5736,IMG_5722,IMG_5749,IMG_5724,IMG_5721,IMG_5723,IMG_5752,IMG_5753,IMG_5758")
ap.add_argument("--y", default="110,270")
ap.add_argument("--half", type=float, default=8.0)
ap.add_argument("--res", type=float, default=0.5)
a = ap.parse_args()
hold = set(json.loads((ROOT / "data/split.json").read_text())["holdout"])
views = a.views.split(",")
if hold & set(views):
    sys.exit("refused: holdout views")
P = tomllib.loads((ROOT / "config/model/lower.toml").read_text())
R = np.array(json.loads((ROOT / "config/frames.json").read_text())["frames"]["tray"]["R"])
pts = np.array(P[a.key]["pts"], float)
D = []
for p0, p1 in zip(pts[:-1], pts[1:]):
    D += [p0 + (p1 - p0) * t for t in np.linspace(0, 1, max(2, int(np.linalg.norm(p1 - p0) / 2)), endpoint=False)]
D = np.array(D)
y0, y1 = map(float, a.y.split(","))
D = D[(D[:, 1] >= y0) & (D[:, 1] <= y1)]
t = P["tilt"]
g = np.arange(-a.half, a.half + 1e-6, a.res)
DX, DZ = [x.ravel() for x in np.meshgrid(g, g, indexing="ij")]
V = load_views(2)
cost = np.zeros(len(DX))
per = {}
for nm in views:
    _, m, _ = rod_mask(nm)
    cm = cost_map(m, cap=40.0)
    Q = D[None].repeat(len(DX), 0).copy()
    Q[..., 0] += DX[:, None]
    Q[..., 2] += DZ[:, None] + t["dzdy"] * (Q[..., 1] - t["y0"])
    W = Q.reshape(-1, 3) @ R.T
    uv, z = V[nm].project(W / 1e3)
    u, w = uv[:, 0], uv[:, 1]
    ok = (z > 0) & (u >= 0) & (w >= 0) & (u < cm.shape[1] - 1) & (w < cm.shape[0] - 1)
    c = np.full(len(u), 40.0, np.float32)
    c[ok] = cm[w[ok].astype(int), u[ok].astype(int)]
    c = c.reshape(len(DX), -1).mean(1)
    cost += c
    per[nm] = c
k0 = int(np.argmin(np.abs(DX) + np.abs(DZ)))
k = int(np.argmin(cost))
print(f"best dx {DX[k]:+.1f} dz {DZ[k]:+.1f}; cost per view {cost[k0] / len(views):.2f} -> {cost[k] / len(views):.2f}")
for nm, c in per.items():
    kk = int(np.argmin(c))
    print(f"  {nm}: own best dx {DX[kk]:+.1f} dz {DZ[kk]:+.1f}; cost {c[k0]:.1f} -> {c[k]:.1f}")
