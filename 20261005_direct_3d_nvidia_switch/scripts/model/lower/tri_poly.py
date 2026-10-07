"""Triangulate a curve from 2D polylines traced in several views (s2 pixel coords).
Input JSON: {"ref": "IMG_A", "polys": {"IMG_A": [[u,v],...], "IMG_B": [[u,v],...], ...}, "depth": [200, 900],
             "n": 40}
For each of n points resampled along the ref polyline, pick the depth along its ray that minimizes the sum over
the other views of the distance from the projection to that view's polyline (robust: capped at 30 px).
Prints world mm points with per-view residuals and the depth curvature (conditioning)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from tools.geom import load_views  # noqa: E402

HOLD = {"IMG_5715", "IMG_5728", "IMG_5737", "IMG_5747", "IMG_5750"}


def resample(P, n):
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, i]) for i in range(2)], 1)


def seg_dist(q, P):
    """distance from points q (m,2) to polyline P (k,2)"""
    a, b = P[:-1], P[1:]
    ab = b - a
    L = (ab ** 2).sum(1)
    t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / np.maximum(L, 1e-9)[None], 0, 1)
    c = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(q[:, None, :] - c, axis=2).min(1)


cfg = json.loads(Path(sys.argv[1]).read_text())
V = load_views(2)
assert not (set(cfg["polys"]) & HOLD)
ref = cfg["ref"]
R = resample(cfg["polys"][ref], cfg.get("n", 40))
others = [k for k in cfg["polys"] if k != ref]
dmin, dmax = cfg.get("depth", [150, 1000])
ds = np.arange(dmin, dmax, 0.25) / 1e3
vr = V[ref]
out = []
for uv in R:
    r = vr.ray(uv)
    X = vr.center[None] + ds[:, None] * r[None]
    cost = np.zeros(len(ds))
    res = {}
    for o in others:
        q, dep = V[o].project(X)
        dd = np.minimum(seg_dist(q, np.asarray(cfg["polys"][o], float)), 30.0)
        dd[dep <= 0] = 30.0
        cost += dd
        res[o] = dd
    i = int(np.argmin(cost))
    # width of the basin where cost < min + 3 px  (depth uncertainty, mm)
    ok = np.where(cost < cost[i] + 3.0)[0]
    out.append({"xyz": (X[i] * 1e3).round(1).tolist(), "res": {o: round(float(res[o][i]), 1) for o in others},
                "unc_mm": round(float((ds[ok[-1]] - ds[ok[0]]) * 1e3), 1)})
for o in out:
    print(o["xyz"], o["res"], "unc", o["unc_mm"])
Path(sys.argv[2] if len(sys.argv) > 2 else "/dev/null").write_text(json.dumps(out))
