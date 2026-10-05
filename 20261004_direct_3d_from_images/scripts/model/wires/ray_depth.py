"""Wire path from a 2D trace in one view + color masks in other views (wires agent helper; uv, not Blender).

  uv run python scripts/model/wires/ray_depth.py WIRE --views IMG_1587,IMG_1590,... [--zmax 90] [--smooth 0.5]
      [--write] [--debug out.jpg]

The trace (traces.json: view + 1/2-scale px list) fixes each point to a pixel ray; the height z along each ray is
chosen by dynamic programming: data cost = sum over side views of the clipped distance (1/4-scale px) from the
projected point to the wire's color mask (fit_wire.HSV); smoothness = smooth * |dz| per step (mm). Bundle wires
get their TOML start/end prepended/appended. Check the debug sheet: same-colored clutter can capture points.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_wire import TOML, catmull, check_views, load_views, mask_dt, wire_points, write_pts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wire")
    ap.add_argument("--views", default="")
    ap.add_argument("--zmax", type=float, default=90.0)
    ap.add_argument("--zmin", type=float, default=0.5)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--debug", default=None)
    ap.add_argument("--sparse", action="store_true", help="data cost from sparse points of the wire color (mm)")
    a = ap.parse_args()
    tr = json.loads((HERE / "traces.json").read_text())[a.wire]
    T = tomllib.loads(TOML.read_text())
    color, P0 = wire_points(T, a.wire)
    V2, V4 = load_views(2), load_views(4)
    check_views([tr["view"], *[v for v in a.views.split(",") if v]])
    va = V2[tr["view"]]
    uv = np.array(tr["uv"], float)
    zs = np.arange(a.zmin, a.zmax + 1e-6, 0.5)
    c = va.center
    rays = np.array([va.ray(p) for p in uv])
    # X[i, k] = point on ray i at height zs[k] (meters)
    tk = (zs[None, :] / 1e3 - c[2]) / rays[:, 2:3]
    X = c[None, None, :] + tk[..., None] * rays[:, None, :]
    cost = np.zeros(X.shape[:2])
    data = []
    full = np.ones((1071, 1428), np.uint8) * 255
    if a.sparse:
        import colorsys

        from scipy.spatial import cKDTree
        root = HERE.parents[2]
        pw = np.load(root / "data" / "points_world.npy")
        rgb = np.load(root / "data" / "points_rgb.npy").astype(float) / 255
        hsv = np.array([colorsys.rgb_to_hsv(*q) for q in rgb])
        sel = (hsv[:, 1] < 0.35) & (hsv[:, 2] > 0.55) if color == "white" else np.ones(len(pw), bool)
        dd, _ = cKDTree(pw[sel] * 1e3).query(X.reshape(-1, 3) * 1e3)
        cost += np.minimum(dd, 5.0).reshape(X.shape[:2]) * 3
    for n in (a.views.split(",") if a.views else []):
        v = V4[n]
        img = v.image()
        dt, m = mask_dt(img, color, full[: img.shape[0], : img.shape[1]], a.clip)
        q, dep = v.project(X.reshape(-1, 3))
        ok = (dep > 0) & (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < v.width - 1) & (q[:, 1] < v.height - 1)
        d = np.full(len(q), a.clip)
        qi = q[ok].astype(int)
        d[ok] = dt[qi[:, 1], qi[:, 0]]
        cost += d.reshape(X.shape[:2])
        data.append((v, m, img))
    # Viterbi over z with L1 smoothness
    n, K = cost.shape
    acc = cost[0].copy()
    back = np.zeros((n, K), int)
    dz = np.abs(zs[:, None] - zs[None, :]) * a.smooth
    for i in range(1, n):
        tot = acc[None, :] + dz  # [k_new, k_old]
        back[i] = np.argmin(tot, axis=1)
        acc = cost[i] + tot[np.arange(K), back[i]]
    ks = [int(np.argmin(acc))]
    for i in range(n - 1, 0, -1):
        ks.append(back[i][ks[-1]])
    ks = ks[::-1]
    P = np.array([X[i, k] * 1e3 for i, k in enumerate(ks)])
    per_pt = np.array([cost[i, k] / max(len(data), 1) for i, k in enumerate(ks)])
    if tr.get("prepend_mm"):
        P = np.vstack([tr["prepend_mm"], P])
    if tr.get("append_mm"):
        P = np.vstack([P, tr["append_mm"]])
    if a.wire.startswith("bundle_"):
        w = T["bundle"]["wires"][a.wire[len("bundle_"):]]
        P = np.vstack([w["start"], P, w["end"]])
    print(f"{a.wire}: mean data cost per view {per_pt.mean():.2f} px (clip {a.clip}); worst points",
          np.round(np.sort(per_pt)[-3:], 1).tolist())
    print("z:", np.round(P[:, 2], 1).tolist())
    print("pts =", np.round(P, 1).tolist())
    if a.debug:
        tiles = []
        for v, m, img in data:
            im = img.copy()
            im[m > 0] = (0.5 * im[m > 0] + [0, 127, 127]).astype(np.uint8)
            q, _ = v.project(catmull(P) / 1e3)
            cv2.polylines(im, [q.astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 2)
            cv2.putText(im, v.name, (10, 40), 0, 1.2, (255, 255, 255), 3)
            tiles.append(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(a.debug, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(a.debug)
    if a.write:
        write_pts(a.wire, P)


if __name__ == "__main__":
    main()
