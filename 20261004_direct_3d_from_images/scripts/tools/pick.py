"""Measuring CLI for modelers (all world coordinates in mm; pixel coords are 1/2-scale undistorted images).

  uv run python scripts/tools/pick.py views X Y Z [--n 10]          best views of a world point (in frame,
                                                                     facing, closest first) + its pixel there
  uv run python scripts/tools/pick.py crop VIEW U V [--half 200] [--step 20] [--up 2] [--out path.jpg]
                                                                     labeled-grid crop for reading pixel coords
  uv run python scripts/tools/pick.py tri VIEW U V [--depth 100,500]  pick in one view -> epipolar NCC match in
                                                                     neighbor views -> robust triangulation
  uv run python scripts/tools/pick.py tri2 VIEW1 U1 V1 VIEW2 U2 V2 [VIEW3 U3 V3 ...]
                                                                     manual correspondences -> triangulation
  uv run python scripts/tools/pick.py ray VIEW U V --z Z             intersect the pixel ray with plane z = Z (mm)
  uv run python scripts/tools/pick.py proj X Y Z VIEW [VIEW ...]    project a world point into views

Holdout views (data/split.json) are excluded and refused unless --allow-holdout (evaluation checks only).
Negative numbers: put "--" before the positional arguments, e.g. pick.py views -- -50 20 30.
Notes: tri works best on textured points away from silhouettes (backgrounds change between views); check
"n_inliers" >= 3 and "err_px" small (< 2). If it fails, use tri2 with picks you read from crops of 2-3 views.
Crops are written to outputs/picks/ unless --out is given; open them with the Read tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, OUTPUTS  # noqa: E402
from tools.geom import grid_crop, load_views, measure_point, neighbors, triangulate_robust  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--half", type=int, default=200)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--up", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth", default="100,500")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--allow-holdout", action="store_true", help="permit holdout views (evaluation only)")
    a = ap.parse_args()
    V = load_views(2)
    hold = set(json.loads((DATA / "split.json").read_text())["holdout"])
    if not a.allow_holdout:
        # holdout views are for evaluation only: never measure on them (audit 2026-10-05)
        for n in [x for x in a.args if x.startswith("IMG_")]:
            if n in hold:
                raise SystemExit(f"{n} is a holdout view; measuring on it leaks into evaluation "
                                 "(use a train view, or --allow-holdout for checks only)")
        V = {k: v for k, v in V.items() if k not in hold}
    if a.cmd == "views":
        X = np.array([float(x) for x in a.args[:3]]) / 1e3
        rows = []
        for v in V.values():
            uv, z = v.project(X)
            u, w = uv[0]
            if z[0] <= 0 or not (0 <= u < v.width and 0 <= w < v.height):
                continue
            rows.append((float(np.linalg.norm(v.center - X)) * 1e3, v.name, round(float(u)), round(float(w))))
        rows.sort()
        for d, n, u, w in rows[: a.n]:
            print(f"{n}  dist {d:.0f} mm  uv ({u}, {w})")
    elif a.cmd == "crop":
        v = V[a.args[0]]
        u, w = float(a.args[1]), float(a.args[2])
        out = Path(a.out) if a.out else OUTPUTS / "picks" / f"{v.name}_{int(u)}_{int(w)}.jpg"
        grid_crop(v, (u, w), a.half, out, step=a.step, upscale=a.up, marks=[(u, w)])
        print(out)
    elif a.cmd == "tri":
        v = V[a.args[0]]
        uv = np.array([float(a.args[1]), float(a.args[2])])
        d0, d1 = (float(x) / 1e3 for x in a.depth.split(","))
        r = measure_point(v, uv, neighbors(v, V, 25, 12), (d0, d1))
        if "X" in r:
            r["X_mm"] = (r.pop("X") * 1e3).round(2).tolist()
        print(json.dumps(r))
    elif a.cmd == "tri2":
        obs = [(V[a.args[i]], np.array([float(a.args[i + 1]), float(a.args[i + 2])])) for i in range(0, len(a.args), 3)]
        try:
            X, err, inl = triangulate_robust(obs, 4.0)
        except np.linalg.LinAlgError:
            print(json.dumps({"ok": False, "error": "no consistent triangulation (all picks disagree)"}))
            return
        print(json.dumps({"ok": bool(inl.sum() >= 2), "X_mm": (X * 1e3).round(2).tolist(),
                          "err_px": err.round(2).tolist(), "inliers": inl.tolist()}))
    elif a.cmd == "ray":
        v = V[a.args[0]]
        d = v.ray(np.array([float(a.args[1]), float(a.args[2])]))
        c = v.center
        s = (a.z / 1e3 - c[2]) / d[2]
        print(json.dumps({"X_mm": ((c + s * d) * 1e3).round(2).tolist()}))
    elif a.cmd == "proj":
        X = np.array([float(x) for x in a.args[:3]]) / 1e3
        for n in a.args[3:]:
            uv, z = V[n].project(X)
            print(f"{n} uv ({uv[0][0]:.1f}, {uv[0][1]:.1f}) depth {z[0] * 1e3:.0f} mm")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
