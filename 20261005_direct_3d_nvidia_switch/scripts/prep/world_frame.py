"""Metric tray frame for the switch capture: COLMAP -> world similarity (s, R, t), written to config/world.yaml.

The modeling "world" frame is the tray's own frame (so tray parts are axis-aligned; gravity is a separate
transform in config/frames.json, used for the environment and exports):
  Z  normal of the tray's parallel layers (RANSAC on sparse points, refit by SVD), pointing out of the open top
  X  tray width: principal direction of the sparse points on the NVIDIA crossbar top (two views, averaged),
     sign so the S2 cameras (IMG_5741-5747) are at +X
  Y  Z x X: from the front panel toward the rear (front-view cameras are at -Y)
Scale: published body width 438 mm (SN6810-LD) over the measured distance between the outer side-wall faces
(screw holes triangulated on both walls; X of each face = mean over its holes).
Origin: X = 0 midway between the outer wall faces; Y = 0 at the front panel face (mode of Y over the sparse
points the front views see); Z = 0 at the crossbar top face (median Z of the crossbar points).
Inputs: config/picks.yaml. Triangulation runs in COLMAP units, so nothing depends on an earlier world.yaml.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_yaml  # noqa: E402
from tools.geom import load_points, load_views, measure_point, neighbors, triangulate_robust  # noqa: E402


def ransac_plane(Q: np.ndarray, thr: float, it: int = 3000, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(it):
        i = rng.choice(len(Q), 3, replace=False)
        n = np.cross(Q[i[1]] - Q[i[0]], Q[i[2]] - Q[i[0]])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((Q - Q[i[0]]) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    R = Q[best]
    c = R.mean(0)
    n = np.linalg.svd(R - c)[2][2]
    return n, float(best.sum())


def main() -> None:
    pk = load_yaml("picks.yaml")
    X, rgb, err, tracks = load_points(frame="colmap")
    good = err < 2.0
    V2 = load_views(2, frame="colmap")
    V4 = load_views(4, frame="colmap")
    hold = set(json.loads((DATA / "split.json").read_text())["holdout"])
    V2m = {k: v for k, v in V2.items() if k not in hold}  # measuring: train views only (audit DevLog-003)
    cams = np.array([v.center for v in V2.values()])

    # Z: dominant parallel layers
    hint = np.array(pk["tray_planes"]["colmap_normal_hint"], float)
    hint /= np.linalg.norm(hint)
    off = X @ hint
    lo, hi = pk["tray_planes"]["offset_band"]
    band = good & (-off > lo) & (-off < hi)
    n, n_inl = ransac_plane(X[band], thr=0.03)
    if n @ hint < 0:
        n = -n
    z = n if np.median((cams - X[band].mean(0)) @ n) > 0 else -n

    # X: crossbar direction
    dirs, zs_cb, cb_pts = [], [], []
    for name, poly in pk["crossbar_polygons_s4"].items():
        v = V4[name]
        seen = np.array([any(s == name for s, _ in tracks[i]) for i in range(len(X))])
        uv, _ = v.project(X)
        P = np.array(poly, np.float32)
        ins = np.array([cv2.pointPolygonTest(P, (float(a), float(b)), False) >= 0 for a, b in uv])
        Q = X[seen & ins & good]
        h = Q @ z
        Q = Q[np.abs(h - np.median(h)) < 15 / 1000 / 0.12]  # drop points off the crossbar top (about 15 mm)
        d = np.linalg.svd((Q - Q.mean(0)) - np.outer((Q - Q.mean(0)) @ z, z))[2][0]
        dirs.append(d if not dirs or d @ dirs[0] > 0 else -d)
        cb_pts.append(Q)
    xd = np.mean(dirs, 0)
    xd -= (xd @ z) * z
    xd /= np.linalg.norm(xd)
    s2 = np.array([V2[f"IMG_{k}"].center for k in range(5741, 5748)])
    if np.median((s2 - X[band].mean(0)) @ xd) < 0:
        xd = -xd
    y = np.cross(z, xd)
    front = np.array([V2[nm].center for nm in pk["front_panel_views"]])
    assert np.median((front - X[band].mean(0)) @ y) < 0, "front-view cameras should be at -Y"
    R = np.stack([xd, y, z])  # rows: world axes in COLMAP coordinates (rotation only)
    angle_between = float(np.degrees(np.arccos(abs(dirs[0] @ dirs[1]) / np.linalg.norm(dirs[0]) / np.linalg.norm(dirs[1]))))

    # Wall holes -> X of the two outer faces (COLMAP units along xd)
    med_cam = float(np.median(np.linalg.norm(cams - X[band].mean(0), axis=1)))
    wall = {}
    hole_log = []
    for side in ("minus_x", "plus_x"):
        xs = []
        for item in pk["wall_holes_s2"][side]:
            if "tri" in item:
                nm, u, w = item["tri"]
                r = measure_point(V2[nm], np.array([u, w], float), neighbors(V2[nm], V2m, 25, 12),
                                  (0.3 * med_cam, 2.5 * med_cam))
                if not r.get("ok"):
                    raise SystemExit(f"tri failed for {item}: {r}")
                P, e = r["X"], r["err_px"]
            else:
                a = item["tri2"]
                obs = [(V2[a[i]], np.array([a[i + 1], a[i + 2]], float)) for i in range(0, len(a), 3)]
                P, e, _ = triangulate_robust(obs, 4.0)
                e = e.round(2).tolist()
            xs.append(float(P @ xd))
            hole_log.append({"side": side, "item": item, "x_units": round(float(P @ xd), 5), "err_px": e})
        wall[side] = float(np.mean(xs))
    width_units = wall["plus_x"] - wall["minus_x"]
    mm_per_unit = pk["scale_reference"]["tray_body_width_mm"] / width_units
    s = mm_per_unit / 1000.0  # meters per COLMAP unit

    # Origin
    x0 = 0.5 * (wall["plus_x"] + wall["minus_x"])
    fv = set(pk["front_panel_views"])
    seen_f = np.array([bool({sv for sv, _ in tracks[i]} & fv) for i in range(len(X))]) & good
    Yf = X[seen_f] @ y
    hist, edges = np.histogram(Yf, bins=np.arange(Yf.min(), Yf.max(), 2.0 / mm_per_unit))
    k = int(np.argmax(hist))
    yc = 0.5 * (edges[k] + edges[k + 1])
    y0 = float(np.median(Yf[np.abs(Yf - yc) < 5.0 / mm_per_unit]))
    z0 = float(np.median(np.concatenate(cb_pts) @ z))
    o = np.array([x0, y0, z0])  # origin in rotated COLMAP coordinates
    t = -s * o  # X_world = s * R @ X_colmap + t

    out = {
        "generated_by": "scripts/prep/world_frame.py",
        "date": datetime.date.today().isoformat(),
        "frame": "tray (X width, Y front panel -> rear, Z out of the open top); meters",
        "scale_m_per_unit": float(s),
        "R": R.tolist(),
        "t": t.tolist(),
        "checks": {
            "tray_layer_plane_inliers": int(n_inl),
            "crossbar_dir_views_angle_deg": round(angle_between, 3),
            "wall_x_mm": {k2: round((v - x0) * mm_per_unit, 2) for k2, v in wall.items()},
            "holes": [{**h, "x_mm": round((h["x_units"] - x0) * mm_per_unit, 2)} for h in hole_log],
            "front_panel_points": int(seen_f.sum()),
            "scale_reference": "tray body width 438 mm (SN6810-LD; references/md/dimensions_research.md)",
        },
    }
    target = CONFIG / "world.yaml" if "--out" not in sys.argv else Path(sys.argv[sys.argv.index("--out") + 1])
    target.write_text("# Generated by scripts/prep/world_frame.py; do not edit by hand.\n"
                                      + yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out["checks"], sort_keys=False))
    print(f"mm per COLMAP unit {mm_per_unit:.3f}; width units {width_units:.4f}")


if __name__ == "__main__":
    main()
