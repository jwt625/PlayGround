"""Bake the case's photo-texture panels (scripts/model/case/case_panels.py) from train views only.

Step 1 (ID renders of all train views with the current model, panels' objects present):
  scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- \
      --out outputs/runs/case_idtrain --build scripts/model/build_all.py --views train --passes id
Step 2:
  uv run python scripts/model/case/bake_case.py --idrun outputs/runs/case_idtrain [--ppm 4] [--panels a,b]
Per panel: rank train views by facing (cos of panel normal to the camera direction) over distance, keep views where
the panel center is in frame, then scripts/tools/bake_id_owned.py (ID-owned pixels only, per-texel median of the
first n_best visible views). Holdout views are never used (data/split.json). Writes assets/textures/case_<panel>.png
and outputs/textures/case_<panel>_views.jpg; specs go to outputs/scratch/case/bake_specs/.
"""

from __future__ import annotations

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
import case_panels  # noqa: E402
from tools.bake_id_owned import owned_mask  # noqa: E402
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, sample_view, surface_grid  # noqa: E402


def bake(spec, V, intr, img_dir, pct):
    """bake_id_owned.py logic (ID-owned texels, first n_best visible views) with a per-texel percentile instead of
    the median: glossy black walls carry view-dependent mat/lamp reflections, a low percentile keeps the dark
    base look (Phase 3, case_034)."""
    X, _ = surface_grid(spec["surface"], float(spec["px_per_mm"]))
    run = ROOT / spec["occlusion_run"]
    id_map = json.loads((run / "id_map.json").read_text())
    cols = [np.array(id_map[n][::-1]) for n in spec["occlusion_object"] if n in id_map]
    texs, vis = [], []
    for nm in spec["views"]:
        v = V[nm]
        t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
        idm = cv2.imread(str(run / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        texs.append(t)
        vis.append(inb & owned_mask(v, X, idm, cols, int(spec["erode_px"])))
    A = np.stack(texs).astype(np.float32)
    M = np.stack(vis)
    M &= np.cumsum(M, axis=0) <= int(spec["n_best"])
    A[~M] = np.nan
    with np.errstate(all="ignore"):
        lum = np.nanmean(A, axis=3)
        q = np.nanpercentile(lum, pct, axis=0, method="nearest")
        pick = np.nanargmin(np.where(np.isnan(lum), np.inf, np.abs(lum - q[None])), axis=0)
    B = np.take_along_axis(A, pick[None, ..., None].repeat(3, -1), axis=0)[0]
    hole = np.isnan(B).any(2) | ~M.any(0)
    B[hole] = np.nanmedian(B[~hole], axis=0) if (~hole).any() else 0
    B = np.clip(B, 0, 255).astype(np.uint8)
    out = ROOT / "assets" / "textures" / f"{spec['name']}.png"
    cv2.imwrite(str(out), B)
    cov = np.clip(M.sum(0) * 40, 0, 255).astype(np.uint8)
    cv2.imwrite(str(ROOT / "outputs" / "textures" / f"{spec['name']}_views.jpg"),
                np.hstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)]))
    return {"texture": str(out.relative_to(ROOT)), "size_px": list(B.shape[:2]), "hole_frac": round(float(hole.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrun", required=True)
    ap.add_argument("--ppm", type=float, default=4.0)
    ap.add_argument("--panels", default="")
    ap.add_argument("--n-views", type=int, default=16)
    ap.add_argument("--pct", type=float, default=50.0, help="per-texel luminance percentile over the n_best views")
    ap.add_argument("--n-best", type=int, default=5)  # 5 vs 3: +0.05 dB (case_030 vs case_029)
    a = ap.parse_args()
    P = tomllib.loads((ROOT / "config" / "model" / "case.toml").read_text())
    split = json.loads((ROOT / "data" / "split.json").read_text())
    train = set(split["train"]) - set(split["holdout"])
    idrun = ROOT / a.idrun
    have = {p.stem for p in (idrun / "id").glob("*.png")}
    V = load_views(4)
    intr, img_dir = full_res_cams()
    sdir = ROOT / "outputs" / "scratch" / "case" / "bake_specs"
    sdir.mkdir(parents=True, exist_ok=True)
    want = set(a.panels.split(",")) if a.panels else None
    for pn in case_panels.panels(P):
        if want and pn["name"] not in want:
            continue
        C = np.array([case_panels.to_world(q, P["frame"]) for q in pn["c"]])
        yaw = np.radians(P["frame"]["yaw_deg"])
        n = np.array(pn["n"], float)
        n /= np.linalg.norm(n)
        n = np.array([np.cos(yaw) * n[0] - np.sin(yaw) * n[1], np.sin(yaw) * n[0] + np.cos(yaw) * n[1], n[2]])
        ctr = C.mean(0) / 1e3
        rows = []
        for nm in sorted(train & have):
            v = V[nm]
            d = v.center - ctr
            dist = np.linalg.norm(d)
            cos = float(n @ d / dist)
            uv, z = v.project(C / 1e3)
            if cos < 0.25 or (z <= 0).any():
                continue
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < v.width) & (uv[:, 1] >= 0) & (uv[:, 1] < v.height)).mean()
            if inside < 0.5:
                continue
            rows.append((cos * inside / (dist * 1e3) * 100, nm))
        rows.sort(reverse=True)
        views = [nm for _, nm in rows[: a.n_views]]
        if not views:
            print(pn["name"], "no views")
            continue
        spec = {"name": f"case_{pn['name']}", "surface": {"type": "quad", "corners_mm": C.round(3).tolist()},
                "px_per_mm": a.ppm, "views": views, "occlusion_run": str(idrun.relative_to(ROOT)),
                "occlusion_object": [pn["obj"]], "n_best": a.n_best, "erode_px": 1, "fill": "median"}
        sp = sdir / f"{pn['name']}.json"
        sp.write_text(json.dumps(spec, indent=1))
        r = bake(spec, V, intr, img_dir, a.pct)
        print(pn["name"], json.dumps(r), "views", views[:4])


if __name__ == "__main__":
    main()
