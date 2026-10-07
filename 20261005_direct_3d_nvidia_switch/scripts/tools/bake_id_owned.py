"""Bake a planar texture (e.g. a PCB top crowded with parts and wires) from ID-owned photo pixels only.

A texel is taken from a view only if, in that view's ID render of the current model, the target object owns the
pixel (so parts, wires and other occluders that are modeled never paint onto the surface). Per texel, the median
of its first n_best visible views in rank order is used; texels never seen fall back to a flat fill color (the
median of the seen texels) instead of inpainting, which smears over large occluded areas.

Step 1, ID renders of the candidate views with the current model (all groups built):
  scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- \
      --out outputs/runs/<id_run> --build scripts/model/build_all.py --views IMG_1560,IMG_1561,... --passes id
Step 0 (optional), rank TRAIN views (data/split.json; holdout views are never used) for a spec whose "views" is
"auto": prints a comma list of the best n_rank views (patch center in frame, facing, by cos x px per mm):
  uv run python scripts/tools/bake_id_owned.py spec.json --rank [--n-rank 12]
Step 2:
  uv run python scripts/tools/bake_id_owned.py spec.json
spec.json:
  {"name": "pcb_board_top",                         # output assets/textures/<name>.png
   "surface": {"type": "quad", "corners_mm": [TL, TR, BR, BL]},    # world mm, image orientation
   "px_per_mm": 8,
   "views": [...] | "auto",                          # ranked best first; must all exist in <id_run>/id/
                                                     #   ("auto" = --rank result, train views only)
   "occlusion_run": "outputs/runs/<id_run>",
   "occlusion_object": "pcb.board",                  # object(s) whose surface is baked: a name or a list (e.g.
                                                     #   the box and the textured quad lying on it)
   "n_best": 3,                                      # per texel: median of the first n_best visible views
   "erode_px": 2,                                    # shrink each view's owned mask (1/4-scale px) against bleed
   "fill": [b, g, r] | "median"}                     # never-seen texels (default "median" of seen texels)
Writes assets/textures/<name>.png and outputs/textures/<name>_views.jpg (bake | per-texel view count x 40).
Pick views with texture_from_photos.py "views": "auto" output (candidates_ranked) or near-nadir views.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS, ROOT  # noqa: E402
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, sample_view, surface_grid  # noqa: E402


def owned_mask(v, X, idm, cols, erode_px):
    """(h, w) bool: texel projects to a pixel owned by one of the object colors cols (BGR) in the ID render idm."""
    own = np.zeros(idm.shape[:2], bool)
    for col in cols:
        own |= np.abs(idm[..., :3].astype(int) - col).sum(2) <= 6
    own_img = (own & (idm[..., 3] > 0)).astype(np.uint8)
    if erode_px > 0:
        own_img = cv2.erode(own_img, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    uv, _ = v.project(X.reshape(-1, 3))
    uu = np.clip(np.round(uv[:, 0]).astype(int), 0, idm.shape[1] - 1)
    vv = np.clip(np.round(uv[:, 1]).astype(int), 0, idm.shape[0] - 1)
    return own_img[vv, uu].reshape(X.shape[:2]) > 0


def rank_train_views(X, N, V, intr, n):
    """Train views (split.json) that see the patch center, facing it, ranked by cos(angle) x px per mm."""
    train = json.loads((ROOT / "data" / "split.json").read_text())["train"]
    ctr = X[X.shape[0] // 2, X.shape[1] // 2]
    nc = N[N.shape[0] // 2, N.shape[1] // 2]
    out = []
    for nm in train:
        v = V[nm]
        d = v.center - ctr
        dist = float(np.linalg.norm(d))
        cosang = float(nc @ d / dist)
        if cosang <= 0.2:
            continue
        uv, z = v.project(ctr[None])
        if z[0] <= 0 or not (2 <= uv[0, 0] < v.width - 2 and 2 <= uv[0, 1] < v.height - 2):
            continue
        out.append((cosang * intr[v.camera_id][0] / dist, nm))
    return [nm for _, nm in sorted(out, reverse=True)[:n]]


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    X, N = surface_grid(spec["surface"], float(spec.get("px_per_mm", 8)))
    V = load_views(4)
    intr, img_dir = full_res_cams()
    n_rank = int(sys.argv[sys.argv.index("--n-rank") + 1]) if "--n-rank" in sys.argv else 12
    if "--rank" in sys.argv:
        print(",".join(rank_train_views(X, N, V, intr, n_rank)))
        return
    if spec["views"] == "auto":
        spec["views"] = rank_train_views(X, N, V, intr, n_rank)
    holdout = set(json.loads((ROOT / "data" / "split.json").read_text())["holdout"])
    if holdout & set(spec["views"]):
        raise SystemExit(f"holdout views in spec: {sorted(holdout & set(spec['views']))}")
    run = ROOT / spec["occlusion_run"]
    id_map = json.loads((run / "id_map.json").read_text())
    names = spec["occlusion_object"] if isinstance(spec["occlusion_object"], list) else [spec["occlusion_object"]]
    cols = [np.array(id_map[n][::-1]) for n in names if n in id_map]
    if not cols:
        raise SystemExit(f"none of {names} in {run / 'id_map.json'}")
    texs, vis = [], []
    for nm in spec["views"]:
        v = V[nm]
        t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
        idm = cv2.imread(str(run / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        texs.append(t)
        vis.append(inb & owned_mask(v, X, idm, cols, int(spec.get("erode_px", 2))))
    A = np.stack(texs).astype(np.float32)
    M = np.stack(vis)
    M &= np.cumsum(M, axis=0) <= int(spec.get("n_best", 3))
    A[~M] = np.nan
    with np.errstate(all="ignore"):
        B = np.nanmedian(A, axis=0)
    hole = np.isnan(B).any(2)
    if hole.all():
        raise SystemExit("no texel visible: check occlusion_object names and the ID run views")
    fill = spec.get("fill", "median")
    B[hole] = np.nanmedian(B[~hole], axis=0) if fill == "median" else np.array(fill, np.float32)
    B = np.clip(B, 0, 255).astype(np.uint8)
    out = ROOT / "assets" / "textures" / f"{spec['name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), B)
    cov = np.clip(M.sum(0) * 40, 0, 255).astype(np.uint8)
    (OUTPUTS / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS / "textures" / f"{spec['name']}_views.jpg"),
                np.hstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)]))
    print(json.dumps({"texture": str(out.relative_to(ROOT)), "size_px": list(B.shape[:2]),
                      "hole_frac": round(float(hole.mean()), 3), "n_views": len(texs)}))


if __name__ == "__main__":
    main()
