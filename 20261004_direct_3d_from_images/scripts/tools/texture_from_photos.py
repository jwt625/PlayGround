"""Extract a rectified texture (labels, printed art, PCB tops) from the full-resolution original photos.

Usage:
  uv run python scripts/tools/texture_from_photos.py spec.json
spec.json:
  {"name": "crt_yoke_label",                       # output assets/textures/<name>.png
   "surface": {"type": "quad", "corners_mm": [TL, TR, BR, BL]}          # world mm, image orientation
          or {"type": "cylinder", "base_mm": [x,y,z], "axis": [ax,ay,az], "radius_mm": r,
              "ref_dir": [rx,ry,rz],            # direction of angle 0 (perpendicular to axis)
              "angle_deg": [a0, a1], "length_mm": [s0, s1]}  # texture u = angle, v = length (top = s1)
   "px_per_mm": 12,                               # output resolution
   "views": "auto" | ["IMG_1560", ...],       # auto = all train views (holdout views are never used)
   "n_best": 3,                                   # auto: blend the best n views (per-texel median)
   "occlusion_run": "outputs/runs/<run>",         # optional: id renders of the model; a texel counts as
   "occlusion_object": "crt.label_yoke",          #   visible in a view only if that object owns the pixel
   "partial": false,                              # optional: accept views that see only part of the patch
   "exclude_object_mask": false,                  # optional (background surfaces like the mat): drop texels
                                                  #   where the photo mask says object or unknown
   "per_texel": false                             # optional (large surfaces): every texel takes the median of
  }                                               #   its own best n_best views by local px/mm x cos(angle);
                                                  #   never-seen texels are inpainted
Writes assets/textures/<name>.png and outputs/textures/<name>_views.jpg (per-view samples + chosen blend) so
the result can be checked visually. View score = cos(view angle to the surface normal) x px per mm, among
views that see the whole patch in frame. Texels are sampled from raw originals with the SIMPLE_RADIAL model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, OUTPUTS, ROOT, scene_cfg  # noqa: E402
from prep.colmap_io import read_cameras  # noqa: E402
from tools.geom import load_views  # noqa: E402


def surface_grid(s: dict, ppm: float):
    """Returns world points (h, w, 3) in meters and normals (h, w, 3)."""
    if s["type"] == "quad":
        C = np.array(s["corners_mm"], float) / 1e3
        w = max(2, int(round(np.linalg.norm(C[1] - C[0]) * 1e3 * ppm)))
        h = max(2, int(round(np.linalg.norm(C[3] - C[0]) * 1e3 * ppm)))
        u = (np.arange(w) + 0.5) / w
        v = (np.arange(h) + 0.5) / h
        U, V = np.meshgrid(u, v)
        top = C[0][None, None] * (1 - U[..., None]) + C[1][None, None] * U[..., None]
        bot = C[3][None, None] * (1 - U[..., None]) + C[2][None, None] * U[..., None]
        X = top * (1 - V[..., None]) + bot * V[..., None]
        n = np.cross(C[3] - C[0], C[1] - C[0])  # (down) x (right) points toward the viewer of the print
        n /= np.linalg.norm(n)
        return X, np.broadcast_to(n, X.shape).copy()
    if s["type"] == "cylinder":
        b = np.array(s["base_mm"], float) / 1e3
        ax = np.array(s["axis"], float)
        ax /= np.linalg.norm(ax)
        r0 = np.array(s["ref_dir"], float)
        r0 -= (r0 @ ax) * ax
        r0 /= np.linalg.norm(r0)
        r1 = np.cross(ax, r0)
        R = s["radius_mm"] / 1e3
        a0, a1 = np.radians(s["angle_deg"])
        s0, s1 = np.array(s["length_mm"], float) / 1e3
        w = max(2, int(round(abs(a1 - a0) * R * 1e3 * ppm)))
        h = max(2, int(round(abs(s1 - s0) * 1e3 * ppm)))
        A, S = np.meshgrid(a0 + (a1 - a0) * (np.arange(w) + 0.5) / w, s1 - (s1 - s0) * (np.arange(h) + 0.5) / h)
        n = np.cos(A)[..., None] * r0 + np.sin(A)[..., None] * r1
        X = b + S[..., None] * ax + R * n
        return X, n
    raise ValueError(s["type"])


def full_res_cams():
    cfg = scene_cfg()
    cams = read_cameras(cfg["source_dir"] / cfg["sparse_model"] / "cameras.bin")
    return {cid: c.params for cid, c in cams.items()}, cfg["source_dir"] / "images"


def sample_view(v, params, img_dir, X):
    f, cx, cy, k1 = params
    Xc = X.reshape(-1, 3) @ v.R.T + v.t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    u = (f * x * d + cx - 0.5).astype(np.float32).reshape(X.shape[:2])
    w = (f * y * d + cy - 0.5).astype(np.float32).reshape(X.shape[:2])
    src = cv2.imread(str(img_dir / f"{v.name}.jpeg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tex = cv2.remap(src, u, w, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    inb = (u >= 0) & (w >= 0) & (u < src.shape[1] - 1) & (w < src.shape[0] - 1) & (z.reshape(u.shape) > 0)
    return tex, inb


def per_texel(spec, X, N, V, intr, img_dir, names):
    k = int(spec.get("n_best", 5))
    h, w = X.shape[:2]
    cols, scores = [], []
    for nm in names:
        v = V[nm]
        t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
        d = v.center[None, None] - X
        dist = np.linalg.norm(d, axis=2)
        cosang = (d * N).sum(2) / dist
        sc = (cosang * intr[v.camera_id][0] / (dist * 1e3)).astype(np.float32)  # px per mm x cos
        if spec.get("exclude_object_mask"):
            pm = cv2.imread(str(DATA / "masks_s4" / f"{nm}.png"), cv2.IMREAD_GRAYSCALE)
            bad = cv2.dilate((pm != 0).astype(np.uint8), np.ones((7, 7), np.uint8))
            uv, _ = v.project(X.reshape(-1, 3))
            uu = np.clip(np.round(uv[:, 0]).astype(int), 0, pm.shape[1] - 1)
            vv = np.clip(np.round(uv[:, 1]).astype(int), 0, pm.shape[0] - 1)
            inb &= (bad[vv, uu] == 0).reshape(inb.shape)
        sc[~inb | (cosang < 0.2)] = -1
        cols.append(t)
        scores.append(sc)
    C = np.stack(cols)  # V,h,w,3 uint8
    S = np.stack(scores)  # V,h,w
    order = np.argsort(-S, axis=0)[:k]  # k,h,w
    Ck = np.take_along_axis(C, order[..., None], 0).astype(np.float32)
    Sk = np.take_along_axis(S, order, 0)
    Ck[Sk <= 0] = np.nan
    with np.errstate(all="ignore"):
        blend = np.nanmedian(Ck, axis=0)
    hole = np.isnan(blend).any(2)
    blend = np.nan_to_num(blend, nan=0).astype(np.uint8)
    if hole.any():
        blend = cv2.inpaint(blend, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    out = ROOT / "assets" / "textures"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{spec['name']}.png"), blend)
    cov = np.clip((Sk > 0).sum(0) * 50, 0, 255).astype(np.uint8)
    (OUTPUTS / "textures").mkdir(parents=True, exist_ok=True)
    sheet = np.hstack([blend, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(OUTPUTS / "textures" / f"{spec['name']}_views.jpg"), sheet)
    print(json.dumps({"texture": str((out / f"{spec['name']}.png").relative_to(ROOT)), "size_px": [h, w],
                      "n_views": len(names), "hole_frac": round(float(hole.mean()), 3)}))


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    ppm = float(spec.get("px_per_mm", 12))
    X, N = surface_grid(spec["surface"], ppm)
    V = load_views(4)
    intr, img_dir = full_res_cams()
    occ = None
    if spec.get("occlusion_run"):
        run = ROOT / spec["occlusion_run"]
        id_map = json.loads((run / "id_map.json").read_text())
        col = id_map[spec["occlusion_object"]]
        occ = (run, np.array(col[::-1]))  # BGR
    cands = []
    hold = set(json.loads((DATA / "split.json").read_text())["holdout"])
    names = spec["views"] if spec.get("views", "auto") != "auto" else [n for n in V if n not in hold]
    if not spec.get("allow_holdout") and any(n in hold for n in names):
        raise SystemExit("spec lists holdout views; textures must come from train views only "
                         "(set allow_holdout only for diagnostics)")
    ctr = X[X.shape[0] // 2, X.shape[1] // 2]
    nc = N[N.shape[0] // 2, N.shape[1] // 2]
    for nm in names:
        v = V[nm]
        f = intr[v.camera_id][0]
        dvec = v.center - ctr
        dist = np.linalg.norm(dvec)
        cosang = float(nc @ dvec / dist)
        if cosang <= 0.2:
            continue
        probe = X[[X.shape[0] // 2], [X.shape[1] // 2]] if spec.get("partial") else X[[0, 0, -1, -1], [0, -1, 0, -1]]
        uv, z = v.project(probe)
        if (z <= 0).any() or (uv < 2).any() or (uv[:, 0] > v.width - 3).any() or (uv[:, 1] > v.height - 3).any():
            continue
        cands.append((cosang * f / dist, nm, cosang))
    cands.sort(reverse=True)
    if not cands:
        raise SystemExit("no view sees the whole patch")
    if spec.get("per_texel"):
        return per_texel(spec, X, N, V, intr, img_dir, [nm for _, nm, _ in cands])
    n_best = int(spec.get("n_best", 3))
    use = cands[: max(n_best, 1)] if spec.get("views", "auto") == "auto" else cands
    texs, vis = [], []
    for score, nm, cosang in use:
        v = V[nm]
        t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
        if occ is not None:
            idm = cv2.imread(str(occ[0] / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
            uv, _ = v.project(X.reshape(-1, 3))
            uu = np.clip(np.round(uv[:, 0]).astype(int), 0, idm.shape[1] - 1)
            vv = np.clip(np.round(uv[:, 1]).astype(int), 0, idm.shape[0] - 1)
            own = (np.abs(idm[vv, uu, :3].astype(int) - occ[1]).sum(1) <= 3) & (idm[vv, uu, 3] > 0)
            inb &= own.reshape(inb.shape)
        if spec.get("exclude_object_mask"):
            pm = cv2.imread(str(DATA / "masks_s4" / f"{nm}.png"), cv2.IMREAD_GRAYSCALE)
            pm = cv2.dilate((pm == 255).astype(np.uint8), np.ones((5, 5), np.uint8))
            uv, _ = v.project(X.reshape(-1, 3))
            uu = np.clip(np.round(uv[:, 0]).astype(int), 0, pm.shape[1] - 1)
            vv = np.clip(np.round(uv[:, 1]).astype(int), 0, pm.shape[0] - 1)
            inb &= (pm[vv, uu] == 0).reshape(inb.shape)
        texs.append(t)
        vis.append(inb)
    T = np.stack(texs).astype(np.float32)
    M = np.stack(vis)
    T[~M] = np.nan
    blend = np.nanmedian(T, axis=0) if len(texs) > 1 else T[0]
    hole = np.isnan(blend).any(2)
    blend = np.nan_to_num(blend, nan=0).astype(np.uint8)
    if hole.any():
        blend = cv2.inpaint(blend, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    out = ROOT / "assets" / "textures"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{spec['name']}.png"), blend)
    sheet_w = 360
    tiles = []
    for (score, nm, cosang), t in zip(use, texs):
        tt = cv2.resize(t, (sheet_w, max(1, sheet_w * t.shape[0] // t.shape[1])))
        cv2.putText(tt, f"{nm} cos {cosang:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(tt)
    bl = cv2.resize(blend, (sheet_w, max(1, sheet_w * blend.shape[0] // blend.shape[1])))
    cv2.putText(bl, "BLEND (median)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    tiles.append(bl)
    (OUTPUTS / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS / "textures" / f"{spec['name']}_views.jpg"), np.vstack(tiles))
    print(json.dumps({"texture": str((out / f"{spec['name']}.png").relative_to(ROOT)), "size_px": blend.shape[:2],
                      "views": [(nm, round(c, 2)) for _, nm, c in use],
                      "candidates_ranked": [nm for _, nm, _ in cands[:10]]}))


if __name__ == "__main__":
    main()
