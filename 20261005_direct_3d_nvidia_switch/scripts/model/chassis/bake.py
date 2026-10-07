"""Bake chassis photo textures (ID-owned, facing-checked, per-texel statistic over the best K train views).

  uv run python scripts/model/chassis/bake.py <id_run> [name ...] [--stat median|pNN] [--k K] [--sessions S1,S3] [--suffix S]
      [--dry-run]

<id_run>: outputs/runs/<run> with an ID pass of TRAIN views of the current model (render_views.py --passes id).
Surfaces: chassis_surfaces.surfaces(P) (local frame, rotated to world with body.roll_deg). Per-surface options
in config/model/chassis.toml [tex.<name>]: sessions (list of S1/S2/S3, default all), stat ("median" or "pNN",
e.g. "p25"), k (views per texel, default 5), px_per_mm (default 4), min_cos (default 0.25), sign_mask_views_only
(only views with a sign SAM mask; CLI --sign-views), views (explicit list).
Per texel: views ranked by cos(angle) x px/mm at the texel; a view counts where the texel faces it, lies in
frame and the ID pass says one of the owners has the pixel (eroded 2 px). Never-seen texels: median fill.
Writes assets/textures/chassis_<name><suffix>.png and outputs/textures/chassis_<name><suffix>_views.jpg
(bake | view count x 40). Holdout views are never used.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
from chassis_surfaces import roll_to_world, surfaces  # noqa: E402
from tools.bake_id_owned import owned_mask  # noqa: E402
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, surface_grid  # noqa: E402

_SRC: dict = {}


def sample_view(v, params, img_dir, X):
    """As tools.texture_from_photos.sample_view, but finds the original by stem (this project: .JPG; the shared
    tool hard-codes .jpeg) and caches the decoded image."""
    f, cx, cy, k1 = params
    Xc = X.reshape(-1, 3) @ v.R.T + v.t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    u = (f * x * d + cx - 0.5).astype(np.float32).reshape(X.shape[:2])
    w = (f * y * d + cy - 0.5).astype(np.float32).reshape(X.shape[:2])
    if v.name not in _SRC:
        path = next(p for p in Path(img_dir).iterdir() if p.stem == v.name)
        if len(_SRC) > 12:
            _SRC.pop(next(iter(_SRC)))
        _SRC[v.name] = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    src = _SRC[v.name]
    tex = cv2.remap(src, u, w, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    inb = (u >= 0) & (w >= 0) & (u < src.shape[1] - 1) & (w < src.shape[0] - 1) & (z.reshape(u.shape) > 0)
    return tex, inb


def sign_mask(v, X, shape, dil_px: int = 3):
    """(h, w) bool: texel projects into the acrylic sign's SAM mask (data/sam_s2/sign/<view>.png; views without a
    mask: none). The sign is not modeled, so the ID pass cannot keep its pixels out of chassis textures."""
    f = ROOT / "data" / "sam_s2" / "sign" / f"{v.name}.png"
    if not f.exists():
        return np.zeros(X.shape[:2], bool)
    m = cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dil_px + 1, 2 * dil_px + 1), np.uint8)) > 0
    uv, _ = v.project(X.reshape(-1, 3))
    uu = np.clip(np.round(uv[:, 0]).astype(int), 0, shape[1] - 1)
    vv = np.clip(np.round(uv[:, 1]).astype(int), 0, shape[0] - 1)
    return m[vv, uu].reshape(X.shape[:2])


def main():
    a = sys.argv[1:]
    stat_cli = a[a.index("--stat") + 1] if "--stat" in a else None
    suffix = a[a.index("--suffix") + 1] if "--suffix" in a else ""
    k_cli = int(a[a.index("--k") + 1]) if "--k" in a else None
    sess_cli = a[a.index("--sessions") + 1].split(",") if "--sessions" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--")
           and (i == 0 or a[i - 1] not in ("--stat", "--suffix", "--k", "--sessions"))]
    run = ROOT / pos[0]
    P = tomllib.loads((ROOT / "config" / "model" / "chassis.toml").read_text())
    S = surfaces(P)
    names = pos[1:] or list(S)
    split = json.loads((ROOT / "data" / "split.json").read_text())
    train = set(split["train"])
    sess_of = {v: s for s, vs in split["sessions"].items() for v in vs}
    V = load_views(4)
    intr, img_dir = full_res_cams()
    id_map = json.loads((run / "id_map.json").read_text())
    roll = P["body"].get("roll_deg", 0.0)
    for nm in names:
        o = P.get("tex", {}).get(nm, {})
        if o.get("skip") and not pos[1:]:
            continue
        sess = set(sess_cli or o.get("sessions", ["S1", "S2", "S3"]))
        stat = stat_cli or o.get("stat", "median")
        k, ppm, min_cos = int(k_cli or o.get("k", 5)), float(o.get("px_per_mm", 4)), float(o.get("min_cos", 0.25))
        corners = [roll_to_world(p, roll) for p in S[nm]["corners"]]
        X, N = surface_grid({"type": "quad", "corners_mm": corners}, ppm)
        cols = [np.array(id_map[n][::-1]) for n in S[nm]["owners"] if n in id_map]
        views = sorted(v for v in train if sess_of.get(v) in sess and (run / "id" / f"{v}.png").exists())
        if o.get("views"):  # explicit view list (still train and session filtered)
            views = [v for v in views if v in set(o["views"])]
        if o.get("sign_mask_views_only") or "--sign-views" in a:  # only views whose sign SAM mask exists
            views = [v for v in views if (ROOT / "data" / "sam_s2" / "sign" / f"{v}.png").exists()]
        if "--dry-run" in a:
            print(nm, len(views), "views", stat, X.shape[:2]); continue
        texs, scores, vnames = [], [], []
        for vn in views:
            v = V[vn]
            d = v.center[None, None] - X
            dist = np.linalg.norm(d, axis=2)
            cosang = (d * N).sum(2) / dist
            idm = cv2.imread(str(run / "id" / f"{vn}.png"), cv2.IMREAD_UNCHANGED)
            vis = (cosang > min_cos) & owned_mask(v, X, idm, cols, 2) & ~sign_mask(v, X, idm.shape[:2])
            if vis.mean() < 0.002:
                continue
            t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
            vis &= inb
            texs.append(t)
            vnames.append(vn)
            scores.append(np.where(vis, cosang * intr[v.camera_id][0] / dist, -1.0))
        if not texs:
            print(json.dumps({"name": nm, "error": "no view sees it"})); continue
        A = np.stack(texs).astype(np.float32)
        Sc = np.stack(scores)
        order = np.argsort(-Sc, axis=0)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(len(texs))[:, None, None].repeat(Sc.shape[1], 1).repeat(Sc.shape[2], 2), 0)
        M = (Sc > 0) & (rank < k)
        A[~M] = np.nan
        q = 50.0 if stat == "median" else float(stat[1:])
        with np.errstate(all="ignore"):
            if q == 50.0:
                B = np.nanmedian(A, axis=0)
            else:  # percentile of luminance, keep that view's color
                L = A.mean(3)
                Lq = np.nanpercentile(L, q, axis=0)
                pick = np.nanargmin(np.where(np.isnan(L), np.inf, np.abs(L - Lq[None])), axis=0)
                B = np.take_along_axis(A, pick[None, ..., None].repeat(3, 3), 0)[0]
        hole = np.isnan(B).any(2)
        B[hole] = np.nanmedian(B[~hole], axis=0)
        B = np.clip(B, 0, 255).astype(np.uint8)
        out = ROOT / "assets" / "textures" / f"chassis_{nm}{suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), B)
        cov = np.clip(M.sum(0) * 40, 0, 255).astype(np.uint8)
        (ROOT / "outputs" / "textures").mkdir(parents=True, exist_ok=True)
        sheet = np.hstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)]) if B.shape[0] > B.shape[1] else \
            np.vstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(str(ROOT / "outputs" / "textures" / f"chassis_{nm}{suffix}_views.jpg"), sheet)
        share = {vnames[i]: round(float(M[i].mean()), 3) for i in range(len(texs)) if M[i].any()}
        by_sess = {}
        for vn_, sh in share.items():
            by_sess[sess_of[vn_]] = round(by_sess.get(sess_of[vn_], 0) + sh, 2)
        print(json.dumps({"name": nm, "texture": str(out.relative_to(ROOT)), "size": list(B.shape[:2]),
                          "hole_frac": round(float(hole.mean()), 3), "stat": stat, "sessions": sorted(sess),
                          "texel_share_by_session": by_sess,
                          "views": dict(sorted(share.items(), key=lambda kv: -kv[1])[:8])}))


if __name__ == "__main__":
    main()
