"""Bake the lower group's photo textures ([[tex]] in config/model/lower.toml) from ID-owned photo pixels.

  uv run python scripts/model/lower/bake_lower.py --id-run outputs/runs/lower_idtrain [--only board,floor]
      [--session S1|S2|S3|all] [--stat median|p25|p35] [--n-best 5] [--n-rank 16] [--ppm 8] [--suffix _x]

Per surface: train views (never holdout) ranked by cos x px/mm at the patch center (bake_id_owned.rank_train_views),
filtered to one capture session (sun shadows moved between sessions); a texel is used from a view only if the
surface's quads own the pixel in the ID render of the current model (bake_id_owned.owned_mask); per texel the
statistic over its first n_best visible views. Never-seen texels get the median of the seen ones.
World corners = local rect at local z, residual [tilt] dzdy shear, then the "tray" frame (as build.py).
Writes assets/textures/<file stem><suffix>.png and outputs/textures/<stem><suffix>_views.jpg (bake | coverage).
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
from tools.bake_id_owned import owned_mask, rank_train_views  # noqa: E402
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, surface_grid  # noqa: E402


def sample_view(v, params, img_dir, X):
    """As tools.texture_from_photos.sample_view, but finds the original by stem (sources here are .JPG; the
    shared tool looks for .jpeg and fails; reported in DevLog-002-lower)."""
    f, cx, cy, k1 = params
    Xc = X.reshape(-1, 3) @ v.R.T + v.t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    u = (f * x * d + cx - 0.5).astype(np.float32).reshape(X.shape[:2])
    w = (f * y * d + cy - 0.5).astype(np.float32).reshape(X.shape[:2])
    path = next(p for p in Path(img_dir).glob(f"{v.name}.*") if p.suffix.lower() in (".jpg", ".jpeg"))
    src = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tex = cv2.remap(src, u, w, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    inb = (u >= 0) & (w >= 0) & (u < src.shape[1] - 1) & (w < src.shape[0] - 1) & (z.reshape(u.shape) > 0)
    return tex, inb


sys.path.insert(0, str(Path(__file__).resolve().parent))
import tubegeom  # noqa: E402


def tube_grid(P, key, ppm, n_around=16):
    """Texel grid of a cable (rows = angle from 0 at the top, cols = along); world mm -> meters, normals."""
    C = tubegeom.centerline(P, key)
    r = P[key]["r"]
    s = tubegeom.arclen(C)
    T, N1, N2 = tubegeom.frames(C)
    w = max(2, int(round(s[-1] * ppm)))
    h = max(2, int(round(2 * np.pi * r * ppm)))
    sc = (np.arange(w) + 0.5) / w * s[-1]
    Ci = np.stack([np.interp(sc, s, C[:, k]) for k in range(3)], 1)
    n1 = np.stack([np.interp(sc, s, N1[:, k]) for k in range(3)], 1)
    n2 = np.stack([np.interp(sc, s, N2[:, k]) for k in range(3)], 1)
    a = 2 * np.pi * (np.arange(h) + 0.5) / h
    Nrm = np.cos(a)[:, None, None] * n1[None] + np.sin(a)[:, None, None] * n2[None]
    Nrm /= np.linalg.norm(Nrm, axis=2, keepdims=True)
    return (Ci[None] + r * Nrm) / 1e3, Nrm


def _tube_stack(a, X, Nrm, obj, views, run, id_map, V, intr, img_dir):
    """Per texel statistic over its best n_best views (cos x px/mm) among `views`; returns (B float, hole)."""
    col = [np.array(id_map[obj][::-1])]
    texs, vis, score = [], [], []
    for nm in views:
        v = V[nm]
        d = v.center[None, None] - X
        dist = np.linalg.norm(d, axis=2)
        cos = (d * Nrm).sum(2) / dist
        idm = cv2.imread(str(run / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        own = owned_mask(v, X, idm, col, a.erode) & (cos > 0.3)
        if own.sum() < 50:
            continue
        tx, inb = sample_view(v, intr[v.camera_id], img_dir, X)
        texs.append(tx)
        vis.append(own & inb)
        score.append(np.where(own & inb, cos * intr[v.camera_id][0] / (dist * 1e3), 0))
    if not texs:
        return None, np.ones(X.shape[:2], bool), 0
    A = np.stack(texs).astype(np.float32)
    M = np.stack(vis)
    S = np.stack(score)
    rank = np.argsort(-S, axis=0).argsort(axis=0)
    M &= rank < a.n_best
    A[~M] = np.nan
    q = 50 if a.stat == "median" else float(a.stat[1:])
    with np.errstate(all="ignore"):
        B = np.nanpercentile(A, q, axis=0)
    return B, np.isnan(B).any(2), len(texs)


def bake_tubes(a, P, sess, hold, run, id_map, have, V, intr, img_dir, fill=None):
    for key in ("braid_a", "braid_b"):
        if a.only and key not in a.only.split(","):
            continue
        X, Nrm = tube_grid(P, key, a.ppm_tube)
        obj = f"lower.{key}"
        views = [v for v in sorted(have) if v in sess and v not in hold and v in V]
        B, hole, nv = _tube_stack(a, X, Nrm, obj, views, run, id_map, V, intr, img_dir)
        hole0 = float(hole.mean())
        if fill:  # texels the primary session never saw: same statistic over the fill session's views
            fv = [v for v in sorted(have) if v in fill and v not in hold and v in V]
            B2, hole2, nv2 = _tube_stack(a, X, Nrm, obj, fv, run, id_map, V, intr, img_dir)
            if B2 is not None:
                B[hole] = B2[hole]
                hole = hole & hole2
        B[hole] = np.nanmedian(B[~hole], axis=0)
        B = np.clip(B, 0, 255).astype(np.uint8)
        stem = Path(P[key]["tex"]).stem + a.suffix
        out = ROOT / "assets/textures" / f"{stem}.png"
        cv2.imwrite(str(out), B)
        cv2.imwrite(str(ROOT / "outputs/textures" / f"{stem}_views.jpg"), B)
        print(json.dumps({"tube": key, "file": str(out.relative_to(ROOT)), "size": list(B.shape[:2]),
                          "hole_primary": round(hole0, 3), "hole": round(float(hole.mean()), 3), "views": nv}))


def world(P, x, y, z):
    """Level front-bay coordinates (mm) -> world: residual dzdy shear, then the "tray" frame (as build.py)."""
    t = P.get("tilt", {})
    if t.get("dzdy", 0.0):
        z = z + t["dzdy"] * (y - t["y0"])
    fr = json.loads((ROOT / "config/frames.json").read_text())["frames"]["tray"]
    p = np.array(fr["R"]) @ np.array([x, y, z], float) + np.array(fr["t_mm"])
    return p.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-run", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--session", default="S1")
    ap.add_argument("--stat", default="median")
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--n-rank", type=int, default=16)
    ap.add_argument("--ppm", type=float, default=8.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--tubes", action="store_true", help="bake the braided cables instead of the flat surfaces")
    ap.add_argument("--ppm-tube", type=float, default=4.0)
    ap.add_argument("--see-through", default="", help="flat bakes: object names/prefixes whose pixels count as owned")
    ap.add_argument("--exclude", default="", help="comma list of train views never used (out-of-sample checks)")
    ap.add_argument("--fill-session", default="", help="fill texels the primary session never saw from these sessions (S1+S2: in order for flat bakes)")
    a = ap.parse_args()
    P = tomllib.loads((ROOT / "config/model/lower.toml").read_text())
    split = json.loads((ROOT / "data/split.json").read_text())
    hold = set(split["holdout"]) | set(filter(None, a.exclude.split(",")))  # excluded views treated like holdout
    sess = set(split["train"]) if a.session == "all" else set(split["sessions"][a.session])
    fill = set().union(*[split["sessions"][f] for f in a.fill_session.split("+")]) if a.fill_session else None
    run = ROOT / a.id_run
    id_map = json.loads((run / "id_map.json").read_text())
    have = {p.stem for p in (run / "id").glob("*.png")}
    V = load_views(4)
    intr, img_dir = full_res_cams()
    only = set(a.only.split(",")) if a.only else None
    if a.tubes:
        (ROOT / "outputs/textures").mkdir(parents=True, exist_ok=True)
        bake_tubes(a, P, sess, hold, run, id_map, have, V, intr, img_dir, fill)
        return
    for t in P["tex"]:
        if only and t["name"] not in only:
            continue
        x0, x1, y0, y1 = t["rect"]
        z = t["z"] + t.get("lift", 0.15)
        C = [world(P, x0, y1, z), world(P, x1, y1, z), world(P, x1, y0, z), world(P, x0, y0, z)]  # TL TR BR BL
        X, N = surface_grid({"type": "quad", "corners_mm": C}, t.get("ppm", a.ppm))
        ranked = rank_train_views(X, N, V, intr, 40)
        obj = f"lower.tex_{t['name']}"
        if obj not in id_map:
            print(f"{t['name']}: {obj} not in ID run, skipped")
            continue
        col = [np.array(id_map[obj][::-1])]
        if a.see_through:  # pixels of these objects count as owned (they lie on/above the surface, e.g. fibers
            col += [np.array(c[::-1]) for n, c in id_map.items()  # on the floor): bakes the photo through them
                    if any(n == p or n.startswith(p) for p in a.see_through.split(","))]

        def stack(sset):
            vs = [v for v in ranked if v in sset and v in have and v not in hold][: a.n_rank]
            texs, vis = [], []
            for nm in vs:
                v = V[nm]
                tx, inb = sample_view(v, intr[v.camera_id], img_dir, X)
                idm = cv2.imread(str(run / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
                texs.append(tx)
                vis.append(inb & owned_mask(v, X, idm, col, a.erode))
            if not texs:
                return None, None, vs
            A = np.stack(texs).astype(np.float32)
            M = np.stack(vis)
            M &= np.cumsum(M, axis=0) <= a.n_best
            A[~M] = np.nan
            q = 50 if a.stat == "median" else float(a.stat[1:])
            with np.errstate(all="ignore"):
                Bq = np.nanpercentile(A, q, axis=0)
            return Bq, M, vs

        B, M, views = stack(sess)
        if B is None:
            print(f"{t['name']}: no views in session {a.session}")
            continue
        hole = np.isnan(B).any(2)
        hole0 = float(hole.mean())
        for fs in (a.fill_session.split("+") if a.fill_session else []):  # holes from other sessions, in order
            B2, M2, _ = stack(set(split["sessions"][fs]))
            if B2 is not None:
                B[hole] = B2[hole]
                M = np.concatenate([M, M2])
                hole = np.isnan(B).any(2)
        if hole.all():
            print(f"{t['name']}: nothing visible")
            continue
        B[hole] = np.nanmedian(B[~hole], axis=0)
        B = np.clip(B, 0, 255).astype(np.uint8)
        stem = t.get("stem", Path(t["file"]).stem) + a.suffix  # "stem": base name when "file" points at a variant
        out = ROOT / "assets/textures" / f"{stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), B)
        cov = np.clip(M.sum(0) * 40, 0, 255).astype(np.uint8)
        (ROOT / "outputs/textures").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ROOT / "outputs/textures" / f"{stem}_views.jpg"),
                    np.hstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)]))
        rec = {"tex": t["name"], "file": str(out.relative_to(ROOT)), "size": list(B.shape[:2]), "session": a.session,
               "fill": a.fill_session, "see_through": a.see_through, "exclude": a.exclude, "id_run": a.id_run,
               "hole_primary": round(hole0, 3), "hole": round(float(hole.mean()), 3), "views": views}
        print(json.dumps(rec))
        (ROOT / "outputs/textures" / f"{stem}_views.json").write_text(json.dumps(rec, indent=1))  # provenance


if __name__ == "__main__":
    main()
