"""Bake top-face photo textures for upper parts (entries with tex = "<t>" in config/model/upper.toml).

  uv run python scripts/model/upper/bake_upper.py <id_run> [t1,t2,...] [--suffix _x] [--stat p25] [--session S3] [--n-best 5] [--erode 1]

Per texel: the first n_best train views (ranked by cos x px/mm at the patch center, optionally limited to one
capture session) in which the part owns the pixel in <id_run>'s ID renders (so modeled occluders never paint onto
the surface); value = median (stat "median") or a percentile ("p25") of those samples. Never-seen texels get the
median of the seen ones. Options per texture in [texbake.<t>] (ppm, n_best, session, stat, erode_px, cap = index
of the [[cyls]] instance to sample); command-line flags override. Boxes with tex_sides = true also get their four
side faces baked to upper_<t>_<xp|xn|yp|yn>.png (same options as <t>). dark_x / dark_q: texels with
|X| > dark_x mm use percentile dark_q (default 10) instead (bright wall parallax baked into a board edge strip). Output assets/textures/upper_<t><suffix>.png and
outputs/textures/upper_<t><suffix>_views.jpg (bake | seen-count x 40). Holdout views are never used.
"""

from __future__ import annotations

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

_IMG = {}


def sample_view(v, params, img_dir, X):
    """Copy of tools.texture_from_photos.sample_view that finds the original by stem (that one hard-codes
    ".jpeg"; this capture has .JPG). Originals cached."""
    f, cx, cy, k1 = params
    Xc = X.reshape(-1, 3) @ v.R.T + v.t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + k1 * (x * x + y * y)
    u = (f * x * d + cx - 0.5).astype(np.float32).reshape(X.shape[:2])
    w = (f * y * d + cy - 0.5).astype(np.float32).reshape(X.shape[:2])
    if v.name not in _IMG:
        p = next(q for q in Path(img_dir).iterdir() if q.stem == v.name)
        _IMG[v.name] = cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    src = _IMG[v.name]
    tex = cv2.remap(src, u, w, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    inb = (u >= 0) & (w >= 0) & (u < src.shape[1] - 1) & (w < src.shape[0] - 1) & (z.reshape(u.shape) > 0)
    return tex, inb

SESS = {"S1": (5711, 5740), "S2": (5741, 5747), "S3": (5748, 5758)}


def entries(P):
    for kind in ("box", "prism", "cyls", "cyl"):
        for e in P.get(kind, []):
            if e.get("tex") and not e.get("tex_shared"):
                yield kind, e


def quad(kind, e, cap):
    if kind == "box":
        x0, x1, y0, y1, _, z1 = e["b"]
    elif kind == "prism":
        xs, ys = [q[0] for q in e["pts"]], [q[1] for q in e["pts"]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if "plane" in e:  # slanted top: corners on the plane (the bilinear quad stays planar)
            a, b, c = e["plane"]
            return [[x, y, a + b * x + c * y] for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]
        z1 = e["z"][1]
    else:
        x, y = e["xy"][cap]
        r = e["r"]
        x0, x1, y0, y1, z1 = x - r, x + r, y - r, y + r, e["z"][1]
    return [[x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [x0, y0, z1]]


def side_quad(b, side):
    """Box side as a label quad TL, TR, BR, BL (up = +Z; normal = outward); matches build.side_uv."""
    x0, x1, y0, y1, z0, z1 = b
    return {"xp": [[x1, y0, z1], [x1, y1, z1], [x1, y1, z0], [x1, y0, z0]],
            "xn": [[x0, y1, z1], [x0, y0, z1], [x0, y0, z0], [x0, y1, z0]],
            "yp": [[x1, y1, z1], [x0, y1, z1], [x0, y1, z0], [x1, y1, z0]],
            "yn": [[x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]]}[side]


AX = {"x": (0, 1, 0, 0, 0, 1), "y": (1, 0, 0, 0, 0, -1), "z": (1, 0, 0, 0, 1, 0)}  # axis -> ref_dir r0, r1 = ax x r0


def cyl_surface(axis, c, r, h):
    """Cylinder side as a texture_from_photos 'cylinder' surface: u = angle -180..180 deg from r0 toward r1,
    rows = along the axis from +h/2 (row 0) to -h/2; matches build.cyl_uv."""
    ax = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    base = [c[i] - ax[i] * h / 2 for i in range(3)]
    return {"type": "cylinder", "base_mm": base, "axis": ax, "ref_dir": list(AX[axis][:3]), "radius_mm": r,
            "angle_deg": [-180, 180], "length_mm": [0, h]}


def jobs(P, opts):
    """(texture file stem, base texture key for options, surface spec, entry)."""
    for kind, e in entries(P):
        t = e["tex"]
        o = opts.get(t, {})
        if kind == "cyl":  # side only
            yield f"{t}_side", t, cyl_surface(e.get("axis", "z"), e["c"], e["r"], e["h"]), e
            continue
        yield t, t, {"type": "quad", "corners_mm": quad(kind, e, int(o.get("cap", 0)))}, e
        if kind == "cyls" and e.get("tex_sides"):  # instanced side texture sampled on instance `cap`
            x, y = e["xy"][int(o.get("cap", 0))]
            z0, z1 = e["z"]
            yield f"{t}_side", t, cyl_surface("z", [x, y, (z0 + z1) / 2], e["r"], z1 - z0), e
        if kind == "box" and e.get("tex_sides"):
            for sd in ("xp", "xn", "yp", "yn"):
                yield f"{t}_{sd}", t, {"type": "quad", "corners_mm": side_quad(e["b"], sd)}, e


def main():
    args = [a for a in sys.argv[1:]]
    flags = {}
    for k in ("--suffix", "--stat", "--session", "--n-best", "--erode"):
        if k in args:
            i = args.index(k)
            flags[k] = args[i + 1]
            del args[i:i + 2]
    run = ROOT / args[0]
    only = set(args[1].split(",")) if len(args) > 1 else None
    P = tomllib.loads((ROOT / "config" / "model" / "upper.toml").read_text())
    opts = P.get("texbake", {})
    V = load_views(4)
    intr, img_dir = full_res_cams()
    id_map = json.loads((run / "id_map.json").read_text())
    for t, tb, surf, e in jobs(P, opts):
        if only and t not in only and tb not in only:
            continue
        o = dict(opts.get(tb, {}))
        sess = flags.get("--session", o.get("session", "all"))
        stat = flags.get("--stat", o.get("stat", "median"))
        nb = int(flags.get("--n-best", o.get("n_best", 5)))
        X, N = surface_grid(surf, float(o.get("ppm", 6)))
        cylside = surf["type"] == "cylinder"
        if cylside:  # all train views, nearest first; per-texel facing test below
            ctr = X.reshape(-1, 3).mean(0)
            train = json.loads((ROOT / "data" / "split.json").read_text())["train"]
            views = sorted(train, key=lambda nm: float(np.linalg.norm(V[nm].center - ctr)))
        else:
            views = rank_train_views(X, N, V, intr, 30)
        if sess != "all":
            a, b = SESS[sess]
            views = [v for v in views if a <= int(v[4:]) <= b]
        views = [v for v in views if (run / "id" / f"{v}.png").exists()]
        names = [f"upper.{e['name']}"]
        cols = [np.array(id_map[n][::-1]) for n in names if n in id_map]
        if not cols or not views:
            print(json.dumps({"tex": t, "error": "no object color or no views", "views": views}))
            continue
        texs, vis = [], []
        for nm in views:
            v = V[nm]
            tx, inb = sample_view(v, intr[v.camera_id], img_dir, X)
            idm = cv2.imread(str(run / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
            texs.append(tx)
            ok = inb & owned_mask(v, X, idm, cols, int(flags.get("--erode", o.get("erode_px", 1))))
            if cylside:  # texel must face the camera (the back side projects onto the same object's pixels)
                d = v.center[None, None] - X
                ok &= (N * d).sum(2) > 0.3 * np.linalg.norm(d, axis=2)
            vis.append(ok)
        A = np.stack(texs).astype(np.float32)
        M = np.stack(vis)
        M &= np.cumsum(M, axis=0) <= nb
        A[~M] = np.nan
        q = 50 if stat == "median" else int(stat[1:])
        with np.errstate(all="ignore"):
            B = np.nanpercentile(A, q, axis=0)
            if "dark_x" in o:  # texels with |X| > dark_x (mm) take a dark percentile (wall parallax strips)
                sel = np.abs(X[..., 0] * 1e3) > float(o["dark_x"])
                B[sel] = np.nanpercentile(A[:, sel], int(o.get("dark_q", 10)), axis=0)
        hole = np.isnan(B).any(2)
        if hole.all():
            print(json.dumps({"tex": t, "error": "no texel visible"}))
            continue
        B[hole] = np.nanmedian(B[~hole], axis=0)
        B = np.clip(B, 0, 255).astype(np.uint8)
        suf = flags.get("--suffix", "")
        out = ROOT / "assets" / "textures" / f"upper_{t}{suf}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), B)
        cov = np.clip(M.sum(0) * 40, 0, 255).astype(np.uint8)
        (ROOT / "outputs" / "textures").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ROOT / "outputs" / "textures" / f"upper_{t}{suf}_views.jpg"),
                    np.hstack([B, cv2.cvtColor(cov, cv2.COLOR_GRAY2BGR)]))
        used = [v for v, m in zip(views, M) if m.any()]
        print(json.dumps({"tex": t, "size": list(B.shape[:2]), "hole": round(float(hole.mean()), 3),
                          "session": sess, "stat": stat, "views_used": used}))


if __name__ == "__main__":
    main()
