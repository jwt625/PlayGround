"""Bake the planar photo texture of the package top (all up-facing package faces share one image over the
substrate square, UV = package xy). Each texel is sampled at its own 3D point (x, y, top height of the part at
(x, y), from config/model/package.toml), in every view where the ID render of the current model says that part
owns the pixel (eroded by 1 px at 1/4 scale). Per texel: median and a low percentile over the owned views.

  uv run python scripts/model/package/bake_top.py --run outputs/runs/<run with id pass> \
      --views IMG_5711,IMG_5712,IMG_5717,IMG_5740,IMG_5736 [--ppm 10] [--pct 25]
Writes assets/textures/package_top_med.png, package_top_p<pct>.png and outputs/textures/package_top_views.jpg
(per-view samples, owned texels only, plus the two blends). Train views only (holdout refused).
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
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, sample_view  # noqa: E402


def part_map(P: dict, X: np.ndarray, Y: np.ndarray):
    """Object name index and top height per texel, mirroring build.py."""
    S, R, F, O, L = P["substrate"], P["ring"], P["field"], P["oe"], P["lid"]
    hx, hy = S["size_x"] / 2, S["size_y"] / 2
    names = ["package.substrate", "package.ring", "package.field", "package.lid"]
    idx = np.zeros(X.shape, int)
    Z = np.zeros(X.shape)
    ax, ay = np.abs(X), np.abs(Y)
    ix, iy, cb = R["inner_x"], R["inner_y"], R["corner_block"]
    notch = (ax > hx - R["notch"]) & (ay > hy - R["notch"])
    opening = ((ax < ix) & (ay < cb)) | ((ax < cb) & (ay < iy))
    ring = ~notch & ~opening
    lip = ((ax > ix) & (ax < R["rim_x"]) & (ay < cb)) | ((ay > iy) & (ay < R["rim_y"]) & (ax < cb))
    idx[ring] = 1
    Z[ring] = np.where(lip[ring], R["lip_top"], R["top"])
    idx[opening] = 2
    Z[opening] = F["thickness"]
    k = O["count"]
    a = np.array([(i - (k - 1) / 2) * O["pitch"] for i in range(k)])
    for part, wkey, hkey, rk, ck in (("chip", "chip_width", "chip_height", "rows_chip_r", "cols_chip_r"),
                                     ("gold", "gold_width", "gold_height", "rows_gold_r", "cols_gold_r")):
        w2, h = O[wkey] / 2, O[hkey]
        (r0, r1), (c0, c1) = O[rk], O[ck]
        along_x = np.min(np.abs(X[..., None] - a), -1) < w2
        along_y = np.min(np.abs(Y[..., None] - a), -1) < w2
        for side, m in (("top", along_x & (Y > r0) & (Y < r1)), ("bottom", along_x & (Y < -r0) & (Y > -r1)),
                        ("right", along_y & (X > c0) & (X < c1)), ("left", along_y & (X < -c0) & (X > -c1))):
            names.append(f"package.oe_{part}_{side}")
            idx[m] = len(names) - 1
            Z[m] = h
    lid = (np.abs(X - L["cx"]) < L["size_x"] / 2) & (np.abs(Y - L["cy"]) < L["size_y"] / 2)
    idx[lid] = 3
    Z[lid] = L["height"]
    return names, idx, Z


def side_map(P: dict, ppm: float):
    """Outer-wall atlas: 4 bands (+y, -y, +x, -x) top to bottom, each from the ring top down to the substrate
    bottom; u runs along the side from -h to +h. Returns points (h, w, 3), names, idx (-1 = no surface)."""
    S, R = P["substrate"], P["ring"]
    hx, hy = S["size_x"] / 2, S["size_y"] / 2
    top, bot = R["top"], -S["thickness"]
    w, hb = int(round(2 * hx * ppm)), int(round((top - bot) * ppm))
    u = (np.arange(w) + 0.5) / w
    z = top - (np.arange(hb) + 0.5) / hb * (top - bot)
    U, Zb = np.meshgrid(u, z)
    names = ["package.ring", "package.substrate"]
    pts, idx, nrm = [], [], []
    for side in ("+y", "-y", "+x", "-x"):
        L = 2 * (hx if side[1] == "y" else hy)
        s_ = -L / 2 + U * L
        other = hy if side[1] == "y" else hx
        sign = 1 if side[0] == "+" else -1
        X3 = np.stack([s_, np.full_like(s_, sign * other), Zb], -1) if side[1] == "y" else \
            np.stack([np.full_like(s_, sign * other), s_, Zb], -1)
        notch = np.abs(s_) > L / 2 - R["notch"]
        k = np.where(Zb < 0, 1, np.where(notch, -1, 0))
        pts.append(X3)
        idx.append(k)
        nv = np.zeros(3)
        nv[0 if side[1] == "x" else 1] = sign
        nrm.append(np.broadcast_to(nv, X3.shape))
    return np.concatenate(pts, 0), names, np.concatenate(idx, 0), np.concatenate(nrm, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--pct", type=float, default=25.0)
    ap.add_argument("--name", default=None, help="output name prefix (default package_top / package_side)")
    ap.add_argument("--sides", action="store_true", help="bake the outer-wall atlas instead of the top")
    ap.add_argument("--min-cos", type=float, default=0.03, help="texel normal . view direction threshold")
    ap.add_argument("--full", action="store_true", help="sample the full-resolution originals (Lanczos) instead of s2")
    ap.add_argument("--region-view", action="append", default=[],
                    help="part=VIEW[,VIEW2]: texels of parts whose name contains <part> come from these views only "
                         "(first owned in order; others as usual), e.g. lid=IMG_5711")
    ap.add_argument("--cache", default=None, help="npz of per-view owned samples: reused when present, else written")
    ap.add_argument("--erode", type=int, default=1, help="ID ownership erosion (px at the ID pass scale)")
    a = ap.parse_args()
    hold = set(json.loads((ROOT / "data" / "split.json").read_text())["holdout"])
    views = a.views.split(",")
    if any(v in hold for v in views):
        raise SystemExit("holdout views are evaluation only")
    P = tomllib.loads((ROOT / "config" / "model" / "package.toml").read_text())
    fr = json.loads((ROOT / "config" / "frames.json").read_text())["frames"]["package"]
    R, t = np.array(fr["R"]), np.array(fr["t_mm"])
    hx, hy = P["substrate"]["size_x"] / 2, P["substrate"]["size_y"] / 2
    a.name = a.name or ("package_side" if a.sides else "package_top")
    if a.sides:
        X3, names, idx, N3 = side_map(P, a.ppm)
        h, w = idx.shape
    else:
        w, h = int(round(2 * hx * a.ppm)), int(round(2 * hy * a.ppm))
        X, Y = np.meshgrid(-hx + (np.arange(w) + 0.5) / a.ppm, hy - (np.arange(h) + 0.5) / a.ppm)  # row 0 = +y
        names, idx, Z = part_map(P, X, Y)
        X3 = np.stack([X, Y, Z], -1)
        N3 = np.broadcast_to(np.array([0.0, 0.0, 1.0]), X3.shape)
    W = (X3.reshape(-1, 3) @ R.T + t) / 1e3
    run = ROOT / a.run
    idmap = json.loads((run / "id_map.json").read_text())
    V2, V4 = load_views(2), load_views(4)
    intr, img_dir = full_res_cams()
    samples, owned_all = [], []
    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        z = np.load(cache)
        assert list(z["views"]) == views and z["S"].shape[1] == w * h, "cache does not match views/grid"
        samples = [z["S"][i].astype(np.float32) for i in range(len(views))]
        for sm in samples:
            sm[sm[:, 0] < 0] = np.nan
    for v in views if not samples else []:
        uv2, z2 = V2[v].project(W)
        idimg = cv2.imread(str(run / "id" / f"{v}.png"))[:, :, ::-1].astype(int)
        uv4 = uv2 if idimg.shape[1] == V2[v].width else V4[v].project(W)[0]   # ID pass at scale 2 or 4
        own = np.zeros(w * h, bool)
        cl = R.T @ (V2[v].center * 1e3 - t)               # camera center in the package frame (mm)
        d = cl - X3.reshape(-1, 3)
        facing = (N3.reshape(-1, 3) * d).sum(1) / np.linalg.norm(d, axis=1) > a.min_cos
        u4 = np.round(uv4[:, 0]).astype(int)
        v4 = np.round(uv4[:, 1]).astype(int)
        inb = (u4 > 0) & (v4 > 0) & (u4 < idimg.shape[1] - 1) & (v4 < idimg.shape[0] - 1) & (z2 > 0) & facing
        flat_idx = idx.ravel()
        for k, nm in enumerate(names):
            if nm not in idmap:
                continue
            m = (np.abs(idimg - np.array(idmap[nm])).sum(2) <= 3).astype(np.uint8)
            if a.erode > 0:
                m = cv2.erode(m, np.ones((2 * a.erode + 1, 2 * a.erode + 1), np.uint8))
            sel = inb & (flat_idx == k)
            own[sel] = m[v4[sel], u4[sel]] > 0
        if a.full:
            col, _ = sample_view(V2[v], intr[V2[v].camera_id], img_dir, W.reshape(h, w, 3))
            col = col.reshape(-1, 3).astype(np.float32)
        else:
            img = V2[v].image()
            col = cv2.remap(img, uv2[:, 0].reshape(h, w).astype(np.float32),
                            uv2[:, 1].reshape(h, w).astype(np.float32), cv2.INTER_LINEAR).reshape(-1, 3).astype(np.float32)
        col[~own] = np.nan
        samples.append(col)
        owned_all.append(own)
        print(v, "owned texels", f"{own.mean():.3f}")
    if a.sides:   # full-resolution debug sheet: per-view owned samples (unseen magenta), band borders
        dbg = [np.where(np.isnan(sm), [255, 0, 255], sm).astype(np.uint8).reshape(h, w, 3) for sm in samples]
        dbg = [cv2.putText(d.copy(), v, (5, 15), 0, 0.5, (0, 255, 255), 1) for d, v in zip(dbg, views)]
        cv2.imwrite(str(ROOT / "outputs" / "textures" / f"{a.name}_perview.jpg"), np.vstack(dbg))
    S = np.stack(samples)  # (nv, N, 3)
    if cache and not cache.exists():
        np.savez(cache, views=np.array(views), S=np.nan_to_num(S, nan=-1).astype(np.int16))
    region = {}   # texel mask -> single-view override (first owned view in the given order)
    for rv in a.region_view:
        part, vl = rv.split("=")
        sel = np.isin(idx.ravel(), [k for k, nm in enumerate(names) if part in nm])
        pick = np.full(len(sel), np.nan)
        pick3 = np.full((len(sel), 3), np.nan, np.float32)
        for vv in reversed(vl.split(",")):
            j = views.index(vv)
            ok = sel & np.isfinite(S[j, :, 0])
            pick3[ok] = S[j, ok]
            pick[ok] = j
        region[rv] = (sel & np.isfinite(pick), pick3)
        print(rv, "texels", int(sel.sum()), "covered", f"{np.isfinite(pick[sel]).mean():.3f}")
    cnt = np.isfinite(S[..., 0]).sum(0)
    out = {}
    for tag, q in (("med", 50.0), (f"p{int(a.pct)}", a.pct)):
        with np.errstate(all="ignore"):
            B = np.nanpercentile(S, q, axis=0)
        miss = cnt == 0
        B[miss] = 0
        for m_, p3 in region.values():
            B[m_] = p3[m_]
        if a.sides:   # unseen texels: median of the seen texels in the same row of any band (same part and z)
            hb = h // 4
            Bm, mm_ = B.reshape(h, w, 3), miss.reshape(h, w)
            for r in range(hb):
                rows = [r + k * hb for k in range(4)]
                seen = Bm[rows][~mm_[rows]]
                fill = np.median(seen, 0) if len(seen) else np.nanmedian(B[~miss], 0)
                for rr in rows:
                    Bm[rr][mm_[rr]] = fill
            B = np.clip(Bm, 0, 255).astype(np.uint8)
        else:
            B = np.clip(B, 0, 255).astype(np.uint8).reshape(h, w, 3)
            B = cv2.inpaint(B, miss.reshape(h, w).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        p = ROOT / "assets" / "textures" / f"{a.name}_{tag}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), B)
        out[tag] = B
        print(tag, p.relative_to(ROOT), "unseen", f"{miss.mean():.4f}")
    tiles = []
    for v, s in zip(views, samples):
        tl = np.nan_to_num(s, nan=0).astype(np.uint8).reshape(h, w, 3)
        tiles.append(cv2.putText(cv2.resize(tl, (w // 3, max(h // 3, 40))), v, (5, 20), 0, 0.6, (0, 255, 255), 2))
    for tag, B in out.items():
        tiles.append(cv2.putText(cv2.resize(B, (w // 3, max(h // 3, 40))), tag, (5, 20), 0, 0.6, (0, 255, 255), 2))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    (ROOT / "outputs" / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "outputs" / "textures" / f"{a.name}_views.jpg"), sheet)
    print("count of views per texel: min", cnt.min(), "median", np.median(cnt))


if __name__ == "__main__":
    main()
