"""Bake photo textures for every pcb component (boxes, cylinders) into per-part atlases.

Usage: uv run python scripts/model/pcb/bake_parts.py [--id-run outputs/runs/coord_idtrain] [--k 5] [--only a,b]
Needs an ID render of TRAIN views of the current model (render_views.py --passes id --views train). Holdout views
are never used (only views in data/split.json "train" that also have an ID image).

Per texel (surface point from scripts/model/pcb/uvparts.py): a view counts if the texel faces it (cos > --min-cos, default 0.15), is in
frame, and the ID pixel (1/4 scale, 1 px interior of the object region) belongs to that part (or a label of it).
The texel takes the median of its K best views by cos x px per mm, sampled from the 1/2-scale undistorted photos.
Never-seen texels get the part's material color. Writes assets/textures/pcb_parts/<name>.png and
outputs/textures/pcb_parts_sheet.jpg.
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import uvparts  # noqa: E402
from eval.evaluate import decode_id  # noqa: E402
from tools.geom import load_views  # noqa: E402


def lin2srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-run", default="outputs/runs/coord_idtrain")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--min-cos", type=float, default=0.15, help="views at grazing angles below this are skipped")
    a = ap.parse_args()
    P = tomllib.loads((ROOT / "config" / "model" / "pcb.toml").read_text())
    specs = uvparts.parts_from_params(P)
    if a.only:
        specs = [s for s in specs if s["name"] in a.only.split(",")]
    run = ROOT / a.id_run
    id_map = json.loads((run / "id_map.json").read_text())
    names = list(id_map)
    keys = np.array([(c[0] << 16) | (c[1] << 8) | c[2] for c in id_map.values()])
    obj2part = np.full(len(names) + 1, -1, np.int32)  # index -1 -> last slot (-1)
    for pi, s in enumerate(specs):
        for oi, n in enumerate(names):
            if n == f"pcb.{s['name']}" or n.startswith(f"pcb.label_{s['name']}_") or n in s.get("owners", []) or \
                    (s["name"] == "flyback" and n in ("pcb.label_flyback_warning", "pcb.label_flyback_side")):
                obj2part[oi] = pi
    # texels
    Xs, Ns, part, rows, cols, atl = [], [], [], [], [], []
    for pi, s in enumerate(specs):
        pl = uvparts.patches(s["prims"])
        W, H = uvparts.layout(pl)
        atl.append((W, H, pl))
        for p in pl:
            jj, ii = np.mgrid[0:p["h"], 0:p["w"]]
            X, N = p["fn"]((ii + 0.5) / p["w"], (jj + 0.5) / p["h"])
            Xs.append(X.reshape(-1, 3))
            Ns.append(np.ascontiguousarray(N).reshape(-1, 3))
            part.append(np.full(ii.size, pi))
            rows.append((jj + p["y0"]).ravel())
            cols.append((ii + p["x0"]).ravel())
    X = np.concatenate(Xs) / 1e3
    N = np.concatenate(Ns)
    part = np.concatenate(part)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = len(X)
    train = json.loads((ROOT / "data" / "split.json").read_text())["train"]
    views = [v for v in train if (run / "id" / f"{v}.png").exists()]
    V2 = load_views(2)
    K = a.k
    best_s = np.zeros((M, K))
    best_v = np.full((M, K), -1, np.int32)
    for vi, nm in enumerate(views):
        v = V2[nm]
        uv, z = v.project(X)
        d = v.center[None] - X
        dist = np.linalg.norm(d, axis=1)
        cos = (N * d).sum(1) / dist
        ok = (z > 0) & (cos > a.min_cos) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < v.width - 1) & \
             (uv[:, 1] < v.height - 1)
        idx = decode_id(run / "id" / f"{nm}.png", keys)
        f = idx.astype(np.float32)
        uni = (cv2.erode(f, np.ones((3, 3))) == f) & (cv2.dilate(f, np.ones((3, 3))) == f) & (idx >= 0)
        u4 = np.clip(np.round(uv[:, 0] / 2 - 0.25).astype(int), 0, idx.shape[1] - 1)
        v4 = np.clip(np.round(uv[:, 1] / 2 - 0.25).astype(int), 0, idx.shape[0] - 1)
        own = uni[v4, u4] & (obj2part[idx[v4, u4]] == part)
        ok &= own
        score = np.where(ok, cos * v.K[0, 0] / (dist * 1e3), 0.0)
        # insert into top-K
        S = np.concatenate([best_s, score[:, None]], 1)
        Vv = np.concatenate([best_v, np.full((M, 1), vi, np.int32)], 1)
        o = np.argsort(-S, axis=1)[:, :K]
        best_s = np.take_along_axis(S, o, 1)
        best_v = np.take_along_axis(Vv, o, 1)
    best_v[best_s <= 0] = -1
    C = np.full((M, K, 3), np.nan, np.float32)
    for vi in np.unique(best_v[best_v >= 0]):
        nm = views[vi]
        v = V2[nm]
        sel = np.nonzero((best_v == vi).any(1))[0]
        uv, _ = v.project(X[sel])
        img = v.image()
        n = len(sel)
        wdt = 4096  # remap needs dst dims < 32767: lay the samples out as a 2D map
        hgt = (n + wdt - 1) // wdt
        mx = np.full(hgt * wdt, -1, np.float32)
        my = np.full(hgt * wdt, -1, np.float32)
        mx[:n], my[:n] = uv[:, 0], uv[:, 1]
        smp = cv2.remap(img, mx.reshape(hgt, wdt), my.reshape(hgt, wdt), cv2.INTER_LINEAR)
        smp = smp.reshape(-1, 3)[:n].astype(np.float32)
        for k in range(K):
            m = best_v[sel, k] == vi
            C[sel[m], k] = smp[m]
    with np.errstate(all="ignore"):
        col = np.nanmedian(C, axis=1)
    seen = ~np.isnan(col).any(1)
    out_dir = ROOT / "assets" / "textures" / "pcb_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, stats = [], {}
    for pi, s in enumerate(specs):
        W, H, pl = atl[pi]
        m = part == pi
        base = lin2srgb(P["mats"][s["mat"]][:3])[::-1]  # BGR
        img = np.zeros((H, W, 3), np.float32) + base
        cc = np.where(seen[m, None], col[m], base)
        img[rows[m], cols[m]] = cc
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{s['name']}.png"), img)
        stats[s["name"]] = round(float(seen[m].mean()), 3)
        t = cv2.resize(img, (160, max(1, int(160 * H / W))), interpolation=cv2.INTER_AREA)
        t = cv2.copyMakeBorder(t, 14, 2, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(t, f"{s['name']} {stats[s['name']]:.2f}", (2, 11), 0, 0.35, (0, 255, 255), 1)
        tiles.append(t)
    # sheet: columns of tiles
    rows_img, line, lw = [], [], 0
    for t in tiles:
        line.append(t)
        if len(line) == 8:
            rows_img.append(line)
            line = []
    if line:
        rows_img.append(line)
    sheet_rows = []
    for ln in rows_img:
        hh = max(t.shape[0] for t in ln)
        ln = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in ln]
        r = np.hstack(ln)
        sheet_rows.append(cv2.copyMakeBorder(r, 0, 4, 0, 8 * 164 - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    (ROOT / "outputs" / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "outputs" / "textures" / "pcb_parts_sheet.jpg"), np.vstack(sheet_rows))
    print(json.dumps({"parts": len(specs), "texels": int(M), "views": len(views), "seen_frac": stats}))


if __name__ == "__main__":
    main()
