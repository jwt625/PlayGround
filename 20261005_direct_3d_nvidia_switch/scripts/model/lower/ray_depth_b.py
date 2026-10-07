"""Braided cable centerline by Viterbi over the braid mask in many train views (DevLog-002-lower 5d).

  uv run python scripts/model/lower/ray_depth_b.py braid_b --y 112,312 --step 4 --half 8 --lam 15 [--write]

The cable runs monotonically in Y, so the centerline is solved at Y stations. State per station = (dx, dz) offset
on a grid of +-half mm (0.5 mm) around the current centerline (tubegeom.centerline): this is the corridor guard.
Data cost = sum over train views of the braid-mask cost (braidmask.cost_map: distance to the mask outside it,
minus half the depth into it inside it, capped) at the projected point; smoothness = lam x L1 change of (dx, dz)
between stations. --write replaces the cable's pts in config/model/lower.toml by the solved points (every
--every stations) plus the unchanged points beyond the solved range, and zeroes the old offset keys.
Writes a debug sheet outputs/scratch/lower/<key>_viterbi.jpg (old path red, new path green, mask tinted).
Refuses holdout views."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.geom import load_views  # noqa: E402

import tubegeom  # noqa: E402
from braidmask import braid_mask, cost_map, teal_mask  # noqa: E402

VIEWS = "IMG_5713,IMG_5721,IMG_5722,IMG_5723,IMG_5724,IMG_5725,IMG_5726,IMG_5730,IMG_5731,IMG_5735,IMG_5736,IMG_5749,IMG_5752,IMG_5753,IMG_5758"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--views", default=VIEWS)
    ap.add_argument("--y", default="112,312")
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--half", type=float, default=8.0)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=15.0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--cue", default="speckle", choices=["speckle", "teal"], help="braid_mask (B) or teal_mask (A)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    hold = set(json.loads((ROOT / "data/split.json").read_text())["holdout"])
    views = a.views.split(",")
    if hold & set(views):
        sys.exit(f"refused: holdout views {sorted(hold & set(views))}")
    P = tomllib.loads((ROOT / "config/model/lower.toml").read_text())
    C = tubegeom.centerline(P, a.key)
    y0, y1 = map(float, a.y.split(","))
    ys = np.arange(y0, y1 + 1e-6, a.step)
    base = np.stack([np.interp(ys, C[:, 1], C[:, 0]), ys, np.interp(ys, C[:, 1], C[:, 2])], 1)
    g = np.arange(-a.half, a.half + 1e-6, a.res)
    DX, DZ = np.meshgrid(g, g, indexing="ij")
    DX, DZ = DX.ravel(), DZ.ravel()
    V = load_views(2)
    cost = np.zeros((len(ys), len(DX)))
    used = []
    for nm in views:
        v = V[nm]
        _, m, _ = (teal_mask if a.cue == "teal" else braid_mask)(nm)
        cm = cost_map(m)
        X = np.stack([base[:, None, 0] + DX[None], np.broadcast_to(base[:, None, 1], (len(ys), len(DX))),
                      base[:, None, 2] + DZ[None]], 2).reshape(-1, 3)
        uv, z = v.project(X / 1e3)
        u, w = uv[:, 0], uv[:, 1]
        inb = (z > 0) & (u >= 0) & (w >= 0) & (u < cm.shape[1] - 1) & (w < cm.shape[0] - 1)
        if inb.mean() < 0.5:
            continue
        c = np.full(len(u), 0.0, np.float32)
        c[inb] = cm[w[inb].astype(int), u[inb].astype(int)]
        cost += c.reshape(len(ys), len(DX))
        used.append(nm)
    # Viterbi with L1 transition lam * (|dx_i - dx_j| + |dz_i - dz_j|)
    T = a.lam * (np.abs(DX[:, None] - DX[None]) + np.abs(DZ[:, None] - DZ[None]))
    acc = cost[0].copy()
    back = np.zeros((len(ys), len(DX)), int)
    for i in range(1, len(ys)):
        tot = acc[:, None] + T  # from j (rows) to k (cols)
        back[i] = tot.argmin(0)
        acc = tot.min(0) + cost[i]
    path = [int(acc.argmin())]
    for i in range(len(ys) - 1, 0, -1):
        path.append(back[i][path[-1]])
    path = path[::-1]
    new = base.copy()
    new[:, 0] += DX[path]
    new[:, 2] += DZ[path]
    at_edge = np.mean((np.abs(DX[path]) >= a.half - 1e-6) | (np.abs(DZ[path]) >= a.half - 1e-6))
    print(f"views used {len(used)}: {','.join(used)}")
    print(f"mean cost per view before {cost[np.arange(len(ys)), np.argmin(np.abs(DX) + np.abs(DZ))].mean() / len(used):.1f}"
          f" after {cost[np.arange(len(ys)), path].mean() / len(used):.1f}; at corridor edge {at_edge:.2f}")
    for i in range(0, len(ys), a.every):
        print(f"Y {ys[i]:6.1f}: old ({base[i, 0]:7.1f}, {base[i, 2]:6.1f}) -> new ({new[i, 0]:7.1f}, {new[i, 2]:6.1f})")
    # debug sheet on 4 views
    tiles = []
    for nm in used[:: max(1, len(used) // 6)][:6]:
        v = V[nm]
        img, m, _ = (teal_mask if a.cue == "teal" else braid_mask)(nm)
        im = img.copy()
        im[m] = (0.6 * im[m] + [0, 0, 100]).astype(np.uint8)
        for L, col in ((base, (0, 0, 255)), (new, (0, 255, 0))):
            uv, _ = v.project(L / 1e3)
            cv2.polylines(im, [uv.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        uv, _ = v.project(base / 1e3)
        x0, yy0 = np.maximum(uv.min(0) - 80, 0).astype(int)
        x1, yy1 = (uv.max(0) + 80).astype(int)
        cr = im[yy0:yy1, x0:x1]
        s = 520 / max(cr.shape[:2])
        cr = cv2.resize(cr, None, fx=s, fy=s)
        t = np.zeros((540, 540, 3), np.uint8)
        t[: cr.shape[0], : cr.shape[1]] = cr
        cv2.putText(t, nm, (4, 534), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    out = ROOT / f"outputs/scratch/lower/{a.key}{a.tag}_viterbi.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print(out.relative_to(ROOT))
    if a.write:
        # new pts: solved stations every --every (and the last), then the old raw points beyond the solved range
        keep = list(range(0, len(ys), a.every))
        if keep[-1] != len(ys) - 1:
            keep.append(len(ys) - 1)
        pts = [p for p in P[a.key]["pts"] if p[1] < y0 - 2]  # fixed anchors before the solved range
        pts += [[round(float(new[i, 0]), 1), round(float(new[i, 1]), 1), round(float(new[i, 2]), 1)] for i in keep]
        pts += [p for p in P[a.key]["pts"] if p[1] > y1 + 8]
        f = ROOT / "config/model/lower.toml"
        s = f.read_text()
        sec = s.index(f"[{a.key}]")
        end = s.find("\n[", sec + 1)
        body = s[sec:end]
        body = re.sub(r"\npts = \[.*\]\]", "\npts = [" + ", ".join(f"[{x}, {y}, {z}]" for x, y, z in pts) + "]", body)
        for k in ("dx", "dz", "dz_slope", "kx125", "kx200", "kx275", "kz125", "kz200", "kz275"):
            body = re.sub(rf"\n{k} = [-0-9.]+", f"\n{k} = 0.0", body)
        s = s[:sec] + body + s[end:]
        tomllib.loads(s)
        f.write_text(s)
        print(f"wrote {len(pts)} pts to [{a.key}]")


if __name__ == "__main__":
    main()
