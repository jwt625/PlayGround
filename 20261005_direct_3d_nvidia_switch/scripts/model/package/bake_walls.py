"""Bake the package inner-wall atlas (near-vertical faces of the ring, field, OEs and lid; build.py _wall_charts):
each chart texel is sampled at its own 3D point on the face, in every view whose ID render says the face's object
owns the pixel (eroded 1 px) and that sees the face's front side (normal . view direction > --min-cos).
Per texel: --pct percentile over the owned views; unseen texels get the median of the seen texels of the same
object and outward direction class, or of the object, or the object's flat TOML color.

  PACKAGE_WALL_CHARTS=$PWD/outputs/scratch/package/wall_charts.json scripts/bslot.sh -b --factory-startup \
      --python scripts/blender/render_views.py -- --out outputs/runs/package_id2 --build scripts/model/build_all.py \
      --views IMG_5711,IMG_5712,IMG_5717,IMG_5736,IMG_5740 --passes id --scale 2
  uv run python scripts/model/package/bake_walls.py --run outputs/runs/package_id2 \
      --charts outputs/scratch/package/wall_charts.json --views IMG_5711,IMG_5712,IMG_5717,IMG_5736,IMG_5740
Writes assets/textures/package_walls_p<pct>.png (atlas W x H from the charts) and outputs/textures/
package_walls_views.jpg. Train views only.
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

FLAT = {"package.ring": "ring", "package.field": "field", "package.lid": "lid", "oe_gold": "oe", "oe_chip": "chip"}


def flat_bgr(P: dict, obj: str) -> np.ndarray:
    key = next((v for k, v in FLAT.items() if k in obj), "ring")
    lin = np.array(P["colors"][key], float)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (srgb[::-1] * 255).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--charts", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--pct", type=float, default=50.0)
    ap.add_argument("--min-cos", type=float, default=0.2)
    ap.add_argument("--name", default="package_walls")
    a = ap.parse_args()
    hold = set(json.loads((ROOT / "data" / "split.json").read_text())["holdout"])
    views = a.views.split(",")
    if any(v in hold for v in views):
        raise SystemExit("holdout views are evaluation only")
    P = tomllib.loads((ROOT / "config" / "model" / "package.toml").read_text())
    fr = json.loads((ROOT / "config" / "frames.json").read_text())["frames"]["package"]
    R, t = np.array(fr["R"]), np.array(fr["t_mm"])
    C = json.loads(Path(a.charts).read_text())
    ppm, W, H = C["ppm"], C["W"], C["H"]
    # texel grid per chart: 3D point, normal, object index, atlas pixel
    objs = sorted({c["object"] for c in C["charts"]})
    X, N, OB, PX, CH = [], [], [], [], []
    for ci, c in enumerate(C["charts"]):
        jj, ii = np.mgrid[0:c["hpx"], 0:c["wpx"]]
        su = np.clip((ii - 1 + 0.5) / ppm, 0, c["w"])
        sz = np.clip(c["h"] - (jj - 1 + 0.5) / ppm, 0, c["h"])
        o, tt, n = np.array(c["o"]), np.array(c["t"]), np.array(c["n"])
        p = o + su[..., None] * tt + sz[..., None] * np.array([0, 0, 1.0])
        X.append(p.reshape(-1, 3))
        N.append(np.broadcast_to(n, (p.shape[0] * p.shape[1], 3)))
        OB.append(np.full(p.shape[0] * p.shape[1], objs.index(c["object"])))
        PX.append(np.stack([(c["y0"] + jj).ravel(), (c["x0"] + ii).ravel()], 1))
        CH.append(np.full(p.shape[0] * p.shape[1], ci))
    X, N, OB, PX, CH = (np.concatenate(v) for v in (X, N, OB, PX, CH))
    Wd = (X @ R.T + t) / 1e3
    run = ROOT / a.run
    idmap = json.loads((run / "id_map.json").read_text())
    V2 = load_views(2)
    samples = []
    for v in views:
        uv2, z2 = V2[v].project(Wd)
        idimg = cv2.imread(str(run / "id" / f"{v}.png"))[:, :, ::-1].astype(int)
        uvi = uv2 if idimg.shape[1] == V2[v].width else load_views(4)[v].project(Wd)[0]
        cl = R.T @ (V2[v].center * 1e3 - t)
        d = cl - X
        facing = (N * d).sum(1) / np.linalg.norm(d, axis=1) > a.min_cos
        ui, vi = np.round(uvi[:, 0]).astype(int), np.round(uvi[:, 1]).astype(int)
        inb = (ui > 0) & (vi > 0) & (ui < idimg.shape[1] - 1) & (vi < idimg.shape[0] - 1) & (z2 > 0) & facing
        own = np.zeros(len(X), bool)
        for k, nm in enumerate(objs):
            if nm not in idmap:
                continue
            m = (np.abs(idimg - np.array(idmap[nm])).sum(2) <= 3).astype(np.uint8)
            m = cv2.erode(m, np.ones((3, 3), np.uint8))
            sel = inb & (OB == k)
            own[sel] = m[vi[sel], ui[sel]] > 0
        img = V2[v].image()
        npad = -len(X) % 4096
        mu = np.pad(uv2[:, 0], (0, npad), constant_values=-10).astype(np.float32).reshape(-1, 4096)
        mv = np.pad(uv2[:, 1], (0, npad), constant_values=-10).astype(np.float32).reshape(-1, 4096)
        col = cv2.remap(img, mu, mv, cv2.INTER_LINEAR).reshape(-1, 3)[:len(X)].astype(np.float32)
        col[~own] = np.nan
        samples.append(col)
        print(v, "owned texels", f"{own.mean():.3f}")
    S = np.stack(samples)
    cnt = np.isfinite(S[..., 0]).sum(0)
    with np.errstate(all="ignore"):
        B = np.nanpercentile(S, a.pct, axis=0)
    miss = cnt == 0
    # fill: same object and same outward direction (8 sectors), else object, else flat color
    sector = np.round(np.arctan2(N[:, 1], N[:, 0]) / (np.pi / 4)).astype(int) % 8
    for k, nm in enumerate(objs):
        mo = OB == k
        seen_o = mo & ~miss
        fo = np.median(B[seen_o], 0) if seen_o.sum() > 50 else flat_bgr(P, nm)
        for sc in range(8):
            ms = mo & (sector == sc)
            seen = ms & ~miss
            fill = np.median(B[seen], 0) if seen.sum() > 50 else fo
            B[ms & miss] = fill
        print(nm, "seen", f"{seen_o.sum() / max(mo.sum(), 1):.3f}")
    A = np.zeros((H, W, 3), np.uint8)
    A[PX[:, 0], PX[:, 1]] = np.clip(B, 0, 255).astype(np.uint8)
    out = ROOT / "assets" / "textures" / f"{a.name}_p{int(a.pct)}.png"
    cv2.imwrite(str(out), A)
    print(out.relative_to(ROOT), "unseen", f"{miss.mean():.3f}")
    dbg = np.zeros((H, W, 3), np.uint8)
    dbg[:] = (255, 0, 255)
    dbg[PX[~miss, 0], PX[~miss, 1]] = np.clip(B[~miss], 0, 255).astype(np.uint8)
    (ROOT / "outputs" / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "outputs" / "textures" / f"{a.name}_views.jpg"), dbg)


if __name__ == "__main__":
    main()
