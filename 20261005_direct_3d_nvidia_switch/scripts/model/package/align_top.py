"""Sharper package top texture: register each view's owned samples (bake_top.py --cache, full resolution) to a
reference view with dense optical flow (DIS), then blend per texel (percentile). Optional per-part single-view
overrides (e.g. the lid from the one view without sky reflections). Train views only (the cache holds them).

  uv run python scripts/model/package/align_top.py --cache outputs/scratch/package/top20_samples.npz \
      --ref IMG_5717 --pct 25 --name package_top20a [--region lid=IMG_5717] [--no-align]
Writes assets/textures/<name>_p<pct>.png and outputs/textures/<name>_flow.jpg (flow magnitude per view, texels).
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_top import part_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pct", type=float, default=25.0)
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--region", action="append", default=[], help="part=VIEW[,VIEW2]: single-view override")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--mode", default="affine", choices=["affine", "flow"])
    ap.add_argument("--views", default=None, help="subset of the cached views to blend (comma list)")
    a = ap.parse_args()
    z = np.load(a.cache)
    views = list(z["views"])
    P = tomllib.loads((ROOT / "config" / "model" / "package.toml").read_text())
    hx, hy = P["substrate"]["size_x"] / 2, P["substrate"]["size_y"] / 2
    w, h = int(round(2 * hx * a.ppm)), int(round(2 * hy * a.ppm))
    S = z["S"].reshape(len(views), h, w, 3).astype(np.float32)
    own = S[..., 0] >= 0
    if a.ref == "median":   # consensus reference: per-texel median of the unaligned owned samples
        with np.errstate(all="ignore"):
            ref = np.nanmedian(np.where(own[..., None], S, np.nan), axis=0)
        ref_own = np.isfinite(ref[..., 0])
        ref = np.nan_to_num(ref, nan=0)
        ri = -1
    else:
        ri = views.index(a.ref)
        ref, ref_own = S[ri].copy(), own[ri]
    refg = cv2.cvtColor(np.clip(np.where(ref_own[..., None], ref, 0), 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out, mags = [], []
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for i, v in enumerate(views):
        img = np.where(own[i][..., None], S[i], ref)        # unowned texels: reference values (zero flow there)
        img = np.where((~own[i] & ~ref_own)[..., None], 0, img)
        if a.no_align or i == ri:
            warped, ow = S[i], own[i]
            mags.append(np.zeros((h, w), np.float32))
        else:
            g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            if a.mode == "flow":
                flow = dis.calc(refg, g, None)            # ref(x) ~ view(x + flow)
                flow = cv2.GaussianBlur(flow, (0, 0), 6)   # smooth: registration error varies slowly
            else:   # one affine per view (ECC on a 1/2-size gray image, both views' owned texels)
                sc = 0.5
                r_ = cv2.resize(refg, None, fx=sc, fy=sc).astype(np.float32)
                g_ = cv2.resize(g, None, fx=sc, fy=sc).astype(np.float32)
                m_ = cv2.resize((own[i] & ref_own).astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
                Wm = np.eye(2, 3, dtype=np.float32)
                _, Wm = cv2.findTransformECC(cv2.GaussianBlur(r_, (0, 0), 1), cv2.GaussianBlur(g_, (0, 0), 1), Wm,
                                             cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                             m_, 5)
                Wm[:, 2] /= sc
                mx_ = Wm[0, 0] * gx + Wm[0, 1] * gy + Wm[0, 2]
                my_ = Wm[1, 0] * gx + Wm[1, 1] * gy + Wm[1, 2]
                flow = np.stack([mx_ - gx, my_ - gy], -1)
                print(v, "affine", np.round(Wm, 4).tolist())
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            warped = cv2.remap(S[i], mx, my, cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
            ow = cv2.remap(own[i].astype(np.float32), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0.5
            mags.append(np.linalg.norm(flow, axis=2))
            print(v, "flow texels median", f"{np.median(mags[-1][own[i]]):.2f}", "p95", f"{np.percentile(mags[-1][own[i]], 95):.2f}")
        warped = warped.copy()
        warped[~ow] = np.nan
        out.append(warped)
    keep = [views.index(v) for v in a.views.split(",")] if a.views else list(range(len(views)))
    A = np.stack([out[k] for k in keep])
    with np.errstate(all="ignore"):
        B = np.nanpercentile(A, a.pct, axis=0)
    names, idx, _ = part_map(P, *np.meshgrid(-hx + (np.arange(w) + 0.5) / a.ppm, hy - (np.arange(h) + 0.5) / a.ppm))
    for rv in a.region:
        part, vl = rv.split("=")
        sel = np.isin(idx, [k for k, nm in enumerate(names) if part in nm])
        for vv in reversed(vl.split(",")):
            j = views.index(vv)
            ok = sel & np.isfinite(out[j][..., 0])
            B[ok] = out[j][ok]
        print(rv, "texels", int(sel.sum()))
    miss = ~np.isfinite(B[..., 0])
    B = np.nan_to_num(B, nan=0)
    B = np.clip(B, 0, 255).astype(np.uint8)
    B = cv2.inpaint(B, miss.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    p = ROOT / "assets" / "textures" / f"{a.name}_p{int(a.pct)}.png"
    cv2.imwrite(str(p), B)
    print(p.relative_to(ROOT), "unseen", f"{miss.mean():.4f}")
    sheet = np.hstack([cv2.resize(np.clip(m * 40, 0, 255).astype(np.uint8), (w // 4, h // 4)) for m in mags])
    cv2.imwrite(str(ROOT / "outputs" / "textures" / f"{a.name}_flow.jpg"), sheet)


if __name__ == "__main__":
    main()
