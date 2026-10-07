"""Photometric comparison of the model (and a 3DGS baseline if one exists) against the photos on the same views.

Usage: uv run python scripts/eval/compare_photometric.py outputs/runs/<run> [--out outputs/compare/<name>]
Region: photo object mask (255) inside the known region, eroded 2 px (mask noise), at 1/4 scale.
Metrics per view, for model and 3DGS: PSNR and SSIM (gray, Gaussian window sigma 2) raw, and after a per-view
affine color fit (3x4, least squares) of the render to the photo, estimated over the known region (object +
environment) only when that region is more than twice the object region (in practice it is not here, so the fit
uses the scored object region itself; audit DevLog-005) and scored inside the object region (our renders use a studio
light rig, not the room's lighting; the fit removes global exposure/white-balance/lighting-level differences
for both methods alike). psnr_fit_blur4: the same after a sigma 4 px Gaussian blur of both images (alignment-
tolerant: colors and structure without the sub-pixel edge penalty). Floor reference "flat": the photo's own mean
color over the region (what a single flat color scores). This capture has no 3DGS baseline; renders in
outputs/baseline_3dgs/s4/ are used only if present. Writes compare.json and compare.png (photo | model | flat).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, OUTPUTS  # noqa: E402
from eval.evaluate import ssim_map  # noqa: E402


def affine_fit(src, dst, m):
    A = np.hstack([src[m].astype(np.float64), np.ones((m.sum(), 1))])
    M = np.linalg.lstsq(A, dst[m].astype(np.float64), rcond=None)[0]
    H, W = src.shape[:2]
    return np.clip(np.hstack([src.reshape(-1, 3), np.ones((H * W, 1))]) @ M, 0, 255).reshape(H, W, 3)


def metrics(img, photo, m):
    d = (img.astype(np.float64) - photo.astype(np.float64))[m]
    psnr = 10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-9))
    g1 = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    return float(psnr), float(ssim_map(g1, g2)[m].mean())


def psnr_blur(img, photo, m, sigma=4.0):
    """PSNR after a Gaussian blur of both images (sigma px at 1/4 scale): tolerant to the 1-3 px edge
    misalignment an explicit model always has, so it measures colors/structure rather than sub-pixel fit."""
    a = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(photo.astype(np.float32), (0, 0), sigma)
    d = (a - b)[m]
    return float(10 * np.log10(255 ** 2 / max(np.mean(d.astype(np.float64) ** 2), 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "render_meta.json").read_text())
    out = Path(a.out) if a.out else OUTPUTS / "compare" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for n in meta["views"]:
        gs_p = OUTPUTS / "baseline_3dgs" / "s4" / f"{n}.png"
        rgb_p = run / "rgb" / f"{n}.png"
        if not rgb_p.exists():
            continue
        photo = cv2.imread(str(DATA / "undistorted_s4" / f"{n}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        pm = cv2.imread(str(DATA / "masks_s4" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.erode((pm == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        rgba = cv2.imread(str(rgb_p), cv2.IMREAD_UNCHANGED)
        a_ = rgba[..., 3:4] / 255.0
        model = (rgba[..., :3] * a_).astype(np.float64)  # black where the model is empty
        r = {"view": n, "px": int(m.sum())}
        if m.sum() < 100:
            continue
        flat = np.zeros_like(photo, np.float64)
        flat[:] = photo[m].mean(0)
        imgs = [("model", model), ("flat", flat)]
        if gs_p.exists():
            imgs.append(("3dgs", cv2.imread(str(gs_p)).astype(np.float64)))
        known = (pm != 128) & (rgba[..., 3] > 0)  # color fit over object + environment where both render
        for key, img in imgs:
            r[f"{key}_psnr"], r[f"{key}_ssim"] = metrics(img, photo, m)
            fit = affine_fit(img, photo, known if known.sum() > 2 * m.sum() else m)
            r[f"{key}_psnr_fit"], r[f"{key}_ssim_fit"] = metrics(fit, photo, m)
            r[f"{key}_psnr_fit_blur4"] = psnr_blur(fit, photo, m)
            if key == "model":
                model_fit = fit
        rows.append(r)
        t = [photo, model_fit.astype(np.uint8), flat.astype(np.uint8)]
        t = [cv2.resize(x, (476, 357)) for x in t]
        cv2.putText(t[1], f"model fit {r['model_psnr_fit']:.1f} dB", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        cv2.putText(t[2], f"flat {r['flat_psnr']:.1f} dB", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(t[0], n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(np.hstack(t))
    keys = sorted({k for r in rows for k in r if k not in ("view", "px")})
    summary = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    (out / "compare.json").write_text(json.dumps({"run": run.name, "n_views": len(rows), "mean": summary,
                                                  "views": rows}, indent=1))
    if tiles:
        cv2.imwrite(str(out / "compare.png"), np.vstack(tiles[:8]))
    print(json.dumps({"n_views": len(rows), **{k: round(v, 3) for k, v in summary.items()}}, indent=1))


if __name__ == "__main__":
    main()
