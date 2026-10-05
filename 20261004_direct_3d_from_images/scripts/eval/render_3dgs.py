"""Render the existing Brush 3DGS (the baseline, used as-is) at the COLMAP views, CPU EWA splatting (numba).

Usage: uv run python scripts/eval/render_3dgs.py [--views holdout] [--scale 4]
Output: outputs/baseline_3dgs/s{scale}/<stem>.png (black background).
Model: standard 3DGS layout (xyz, log scales, logit opacity, quaternion wxyz, SH degree 3); the ply is in the
COLMAP frame (checked: median distance of sparse points to the nearest Gaussian center 0.55 mm).
Rasterization follows the reference 3DGS forward pass: 2D covariance via the projection Jacobian plus a 0.3 px^2
low-pass, 3-sigma footprints, front-to-back alpha compositing (alpha capped at 0.99, skip < 1/255, stop at
T < 1e-4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, OUTPUTS, scene_cfg  # noqa: E402
from tools.geom import load_views  # noqa: E402

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)


def load_ply(path: Path):
    raw = path.read_bytes()
    head = raw[: raw.index(b"end_header\n")].decode()
    n = int(next(l for l in head.splitlines() if l.startswith("element vertex")).split()[-1])
    props = [l.split()[-1] for l in head.splitlines() if l.startswith("property")]
    h = raw.index(b"end_header\n") + len(b"end_header\n")
    d = np.frombuffer(raw[h:h + n * len(props) * 4], dtype="<f4").reshape(n, len(props)).astype(np.float64)
    col = {p: i for i, p in enumerate(props)}
    xyz = d[:, [col["x"], col["y"], col["z"]]]
    scale = np.exp(d[:, [col[f"scale_{i}"] for i in range(3)]])
    q = d[:, [col[f"rot_{i}"] for i in range(4)]]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = 1 / (1 + np.exp(-d[:, col["opacity"]]))
    dc = d[:, [col[f"f_dc_{i}"] for i in range(3)]]
    rest = d[:, [col[f"f_rest_{i}"] for i in range(45)]].reshape(n, 3, 15).transpose(0, 2, 1)  # n,15,3
    sh = np.concatenate([dc[:, None, :], rest], 1)  # n,16,3
    return xyz, scale, q, op, sh


def sh_color(sh, dirs):
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    c = SH_C0 * sh[:, 0]
    c = c - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    c = (c + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5] + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
         + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8])
    c = (c + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
         + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
         + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
         + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return np.clip(c + 0.5, 0, None)


def quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


@njit(cache=True)
def raster(order, mu, conic, rad, op, col, W, H):
    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    for k in range(order.shape[0]):
        i = order[k]
        u, v = mu[i, 0], mu[i, 1]
        r = rad[i]
        x0, x1 = max(int(u - r), 0), min(int(u + r) + 1, W)
        y0, y1 = max(int(v - r), 0), min(int(v + r) + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        a, b, c = conic[i, 0], conic[i, 1], conic[i, 2]
        for y in range(y0, y1):
            dy = y - v
            for x in range(x0, x1):
                if T[y, x] < 1e-4:
                    continue
                dx = x - u
                p = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                if p > 0:
                    continue
                al = min(0.99, op[i] * np.exp(p))
                if al < 1.0 / 255:
                    continue
                w = al * T[y, x]
                img[y, x, 0] += w * col[i, 0]
                img[y, x, 1] += w * col[i, 1]
                img[y, x, 2] += w * col[i, 2]
                T[y, x] *= 1 - al
    return img


def render(view, G):
    xyz, scale, q, op, sh, Rg = G
    Xc = xyz @ view.R.T + view.t
    z = Xc[:, 2]
    keep = z > 0.2
    fx, fy, cx, cy = view.K[0, 0], view.K[1, 1], view.K[0, 2], view.K[1, 2]
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    keep &= (u > -view.width * 0.3) & (u < view.width * 1.3) & (v > -view.height * 0.3) & (v < view.height * 1.3)
    idx = np.nonzero(keep)[0]
    Xk, zk = Xc[idx], z[idx]
    M = Rg[idx] * scale[idx][:, None, :]  # R S
    S3 = M @ M.transpose(0, 2, 1)
    Wc = view.R
    Sc = Wc[None] @ S3 @ Wc.T[None]
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = fx / zk
    J[:, 0, 2] = -fx * Xk[:, 0] / zk ** 2
    J[:, 1, 1] = fy / zk
    J[:, 1, 2] = -fy * Xk[:, 1] / zk ** 2
    S2 = J @ Sc @ J.transpose(0, 2, 1)
    S2[:, 0, 0] += 0.3
    S2[:, 1, 1] += 0.3
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    ok = det > 0
    conic = np.stack([S2[:, 1, 1] / det, -S2[:, 0, 1] / det, S2[:, 0, 0] / det], 1)
    mid = 0.5 * (S2[:, 0, 0] + S2[:, 1, 1])
    lam = mid + np.sqrt(np.maximum(0.1, mid * mid - det))
    rad = np.ceil(3 * np.sqrt(lam))
    cam_c = view.center
    dirs = xyz[idx] - cam_c
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    col = sh_color(sh[idx], dirs)
    sel = np.nonzero(ok & (rad < 0.5 * max(view.width, view.height)))[0]
    order = sel[np.argsort(zk[sel])]
    mu = np.stack([u[idx], v[idx]], 1)
    img = raster(order, mu, conic, rad, op[idx], col, view.width, view.height)
    return np.clip(img * 255, 0, 255).astype(np.uint8)[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="holdout")
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    cfg = scene_cfg()
    ply = next(cfg["source_dir"].glob("*.ply"))
    xyz, scale, q, op, sh = load_ply(ply)
    G = (xyz, scale, q, op, sh, quat_to_rot(q))
    V = load_views(a.scale, "colmap")
    split = json.loads((DATA / "split.json").read_text())
    sel = split[a.views] if a.views in split else (list(V) if a.views == "all" else a.views.split(","))
    out = OUTPUTS / "baseline_3dgs" / f"s{a.scale}"
    out.mkdir(parents=True, exist_ok=True)
    for n in sel:
        p = out / f"{n}.png"
        if p.exists():
            continue
        cv2.imwrite(str(p), render(V[n], G))
        print("rendered", n, flush=True)


if __name__ == "__main__":
    main()
