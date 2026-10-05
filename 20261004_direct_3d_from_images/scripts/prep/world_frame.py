"""Metric world frame for the CRT capture: COLMAP -> world similarity (s, R, t), written to config/world.yaml.

1. Mat plane: RANSAC + SVD refit on blue sparse points (the silicone mat), normal oriented toward the cameras.
2. Scale: the mat's molded mm ruler. A strip along the ruler (seeded by two picks in one view) is rectified
   onto the mat plane in every view where it is visible at >= 4 px/mm; the FFT peak of the tick profile gives
   the 1 mm period in COLMAP units (median over views, small rotation search per view).
3. Axes and origin from case top-rim corner picks (config/picks.yaml), each triangulated by epipolar NCC
   matching in neighboring views: X along the far long edge, Z = mat normal, -Y toward the near long side,
   origin at the rim rectangle center projected onto the mat plane (z = 0 on the mat top surface).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, load_yaml  # noqa: E402
from tools.geom import load_points, load_views, measure_point, neighbors  # noqa: E402


def fit_mat_plane(X: np.ndarray, rgb: np.ndarray, up_hint: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    blue = (hsv[:, 0] > 95) & (hsv[:, 0] < 125) & (hsv[:, 1] > 120) & (hsv[:, 2] > 80)
    B = X[blue]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4000):
        s = B[rng.choice(len(B), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up_hint) < 0.8:
            continue
        inl = np.abs(B @ n - n @ s[0]) < 0.01
        if best is None or inl.sum() > best.sum():
            best = inl
    P = B[best]
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    if n @ up_hint < 0:
        n = -n
    print(f"mat plane: {best.sum()} / {len(B)} blue points inside 0.01 u")
    return n, float(-n @ c)


def ruler_period(views, n, d, A, ax0) -> tuple[float, float, int]:
    def strip(v, ax, res=0.0005, L=1.2, b0=0.02, b1=0.06):
        ay = np.cross(n, ax)
        a = np.arange(int(L / res)) * res
        b = np.arange(int((b1 - b0) / res)) * res + b0
        AA, BB = np.meshgrid(a, b)
        X = A[None, None] + AA[..., None] * ax + BB[..., None] * ay
        uv, z = v.project(X.reshape(-1, 3))
        m = uv.reshape(*AA.shape, 2).astype(np.float32)
        ok = (m[..., 0] > 2) & (m[..., 0] < v.width - 3) & (m[..., 1] > 2) & (m[..., 1] < v.height - 3)
        ok &= z.reshape(AA.shape) > 0
        img = cv2.remap(v.image(gray=True), m[..., 0], m[..., 1], cv2.INTER_LINEAR).astype(float)
        return img, ok.all(0)

    def period(prof, res):
        prof = prof - np.convolve(prof, np.ones(41) / 41, "same")
        prof = prof[30:-30] * np.hanning(len(prof) - 60)
        F = np.abs(np.fft.rfft(prof, 1 << 19))
        f = np.fft.rfftfreq(1 << 19, res)
        m = (f > 1 / 0.03) & (f < 1 / 0.012)
        k = np.argmax(F * m)
        return 1 / f[k], F[k] / np.median(F[m])

    def rot(a, deg):
        t = np.radians(deg)
        return a * np.cos(t) + np.cross(n, a) * np.sin(t)

    per = []
    for v in views.values():
        best = None
        for deg in np.arange(-2, 2.01, 0.25):
            ax = rot(ax0, deg)
            img, ok = strip(v, ax)
            if ok.sum() < 1200:
                break
            cols = np.where(ok)[0]
            uv, _ = v.project(np.stack([A + ax * 0.5, A + ax * 0.5 + ax * 0.019]))
            mm_px = np.linalg.norm(uv[1] - uv[0])
            p, snr = period(img[:, cols.min():cols.max()].mean(0), 0.0005)
            if best is None or snr > best[1]:
                best = (p, snr, mm_px)
        if best and best[1] > 15 and best[2] > 4:
            per.append(best[0])
    per = np.array(per)
    return float(np.median(per)), float(per.std()), len(per)


def main() -> None:
    picks = load_yaml("picks.yaml")
    V = load_views(2, "colmap")
    X, rgb, _, _ = load_points("colmap")
    up_hint = -np.mean([v.R[2] for v in V.values()], 0)
    n, d = fit_mat_plane(X, rgb, up_hint / np.linalg.norm(up_hint))

    def on_plane(view, uv):
        r = view.ray(np.array(uv, float))
        c = view.center
        return c - (n @ c + d) / (n @ r) * r

    rp = picks["ruler_seed"]
    rv = V[rp["view"]]
    A, B = on_plane(rv, rp["uv_a"]), on_plane(rv, rp["uv_b"])
    ax0 = B - A
    ax0 -= (ax0 @ n) * n
    ax0 /= np.linalg.norm(ax0)
    p_mm, p_std, n_views = ruler_period(V, n, d, A, ax0)
    s = 1e-3 / p_mm  # meters per COLMAP unit
    print(f"ruler: 1 mm = {p_mm:.6f} u (std {p_std:.6f}, {n_views} views) -> {s * 1e3:.3f} mm/u")

    cp = picks["case_rim_corners"]
    cv_ = V[cp["view"]]
    nb = neighbors(cv_, V, 25, 12)
    C = {}
    for name, uv in cp["uv"].items():
        r = measure_point(cv_, uv, nb, (2.0, 8.0))
        if r["n_inliers"] < 3 and name != "near_left":
            raise RuntimeError(f"corner {name} not triangulated robustly: {r}")
        C[name] = r["X"]
        print(f"corner {name}: inliers {r['n_inliers']} {r['views']} err {r['err_px']} "
              f"height {(r['X'] @ n + d) * s * 1e3:.1f} mm")
    x = C["far_right"] - C["far_left"]
    x -= (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    if (C["near_left"] - C["far_left"]) @ y > 0:  # near side must be -Y
        x, y = -x, -y
    R = np.stack([x, y, n])
    O = 0.5 * (C["near_left"] + C["far_right"])
    O = O - (n @ O + d) * n
    t = -s * R @ O
    width = np.linalg.norm(C["near_left"] - C["far_left"]) * s * 1e3
    length = np.linalg.norm(C["far_right"] - C["far_left"]) * s * 1e3
    cosang = (C["near_left"] - C["far_left"]) @ (C["far_right"] - C["far_left"])
    cosang /= np.linalg.norm(C["near_left"] - C["far_left"]) * np.linalg.norm(C["far_right"] - C["far_left"])
    out = {
        "generated_by": "scripts/prep/world_frame.py",
        "date": str(date.today()),
        "scale_m_per_unit": float(s),
        "R": R.tolist(),
        "t": t.tolist(),
        "mat_plane_colmap": {"n": n.tolist(), "d": d},
        "ruler": {"mm_period_units": p_mm, "std_units": p_std, "n_views": n_views},
        "case_rim_check_mm": {"width_short_edge": float(width), "length_far_edge": float(length),
                              "corner_angle_deg": float(np.degrees(np.arccos(cosang))),
                              "stated_width_mm": 120.0},
    }
    (CONFIG / "world.yaml").write_text("# Generated; do not edit by hand.\n" + yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out["case_rim_check_mm"]))


if __name__ == "__main__":
    main()
