"""Fit a wire's control points to its color in several photos (wires agent helper; runs with uv, not Blender).

  uv run python scripts/model/wires/fit_wire.py WIRE --views IMG_1580,IMG_1513,... [--fix-ends] [--write]
      [--debug outputs/scratch/wires/fit_<wire>.jpg] [--roi 8] [--max-move 4] [--reg 0.02]

Guards against clutter: the color mask is restricted to a corridor (--roi px at 1/4 scale, default 8) around
the current projected path, and each control point may move at most --max-move mm (default 4).
WIRE is a [wire.<name>] key or bundle_<color> (bundle wires: start/end + bundle path + offset, written back as
explicit pts). Cost per view: distance (px, 1/4-scale image, clipped) from projected curve samples to the
nearest pixel of the wire's color mask, inside an ROI around the initial projected curve. Ends stay fixed when
--fix-ends is given (connector pins, solder holes). Regularized toward the initial points (mm^2 * reg).
Occlusion is not modeled: pick views where the wire is visible. Always check the debug sheet.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from tools.geom import load_views  # noqa: E402

TOML = ROOT / "config" / "model" / "wires.toml"
HOLDOUT = set(__import__("json").loads((ROOT / "data" / "split.json").read_text())["holdout"])


def check_views(names) -> None:
    """Holdout views are for final checks only: never measure, fit or texture with them."""
    bad = sorted(set(names) & HOLDOUT)
    if bad:
        raise SystemExit(f"refusing holdout views {bad} (data/split.json)")
# OpenCV HSV ranges (H 0-180): list of (lo, hi) boxes
HSV = {
    "red": [((0, 110, 70), (7, 255, 255)), ((170, 110, 70), (180, 255, 255))],
    "orange": [((8, 150, 130), (18, 255, 255))],
    "yellow": [((19, 110, 120), (34, 255, 255))],
    "green": [((40, 80, 35), (90, 255, 255))],
    "brown": [((0, 50, 25), (22, 210, 120))],
    "blue": [((100, 150, 50), (125, 255, 255))],
    "white": [((0, 0, 170), (180, 50, 255))],
    "cream": [((12, 25, 140), (35, 110, 255))],
    "grey": [((0, 0, 35), (180, 45, 115))],
}


def catmull(P: np.ndarray, n: int = 12) -> np.ndarray:
    P = np.asarray(P, float)
    if len(P) < 3:
        return np.linspace(P[0], P[-1], n * (len(P) - 1) + 1)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1:i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1][None])
    return np.vstack(out)


def wire_points(T: dict, name: str) -> tuple[str, np.ndarray]:
    if name.startswith("bundle_"):
        b = T["bundle"]
        col = name[len("bundle_"):]
        w = b["wires"][col]
        if "pts" in w:
            return col, np.array(w["pts"], float)
        mid = [[a + d for a, d in zip(p, w["offset"])] for p in b["path"]]
        return col, np.array([w["start"], *mid, w["end"]], float)
    w = T["wire"][name]
    return w.get("color", "white"), np.array(w["pts"], float)


def mask_dt(img: np.ndarray, color: str, roi: np.ndarray, clip: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in HSV[color]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dt = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5)
    return np.minimum(dt, clip), m


def bilinear(a: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = a.shape
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    u0, v0 = u.astype(int), v.astype(int)
    du, dv = u - u0, v - v0
    return (a[v0, u0] * (1 - du) * (1 - dv) + a[v0, u0 + 1] * du * (1 - dv) + a[v0 + 1, u0] * (1 - du) * dv
            + a[v0 + 1, u0 + 1] * du * dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wire")
    ap.add_argument("--views", required=True)
    ap.add_argument("--fix-ends", action="store_true")
    ap.add_argument("--roi", type=int, default=8, help="corridor half-width (1/4-scale px) around the current path")
    ap.add_argument("--clip", type=float, default=15.0)
    ap.add_argument("--reg", type=float, default=0.02)
    ap.add_argument("--max-move", type=float, default=4.0, help="bound on each control point move (mm)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    T = tomllib.loads(TOML.read_text())
    color, P0 = wire_points(T, a.wire)
    V = load_views(4)
    check_views(a.views.split(","))
    views = [V[n] for n in a.views.split(",")]
    data = []
    S0 = catmull(P0) / 1e3
    for v in views:
        img = v.image()
        uv, z = v.project(S0)
        roi = np.zeros(img.shape[:2], np.uint8)
        cv2.polylines(roi, [uv.astype(np.int32).reshape(-1, 1, 2)], False, 255, 2 * a.roi)
        dt, m = mask_dt(img, color, roi, a.clip)
        data.append((v, dt, m, img))
    free = np.arange(len(P0))
    if a.fix_ends:
        free = free[1:-1]

    def unpack(x):
        P = P0.copy()
        P[free] = x.reshape(-1, 3)
        return P

    def res(x):
        P = unpack(x)
        S = catmull(P) / 1e3
        r = []
        for v, dt, _, _ in data:
            uv, z = v.project(S)
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < v.width) & (uv[:, 1] < v.height)
            d = np.where(ok, bilinear(dt, uv), 0.0)
            r.append(d / np.sqrt(len(S)))
        r.append(np.sqrt(a.reg) * (P[free] - P0[free]).ravel())
        return np.concatenate(r)

    x0 = P0[free].ravel()
    r0 = res(x0)
    sol = least_squares(res, x0, diff_step=0.3 / np.maximum(np.abs(x0), 1), max_nfev=400,
                        bounds=(x0 - a.max_move, x0 + a.max_move))
    P = unpack(sol.x)
    nS = len(catmull(P))
    per = []
    for v, dt, _, _ in data:
        uv, z = v.project(catmull(P) / 1e3)
        per.append(float(np.median(bilinear(dt, uv))))
    print(f"{a.wire} color {color} cost {np.sum(r0**2):.1f} -> {np.sum(sol.fun**2):.1f} (n samples {nS})")
    print("median px per view:", dict(zip([d[0].name for d in data], np.round(per, 1).tolist())))
    print("moves mm:", np.round(np.linalg.norm(P - P0, axis=1), 1).tolist())
    print("pts =", np.round(P, 1).tolist())
    if a.debug:
        tiles = []
        for v, dt, m, img in data:
            im = img.copy()
            im[m > 0] = (0.5 * im[m > 0] + [0, 127, 127]).astype(np.uint8)
            for PP, c in ((P0, (255, 0, 255)), (P, (255, 255, 255))):
                uv, _ = v.project(catmull(PP) / 1e3)
                cv2.polylines(im, [uv.astype(np.int32).reshape(-1, 1, 2)], False, c, 1)
            cv2.putText(im, v.name, (10, 40), 0, 1.2, (255, 255, 255), 3)
            tiles.append(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(a.debug, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(a.debug)
    if a.write:
        write_pts(a.wire, P)


def thin_pts(P: np.ndarray, min_mm: float = 2.5) -> np.ndarray:
    """Drop interior control points closer than min_mm to the previous kept point (fits and Viterbi can stack
    points, which makes Bezier kinks). End points are always kept."""
    keep = [0]
    for i in range(1, len(P) - 1):
        if np.linalg.norm(P[i] - P[keep[-1]]) >= min_mm:
            keep.append(i)
    if len(P) > 1:
        if len(keep) > 1 and np.linalg.norm(P[-1] - P[keep[-1]]) < min_mm:
            keep.pop()
        keep.append(len(P) - 1)
    return P[keep]


def write_pts(name: str, P: np.ndarray) -> None:
    """Replace (or add) the pts = [...] entry of [wire.<name>] / [bundle.wires.<color>] in the TOML (thinned)."""
    P = thin_pts(np.asarray(P, float))
    txt = TOML.read_text()
    sec = f"[bundle.wires.{name[len('bundle_'):]}]" if name.startswith("bundle_") else f"[wire.{name}]"
    i = txt.index(sec)
    j = txt.find("\n[", i + len(sec))
    j = len(txt) if j < 0 else j
    body = txt[i:j]
    pts = "pts = [" + ", ".join(f"[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]" for p in P) + "]"
    if re.search(r"^pts = \[\[", body, flags=re.M):  # pts are always a single line (trailing comment dropped)
        body = re.sub(r"^pts = \[\[.*$", lambda _m: pts, body, count=1, flags=re.M)
    else:
        lines = body.rstrip("\n").split("\n")
        k = len(lines)
        while k > 1 and (not lines[k - 1].strip() or lines[k - 1].lstrip().startswith("#")):
            k -= 1  # keep trailing comment lines (next section's header) after the inserted pts
        body = "\n".join(lines[:k] + [pts] + lines[k:]) + "\n"
    TOML.write_text(txt[:i] + body + txt[j:])


if __name__ == "__main__":
    main()
