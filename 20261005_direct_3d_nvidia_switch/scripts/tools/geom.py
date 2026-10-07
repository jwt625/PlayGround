"""Camera views, projection, triangulation and epipolar matching on the undistorted images.

Pixel convention for every public function here: (u, v) are pixel-index coordinates at the given scale
(the center of the top-left pixel is (0, 0)). COLMAP continuous coordinates are index + 0.5.

World frame: if config/world.yaml exists, views are expressed in the metric world frame (meters, Z up,
origin at the case footprint center); pass frame="colmap" to get raw COLMAP coordinates.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_yaml, scene_cfg  # noqa: E402
from prep.colmap_io import read_model  # noqa: E402


@dataclass
class View:
    name: str  # image stem, e.g. IMG_1518
    camera_id: int
    K: np.ndarray  # 3x3, index-pixel convention at this scale (cx, cy already shifted by -0.5)
    R: np.ndarray  # world-to-camera rotation
    t: np.ndarray  # world-to-camera translation
    width: int
    height: int
    scale: int
    path: Path

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    def image(self, gray: bool = False) -> np.ndarray:
        return _read(str(self.path), gray)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (n,3) world points; returns (n,2) pixel-index coords and (n,) camera depths."""
        Xc = (self.R @ np.atleast_2d(X).T).T + self.t
        uv = (self.K @ Xc.T).T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    def ray(self, uv: np.ndarray) -> np.ndarray:
        """Unit ray direction in world frame through pixel-index coords uv (2,)."""
        d = np.linalg.solve(self.K, np.array([uv[0], uv[1], 1.0]))
        d = self.R.T @ d
        return d / np.linalg.norm(d)


@lru_cache(maxsize=64)
def _read(path: str, gray: bool) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(path)
    return img


def world_transform() -> tuple[float, np.ndarray, np.ndarray] | None:
    """(s, R, t) with X_world = s * R @ X_colmap + t, or None if not yet defined."""
    if not (CONFIG / "world.yaml").exists():
        return None
    w = load_yaml("world.yaml")
    return float(w["scale_m_per_unit"]), np.array(w["R"], float), np.array(w["t"], float)


@lru_cache(maxsize=8)
def load_views(scale: int = 4, frame: str = "world") -> dict[str, View]:
    cfg = scene_cfg()
    cams, imgs, _ = read_model(cfg["sparse_dir"])
    intr = json.loads((DATA / f"undistorted_s{scale}" / "intrinsics.json").read_text())["cameras"]
    wt = world_transform() if frame == "world" else None
    sp = DATA / "split.json"
    excluded = set(json.loads(sp.read_text()).get("excluded", [])) if sp.exists() else set()  # bad poses
    out = {}
    for im in imgs.values():
        if Path(im.name).stem in excluded:
            continue
        c = intr[str(im.camera_id)]
        K = np.array([[c["fx"], 0, c["cx"] - 0.5], [0, c["fy"], c["cy"] - 0.5], [0, 0, 1.0]])
        R, t = im.R, im.tvec
        if wt is not None:
            s, Rw, tw = wt
            # X_c = R X_col + t, X_col = Rw^T (X_w - tw) / s  ->  X_c = (R Rw^T / s) X_w + (t - R Rw^T tw / s)
            # Scale the camera frame by s to keep a proper rotation: X_c' = s X_c.
            R_new = R @ Rw.T
            t_new = s * t - R_new @ tw
            R, t = R_new, t_new
        stem = Path(im.name).stem
        out[stem] = View(stem, im.camera_id, K, R, t, c["width"], c["height"], scale,
                         DATA / f"undistorted_s{scale}" / f"{stem}.jpg")
    return dict(sorted(out.items()))


@lru_cache(maxsize=2)
def load_points(frame: str = "world") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sparse points: (xyz (n,3), rgb (n,3), reproj error (n,), track dict id->[(image stem, idx)])."""
    cfg = scene_cfg()
    _, imgs, pts = read_model(cfg["sparse_dir"])
    X = pts.xyz
    wt = world_transform() if frame == "world" else None
    if wt is not None:
        s, Rw, tw = wt
        X = s * (Rw @ X.T).T + tw
    stems = {im.id: Path(im.name).stem for im in imgs.values()}
    tracks = {i: [(stems[int(a)], int(b)) for a, b in pts.tracks[pid]] for i, pid in enumerate(pts.ids)}
    return X, pts.rgb, pts.error, tracks


def triangulate(obs: list[tuple[View, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Linear DLT then Gauss-Newton refinement. obs: [(view, uv)]. Returns X (3,) and per-view reproj err (px)."""
    A = []
    for v, uv in obs:
        P = v.P
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1, :3] / vt[-1, 3]
    for _ in range(10):
        J, r = [], []
        for v, uv in obs:
            p, _ = v.project(X)
            r.extend(p[0] - uv)
            eps = 1e-6 * max(1.0, np.linalg.norm(X))
            cols = []
            for k in range(3):
                dX = np.zeros(3)
                dX[k] = eps
                cols.append((v.project(X + dX)[0][0] - p[0]) / eps)
            J.extend(np.array(cols).T)
        J, r = np.array(J), np.array(r)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + step
        if np.linalg.norm(step) < 1e-9:
            break
    err = np.array([np.linalg.norm(v.project(X)[0][0] - uv) for v, uv in obs])
    return X, err


def triangulate_robust(obs: list[tuple[View, np.ndarray]], thresh: float = 3.0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC over view pairs (the first observation is the anchor pick and is always kept), then refine on
    inliers. Returns X, per-observation reproj error, inlier mask."""
    best = None
    for j in range(1, len(obs)):
        X, _ = triangulate([obs[0], obs[j]])
        err = np.array([np.linalg.norm(v.project(X)[0][0] - uv) if v.project(X)[1][0] > 0 else 1e9
                        for v, uv in obs])
        inl = err < thresh
        if best is None or inl.sum() > best[1].sum() or (inl.sum() == best[1].sum() and err[inl].sum() < best[2]):
            best = (X, inl, err[inl].sum())
    if best is None:
        raise ValueError("need at least two observations")
    inl = best[1]
    X, _ = triangulate([o for o, k in zip(obs, inl) if k])
    err = np.array([np.linalg.norm(v.project(X)[0][0] - uv) for v, uv in obs])
    return X, err, err < thresh


def neighbors(anchor: View, views: dict[str, View], max_angle_deg: float = 25.0, k: int = 12) -> list[View]:
    """Views whose optical axes are within max_angle_deg of the anchor's (closest first, at most k)."""
    za = anchor.R[2]
    cand = []
    for v in views.values():
        if v.name == anchor.name:
            continue
        ang = np.degrees(np.arccos(np.clip(za @ v.R[2], -1, 1)))
        if ang <= max_angle_deg:
            cand.append((ang, v))
    return [v for _, v in sorted(cand, key=lambda t: t[0])[:k]]


def measure_point(anchor: View, uv: np.ndarray, others: list[View], depth_range: tuple[float, float],
                  min_ncc: float = 0.85, r: int = 15, thresh: float = 3.0) -> dict:
    """One pick in one view -> epipolar NCC matches in the other views -> robust triangulation.
    Pass nearby views (see neighbors()): wide baselines make patch NCC unreliable, especially on contours."""
    obs = [(anchor, np.asarray(uv, float))]
    for vb in others:
        m, s = epipolar_match(anchor, obs[0][1], vb, depth_range, r=r)
        if m is not None and s >= min_ncc:
            obs.append((vb, m))
    if len(obs) < 2:
        return {"ok": False, "n_obs": len(obs)}
    X, err, inl = triangulate_robust(obs, thresh)
    return {"ok": int(inl.sum()) >= 3, "X": X, "n_obs": len(obs), "n_inliers": int(inl.sum()),
            "views": [o[0].name for o, k in zip(obs, inl) if k], "err_px": err[inl].round(2).tolist()}


def _patch(img: np.ndarray, uv: np.ndarray, r: int) -> np.ndarray | None:
    u, v = int(round(uv[0])), int(round(uv[1]))
    if u - r < 0 or v - r < 0 or u + r >= img.shape[1] or v + r >= img.shape[0]:
        return None
    return img[v - r:v + r + 1, u - r:u + r + 1].astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else -1.0


def _affine_patch(img: np.ndarray, center: np.ndarray, A: np.ndarray, r: int) -> np.ndarray | None:
    """Sample a (2r+1)^2 patch of img around center, with offsets (du, dv) in the reference view mapped by the
    2x2 matrix A (local affine from the reference view to this view)."""
    M = np.hstack([A, (center - A @ np.array([r, r], float))[:, None]]).astype(np.float32)
    if not (r < center[0] < img.shape[1] - r - 1 and r < center[1] < img.shape[0] - r - 1):
        return None
    return cv2.warpAffine(img, M, (2 * r + 1, 2 * r + 1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def epipolar_match(va: View, uva: np.ndarray, vb: View, depth_range: tuple[float, float],
                   r: int = 12, n: int = 400) -> tuple[np.ndarray | None, float]:
    """Find the pixel in vb matching uva in va by NCC along the epipolar segment for depths in depth_range
    (distance along the ray from va's center, world units). The vb patch is warped by the local affine induced by
    a plane fronto-parallel to va at each depth, so rotated/zoomed neighbor views still match.
    Returns (uv_b or None, best NCC)."""
    ia, ib = va.image(gray=True), vb.image(gray=True)
    pa = _patch(ia, uva, r)
    if pa is None:
        return None, -1.0
    ray = va.ray(uva)
    ex, ey = va.R[0], va.R[1]  # va camera axes in world
    f = va.K[0, 0]
    best, best_uv, best_A = -1.0, None, None
    for dep in np.linspace(*depth_range, n):
        X = va.center + dep * ray
        uvb, z = vb.project(X)
        if z[0] <= 0:
            continue
        zc = (va.R @ X + va.t)[2]
        px = zc / f  # world size of one va pixel at this depth
        P3 = np.stack([X, X + ex * px, X + ey * px])
        q, _ = vb.project(P3)
        A = np.stack([q[1] - q[0], q[2] - q[0]], 1)  # columns: vb displacement per va pixel in u, v
        pb = _affine_patch(ib, uvb[0], A, r)
        if pb is None:
            continue
        s = _ncc(pa, pb)
        if s > best:
            best, best_uv, best_A = s, uvb[0], A
    if best_uv is not None:  # subpixel refine with a small 2D search around the best epipolar hit
        u0, v0 = best_uv
        for du in np.linspace(-1.5, 1.5, 7):
            for dv in np.linspace(-1.5, 1.5, 7):
                c = np.array([u0 + du, v0 + dv])
                pb = _affine_patch(ib, c, best_A, r)
                if pb is not None and (s := _ncc(pa, pb)) > best:
                    best, best_uv = s, c
    return best_uv, best


def grid_crop(view: View, center: tuple[float, float], half: int, out: Path, step: int | None = None,
              marks: list[tuple[float, float]] | None = None, upscale: int = 2) -> Path:
    """Write a crop around center with labeled pixel grid lines (coords in this view's scale) for picking."""
    img = view.image()
    cu, cv_ = int(center[0]), int(center[1])
    x0, y0 = max(cu - half, 0), max(cv_ - half, 0)
    x1, y1 = min(cu + half, view.width), min(cv_ + half, view.height)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    step = step or max(10, (half // 5) // 10 * 10)
    for g in range((x0 // step + 1) * step, x1, step):
        X = (g - x0) * upscale
        cv2.line(crop, (X, 0), (X, crop.shape[0]), (0, 255, 255), 1)
        cv2.putText(crop, str(g), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for g in range((y0 // step + 1) * step, y1, step):
        Y = (g - y0) * upscale
        cv2.line(crop, (0, Y), (crop.shape[1], Y), (0, 255, 255), 1)
        cv2.putText(crop, str(g), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for m in marks or []:
        cv2.drawMarker(crop, (int((m[0] - x0) * upscale), int((m[1] - y0) * upscale)), (0, 0, 255),
                       cv2.MARKER_CROSS, 16, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    return out


def refuse_holdout(stems, allow: bool = False) -> None:
    """Measuring/viewing tools: holdout views are for evaluation only (audits 2026-10-05/06)."""
    if allow:
        return
    sp = DATA / "split.json"
    hold = set(json.loads(sp.read_text())["holdout"]) if sp.exists() else set()
    bad = [n for n in stems if n in hold]
    if bad:
        raise SystemExit(f"{', '.join(bad)}: holdout view(s); measuring or inspecting them leaks into evaluation "
                         "(use train views, or --allow-holdout for evaluation checks only)")


def train_point_mask() -> np.ndarray:
    """Sparse points usable for measuring: at least 2 observations in train views (data/points_train_mask.npy,
    from scripts/prep/export_points.py). The full cloud (holdout-assisted triangulations included) is for
    evaluation only (audit DevLog-003: 9.8 percent of tray points exist only because of holdout views)."""
    return np.load(DATA / "points_train_mask.npy")
