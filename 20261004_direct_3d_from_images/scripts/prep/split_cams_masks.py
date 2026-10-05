"""Holdout split, camera export for Blender, and ground-truth object masks.

Outputs (data/):
  split.json                  {"holdout": [...], "train": [...], "probe": [...]} image stems
                              probe = small diverse train subset used for fast iteration renders
  cameras_s{N}.json           per view: K (index-pixel convention), R, t (world-to-camera, meters), size
  masks_s{N}/<stem>.png       255 = object (CRT unit, its wires and boards), 0 = mat (known background),
                              128 = unknown (ray leaves the mat; silhouette metrics ignore these pixels)

Mask rule: a pixel is in the known region if its viewing ray meets the mat plane (z = 0) inside the mat
rectangle (shrunk by a margin): nothing but the object can stand between the camera and the mat there.
Inside it, object = not mat-colored. Mat color: blue hue and either strong Lab blue chroma (b < 108, L > 40) or HSV S >= 90 and V >= 58
(shadowed, defocused mat);
the glossy black case reflects the mat as dark, weakly blue pixels, which this keeps as object.
Mirror check (once assets/textures/env_mat.png exists): a mat-blue pixel whose ray hits the mat inside the case
footprint and that is much darker (Lab L < 0.6x) than the median mat is a reflection in the glossy case walls.
Sparse-point evidence: pixels where a point seen in that view (reprojection error < 2 px, z > 5 mm) projects are
object even if mat-colored (light-blue socket cap).
Mat rectangle (world, meters) comes from blue sparse points (min-area rect), stored in config/world_extra.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, ROOT, scene_cfg  # noqa: E402
from tools.geom import load_points, load_views  # noqa: E402


def farthest_views(views, k: int, seed: int, exclude=()) -> list[str]:
    names = [n for n in views if n not in exclude]
    D = np.array([-views[n].R[2] for n in names])  # direction from object toward camera (approx)
    C = np.array([views[n].center for n in names])
    F = np.hstack([D, C / np.linalg.norm(C, axis=1, keepdims=True).max()])
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(names)))]
    dist = np.linalg.norm(F - F[sel[0]], axis=1)
    while len(sel) < k:
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, np.linalg.norm(F - F[j], axis=1))
    return sorted(names[i] for i in sel)


def mat_rect() -> dict:
    X, rgb, _, _ = load_points()
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    blue = (hsv[:, 0] > 95) & (hsv[:, 0] < 125) & (hsv[:, 1] > 120) & (hsv[:, 2] > 80) & (np.abs(X[:, 2]) < 0.004)
    (cx, cy), (w, h), ang = cv2.minAreaRect(X[blue][:, :2].astype(np.float32))
    return {"center": [float(cx), float(cy)], "size": [float(w), float(h)], "angle_deg": float(ang),
            "n_points": int(blue.sum()), "printed_size_mm": "450 x 300 (molded text on the mat)"}


def mat_inside(P: np.ndarray, rect: dict, margin: float) -> np.ndarray:
    c = np.array(rect["center"])
    a = np.radians(rect["angle_deg"])
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    d = P - c
    return (np.abs(d @ u) < rect["size"][0] / 2 - margin) & (np.abs(d @ v) < rect["size"][1] / 2 - margin)


def point_evidence(view, X: np.ndarray, vis_idx: np.ndarray, zmin: float = 0.005, r: int = 3) -> np.ndarray:
    """Pixels where sparse points seen in this view and at least zmin above the mat project (dilated by r px):
    they show the object regardless of color (e.g. the light-blue CRT socket cap, which is mat-colored)."""
    ev = np.zeros((view.height, view.width), np.uint8)
    P = X[vis_idx]
    P = P[P[:, 2] > zmin]
    if len(P):
        uv, z = view.project(P)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < view.width - 0.5) & (uv[:, 1] < view.height - 0.5)
        uv = np.round(uv[ok]).astype(int)
        ev[uv[:, 1], uv[:, 0]] = 1
        ev = cv2.dilate(ev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return ev > 0


FOOTPRINT_MM = (104.0, 52.0)  # case half-extents in x, y (case agent, 2026-10-05): mat under it is never seen


def load_mat_texture():
    """Mat texture (from scripts/tools/texture_from_photos.py, config/model/env_mat_texture_spec.json) and its
    corners; None if not built yet."""
    tex_p = ROOT / "assets" / "textures" / "env_mat.png"
    spec_p = CONFIG / "model" / "env_mat_texture_spec.json"
    if not tex_p.exists() or not spec_p.exists():
        return None
    tex = cv2.GaussianBlur(cv2.imread(str(tex_p)), (0, 0), 2.0)
    C = np.array(json.loads(spec_p.read_text())["surface"]["corners_mm"], float)[:, :2] / 1e3
    lab = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB)
    return tex, lab, C


def expected_mat_L(P: np.ndarray, mt) -> np.ndarray:
    """Expected Lab lightness of the mat at plane points P (n, 2), meters; median mat lightness under the case."""
    tex, lab, C = mt
    eu, ev = C[1] - C[0], C[3] - C[0]
    u = ((P - C[0]) @ eu) / (eu @ eu)
    v = ((P - C[0]) @ ev) / (ev @ ev)
    h, w = lab.shape[:2]
    uu = np.clip((u * w).astype(int), 0, w - 1)
    vv = np.clip((v * h).astype(int), 0, h - 1)
    L = lab[vv, uu, 0].astype(float)
    under = (np.abs(P[:, 0]) < FOOTPRINT_MM[0] / 1e3 + 0.003) & (np.abs(P[:, 1]) < FOOTPRINT_MM[1] / 1e3 + 0.003)
    L[under] = np.median(lab[..., 0])
    return L


def object_mask(view, rect: dict, margin: float = 0.02, evidence: np.ndarray | None = None,
                mat_tex=None) -> np.ndarray:
    img = view.image()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(int)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    hue_blue = (hsv[..., 0] >= 90) & (hsv[..., 0] <= 122)
    blue = hue_blue & (((lab[..., 2] < 108) & (lab[..., 0] > 40)) | ((hsv[..., 1] >= 90) & (hsv[..., 2] >= 58)))
    # rays to the plane
    h, w = img.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Kinv = np.linalg.inv(view.K)
    d = (view.R.T @ (Kinv @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))).T
    c = view.center
    s = -c[2] / d[:, 2]
    P = c[None, :2] + s[:, None] * d[:, :2]
    inside = (mat_inside(P, rect, margin) & (s > 0)).reshape(h, w)
    if mat_tex is not None:
        # Glossy black plastic mirrors the mat: blue but much darker than the mat really is at the ray's hit point.
        # Only rays that hit the mat inside the case footprint can be reflections in the case's outer walls
        # (a ray through an outer wall point continues down into the footprint); elsewhere dark mat is shading.
        Lexp = expected_mat_L(P, mat_tex).reshape(h, w)
        under = ((np.abs(P[:, 0]) < FOOTPRINT_MM[0] / 1e3 + 0.003)
                 & (np.abs(P[:, 1]) < FOOTPRINT_MM[1] / 1e3 + 0.003)).reshape(h, w)
        blue &= ~((lab[..., 0] < 0.6 * Lexp) & under)
    m = ((~blue) | (evidence if evidence is not None else False)) & inside
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # drop small specks not connected to large components
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.0004 * h * w
    out = (keep[lab] * 255).astype(np.uint8)
    known = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out[~known] = 128
    return out


def main() -> None:
    cfg = scene_cfg()
    V4 = load_views(4)
    n_hold = max(1, round(cfg["holdout_fraction"] * len(V4)))
    hold = farthest_views(V4, n_hold, cfg["holdout_seed"])
    train = [n for n in V4 if n not in hold]
    probe = farthest_views(V4, 12, cfg["holdout_seed"] + 1, exclude=hold)
    DATA.mkdir(exist_ok=True)
    (DATA / "split.json").write_text(json.dumps({"holdout": hold, "train": train, "probe": probe}, indent=1))
    print(f"split: {len(hold)} holdout, {len(train)} train, probe {probe}")

    rect = mat_rect()
    (CONFIG / "world_extra.yaml").write_text("# Generated by scripts/prep/split_cams_masks.py\n"
                                             + yaml.safe_dump({"mat_rect_world_m": rect}, sort_keys=False))
    print("mat rect", rect)

    for s in cfg["work_scales"]:
        V = load_views(s)
        cams = {n: {"K": v.K.tolist(), "R": v.R.tolist(), "t": v.t.tolist(), "width": v.width, "height": v.height,
                    "camera_id": v.camera_id, "image": str(v.path.relative_to(DATA.parent))} for n, v in V.items()}
        (DATA / f"cameras_s{s}.json").write_text(json.dumps({"scale": s, "views": cams}))
    out = DATA / "masks_s4"
    out.mkdir(exist_ok=True)
    X, _, err, tracks = load_points()
    seen: dict[str, list[int]] = {}
    for i, tr in tracks.items():
        if err[i] < 2.0:
            for stem, _ in tr:
                seen.setdefault(stem, []).append(i)
    mt = load_mat_texture()
    for n, v in V4.items():
        p = out / f"{n}.png"
        if not p.exists() or "--force" in sys.argv:
            ev = point_evidence(v, X, np.array(seen.get(n, []), int))
            cv2.imwrite(str(p), object_mask(v, rect, evidence=ev, mat_tex=mt))
    print("masks written:", len(list(out.glob("*.png"))))


if __name__ == "__main__":
    main()
