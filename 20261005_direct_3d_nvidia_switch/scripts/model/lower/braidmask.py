"""Braided-cable likelihood mask for a photo (s2, raw pixel order): the braid is dark with a fine speckled weave,
the chassis around it is smooth black and the plates are smooth gray. mask = local high-frequency energy above a
threshold in pixels whose local mean is dark. Used by ray_depth_b.py (Viterbi depth) and its debug sheets."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def braid_mask(stem: str, rel_thr: float = 0.18, dens_thr: float = 0.33, lum_max: float = 110.0, scale: int = 2):
    """Speckle density: fraction of pixels (Gaussian window sigma 5 px) whose high-pass (sigma 1.5) magnitude,
    normalized by the local mean + 12, exceeds rel_thr. A thin edge cannot reach the density threshold; the
    weave can, also in shadow (normalized contrast)."""
    img = cv2.imread(str(ROOT / f"data/undistorted_s{scale}/{stem}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), 6.0)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5)) / (mean + 12.0)
    dens = cv2.GaussianBlur((hp > rel_thr).astype(np.float32), (0, 0), 5.0)
    m = (dens > dens_thr) & (mean < lum_max)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return img, m > 0, dens


def teal_mask(stem: str, scale: int = 2, dens_thr: float = 0.25):
    """Braid A (teal mesh sheath): pixels with G and B above R and G close to B (blue fibers have B >> G),
    combined with the speckle density of braid_mask (a relaxed threshold), then cleaned."""
    img, _, dens = braid_mask(stem, scale=scale)
    b, g, r = [c.astype(np.int16) for c in cv2.split(cv2.GaussianBlur(img, (0, 0), 1.5))]
    teal = (g > r + 8) & (b > r + 8) & (np.abs(g - b) < 0.35 * (b - r) + 12) & (b < 235)
    tf = cv2.GaussianBlur(teal.astype(np.float32), (0, 0), 4.0)
    m = (tf > 0.3) & (dens > dens_thr)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return img, m > 0, tf


def rod_mask(stem: str, scale: int = 2, lum_min: float = 105.0, sat_max: float = 45.0, dens_max: float = 0.22):
    """Silver rod: bright, unsaturated and smooth (low speckle density), unlike the braids and the black floor."""
    img, _, dens = braid_mask(stem, scale=scale)
    f = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.int16)
    lum = f.mean(2)
    sat = f.max(2) - f.min(2)
    m = (lum > lum_min) & (sat < sat_max) & (dens < dens_max)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return img, m > 0, dens


def cost_map(m: np.ndarray, cap: float = 60.0, inside_w: float = 0.5):
    """V-shaped cost: distance to the mask outside it, minus inside_w x depth into the mask inside it."""
    out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    ins = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return np.minimum(out, cap) - inside_w * np.minimum(ins, 40.0)
