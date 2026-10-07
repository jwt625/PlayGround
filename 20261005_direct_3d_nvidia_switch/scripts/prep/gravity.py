"""Gravity ("scene") frame from the cameras' hand-held orientation, written into config/frames.json.

Method: people hold phones with little roll, so the true up direction is nearly perpendicular to every camera's
display-right vector. Up = eigenvector of sum(r r^T) with the smallest eigenvalue, signed to agree with the mean
display-up vector. Display-right/up per EXIF orientation (raw pixel order, x right, y down): 1 -> (+x, -y),
6 -> (-y, -x), 3 -> (-x, +y), 8 -> (+y, +x). Excluded views (bad poses) are skipped. A bootstrap over views gives
the uncertainty (median and 95th percentile angle to the full estimate); the roll assumption itself can bias the
result by a few degrees if photographers aligned frames with the tilted tray (audit DevLog-003).
Scene frame: Z = up, Y = horizontal component of the tray +Y axis, X = Y x Z; origin = tray frame origin.
Usage: uv run python scripts/prep/gravity.py [--dry-run]
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, scene_cfg  # noqa: E402
from tools.geom import load_views  # noqa: E402

RIGHT = {1: (1, 0, 0), 6: (0, -1, 0), 3: (-1, 0, 0), 8: (0, 1, 0)}
UP = {1: (0, -1, 0), 6: (-1, 0, 0), 3: (0, 1, 0), 8: (1, 0, 0)}


def estimate(rs: np.ndarray, us: np.ndarray) -> np.ndarray:
    w, e = np.linalg.eigh(rs.T @ rs)
    up = e[:, 0]
    return up if up @ us.mean(0) > 0 else -up


def main() -> None:
    src = scene_cfg()["source_dir"] / "images"
    V = load_views(4)
    rs, us = [], []
    for n, v in V.items():
        o = Image.open(next(src.glob(f"{n}.*"))).getexif().get(274, 1)
        rs.append(v.R.T @ np.array(RIGHT[o], float))
        us.append(v.R.T @ np.array(UP[o], float))
    rs, us = np.array(rs), np.array(us)
    up = estimate(rs, us)
    roll = np.degrees(np.arcsin(np.clip(rs @ up, -1, 1)))
    rng = np.random.default_rng(0)
    ang = []
    for _ in range(500):
        k = rng.integers(0, len(rs), len(rs))
        ang.append(np.degrees(np.arccos(np.clip(estimate(rs[k], us[k]) @ up, -1, 1))))
    ys = np.array([0.0, 1.0, 0.0]) - up[1] * up
    ys /= np.linalg.norm(ys)
    xs = np.cross(ys, up)
    R = np.stack([xs, ys, up], 1)
    note = (f"gravity frame (scripts/prep/gravity.py, {len(rs)} views): Z up = smallest eigenvector of the cameras' "
            f"display-right vectors (roll rms {np.sqrt((roll ** 2).mean()):.1f} deg); bootstrap angle median "
            f"{np.median(ang):.1f} deg, 95th percentile {np.percentile(ang, 95):.1f} deg; Y = horizontal direction of "
            "tray +Y; for the environment and exports")
    print("up (world)", up.round(4), "|", note)
    if "--dry-run" in sys.argv:
        return
    p = CONFIG / "frames.json"
    d = json.loads(p.read_text())
    old = np.array(d["frames"]["scene"]["R"])[:, 2]
    print(f"change vs previous scene up: {np.degrees(np.arccos(np.clip(old @ up, -1, 1))):.2f} deg")
    d["frames"]["scene"] = {"R": R.tolist(), "t_mm": [0.0, 0.0, 0.0], "note": note}
    d["date"] = datetime.date.today().isoformat()
    p.write_text(json.dumps(d, indent=1) + "\n")


if __name__ == "__main__":
    main()
