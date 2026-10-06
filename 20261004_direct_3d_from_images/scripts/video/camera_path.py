"""Camera path for the model-vs-3DGS comparison video (world frame, meters; see config/world.yaml).

Keyframes are (time s, target xyz mm, distance mm, azimuth deg from +X toward +Y, elevation deg). Each parameter is
interpolated with a Catmull-Rom spline (time-uniform), so moves are smooth and keep moving through keyframes.
Writes outputs/video/path.json: per frame camera center C and look-at target (meters), focal length in px, size.

Usage: uv run python scripts/video/camera_path.py [--fps 30] [--size 1080]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS  # noqa: E402

LENS_MM, SENSOR_MM = 40.0, 36.0  # square frame: horizontal FOV about 48 deg

# t, target (mm), dist (mm), azimuth, elevation   -- -Y is the near long side, +X the electronics end
KEYS = [  # kept inside the training cameras' range (distance 170-320 mm, elevation 17-74 deg) so the 3DGS is fair
    (0.0, (0, 0, 15), 320, -90, 36),     # wide, near side
    (4.0, (0, 0, 15), 310, -20, 34),     # orbit ...
    (8.0, (0, 0, 15), 300, 70, 32),
    (12.0, (0, 0, 15), 300, 160, 32),
    (15.0, (10, 0, 20), 260, 230, 40),   # ... finish the turn, start moving in
    (19.0, (30, 0, 40), 150, 290, 55),   # yoke and its label
    (23.0, (32, 0, 40), 115, 320, 50),   # slow arc around the yoke
    (27.0, (70, -38, 27), 110, 290, 68),  # flyback WARNING label
    (31.0, (88, -10, 24), 120, 330, 35),  # socket cap, neck board
    (35.0, (108, -18, 12), 140, 15, 40),  # external board, pots
    (39.0, (60, -50, 14), 200, 262, 20),  # low tracking shot along the near wall ...
    (44.0, (-60, -50, 14), 200, 256, 20),
    (48.0, (-50, 0, 20), 180, 200, 58),  # screen section, window, card
    (52.0, (0, 0, 30), 190, 160, 74),    # bracket and junction from above
    (56.0, (0, 0, 15), 270, 110, 50),
    (60.0, (0, 0, 15), 320, 60, 40),     # wide end
]


def catmull(ts, ys, t):
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    p0 = ys[max(i - 1, 0)]
    p1, p2 = ys[i], ys[i + 1]
    p3 = ys[min(i + 2, len(ys) - 1)]
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1080)
    a = ap.parse_args()
    ts = np.array([k[0] for k in KEYS])
    tgt = np.array([k[1] for k in KEYS], float)
    dist = np.array([k[2] for k in KEYS], float)
    az = np.unwrap(np.radians([k[3] for k in KEYS]))
    el = np.radians([k[4] for k in KEYS])
    n = int(round(ts[-1] * a.fps))
    frames = []
    for f in range(n):
        t = f / a.fps
        T = catmull(ts, tgt, t)
        d = catmull(ts, dist, t)
        A = catmull(ts, az, t)
        E = catmull(ts, el, t)
        C = T + d * np.array([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)])
        frames.append({"C": (C / 1e3).tolist(), "target": (T / 1e3).tolist()})
    out = OUTPUTS / "video"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"fps": a.fps, "size": a.size, "lens_mm": LENS_MM, "sensor_mm": SENSOR_MM,
            "f_px": LENS_MM / SENSOR_MM * a.size, "n_frames": n, "frames": frames}
    (out / "path.json").write_text(json.dumps(meta))
    print("frames", n, "f_px", round(meta["f_px"], 1))


if __name__ == "__main__":
    main()
