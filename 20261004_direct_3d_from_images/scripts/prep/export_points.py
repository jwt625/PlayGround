"""Export the sparse COLMAP points in the world frame for the renderer and evaluator (run after world_frame.py).

Writes data/points_world.npy (n,3 meters), points_rgb.npy (n,3 uint8), points_err.npy (n, reprojection px) and
points_views.json (point index -> image stems that observed it). Used by render_views.py (pts pass), evaluate.py
and the wires agent's ray_depth.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402
from tools.geom import load_points  # noqa: E402

X, rgb, err, tracks = load_points()
np.save(DATA / "points_world.npy", X.astype(np.float64))
np.save(DATA / "points_rgb.npy", rgb)
np.save(DATA / "points_err.npy", err)
(DATA / "points_views.json").write_text(json.dumps({str(i): sorted({s for s, _ in tr}) for i, tr in tracks.items()}))
print("points", len(X))
