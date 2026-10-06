"""Set up outputs/video/viewer/ for the demo videos: cfg.json (camera path in the glb frame, splat placement,
panel size, field of view) and the symlinks the pages need.

Usage: uv run python scripts/video/make_viewer_cfg.py --glb outputs/snapshots/<tag>/glb/crt_model_with_mat.glb
         [--ply <3dgs ply>] [--node-modules ~/Documents/GitHub/3dgs-viewer/node_modules] [--w 1080 --h 810]
Run scripts/video/camera_path.py first (outputs/video/path.json). The ply must be in the COLMAP frame of
sparse/0 (true for the Brush export of the CRT set; check a new capture's ply first, e.g. nearest-neighbour
distance of sparse points to Gaussian centers as in DevLog-001).
Frames: world (Z up, config/world.yaml) -> glb (Y up, Blender glTF exporter: x, z, -y); splat placement =
glb_from_world o world_from_colmap (rotation quaternion xyzw, uniform scale, position).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS, ROOT, scene_cfg  # noqa: E402
from tools.geom import world_transform  # noqa: E402

A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])  # world (Z up) -> glTF (Y up)


def quat_xyzw(m):
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1) * 2
        return [(m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S, 0.25 * S]
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        S = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return [0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S, (m[2, 1] - m[1, 2]) / S]
    if i == 1:
        S = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return [(m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S, (m[0, 2] - m[2, 0]) / S]
    S = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return [(m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S, (m[1, 0] - m[0, 1]) / S]


def link(dst: Path, src: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--ply", default=None)
    ap.add_argument("--node-modules", default=str(Path("~/Documents/GitHub/3dgs-viewer/node_modules").expanduser()))
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=810)
    a = ap.parse_args()
    v = OUTPUTS / "video" / "viewer"
    v.mkdir(parents=True, exist_ok=True)
    P = json.loads((OUTPUTS / "video" / "path.json").read_text())
    s, Rw, tw = world_transform()
    M = A @ Rw
    cams = [{"C": (A @ np.array(f["C"])).tolist(), "T": (A @ np.array(f["target"])).tolist()} for f in P["frames"]]
    cfg = {"splat": {"position": (A @ tw).tolist(), "rotation": quat_xyzw(M), "scale": [s, s, s]},
           "fov_deg": float(np.degrees(2 * np.arctan(P["sensor_mm"] / 2 / P["lens_mm"]))),  # horizontal FOV
           "size": a.w, "w": a.w, "h": a.h, "cams": cams}
    (v / "cfg.json").write_text(json.dumps(cfg))
    nm = Path(a.node_modules)
    ply = Path(a.ply) if a.ply else next(scene_cfg()["source_dir"].glob("*.ply"))
    for dst, src in [("index.html", ROOT / "scripts/video/viewer/index.html"),
                     ("montage.html", ROOT / "scripts/video/viewer/montage.html"),
                     ("model.glb", ROOT / a.glb), ("scene.ply", ply),
                     ("three", nm / "three"), ("spark", nm / "@sparkjsdev/spark"),
                     ("montage", OUTPUTS / "video" / "montage")]:
        (OUTPUTS / "video" / "montage").mkdir(parents=True, exist_ok=True)
        link(v / dst, src)
    print("cfg.json", len(cams), "cams; det", round(float(np.linalg.det(M)), 4), "links ok")


if __name__ == "__main__":
    main()
