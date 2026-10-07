"""Undistort the capture (SIMPLE_RADIAL -> PINHOLE, same focal length and principal point) at reduced scales.

Pipeline per image: raw pixels (EXIF orientation ignored, as COLMAP did) -> INTER_AREA downscale by s ->
remap with the scaled intrinsics. Pixel convention: continuous coordinate u = index + 0.5, as in COLMAP.
Output: data/undistorted_s{s}/<name>.jpg and data/undistorted_s{s}/intrinsics.json. Skips existing files.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, scene_cfg  # noqa: E402
from prep.colmap_io import read_cameras, read_images  # noqa: E402


def load_raw(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(path)
    return img


def undistort_maps(w: int, h: int, f: float, cx: float, cy: float, k1: float) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = (u - cx) / f, (v - cy) / f
    r2 = x * x + y * y
    d = 1 + k1 * r2
    return (f * x * d + cx - 0.5).astype(np.float32), (f * y * d + cy - 0.5).astype(np.float32)


def scaled_intrinsics(cam, s: int) -> dict:
    f, cx, cy, k1 = cam.params
    return {"width": cam.width // s, "height": cam.height // s, "fx": f / s, "fy": f / s, "cx": cx / s,
            "cy": cy / s, "k1_removed": k1}


def work(args):
    src, dst, s, intr = args
    if dst.exists():
        return dst.name, "skip"
    img = load_raw(src)
    w, h = intr["width"], intr["height"]
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    mx, my = undistort_maps(w, h, intr["fx"], intr["cx"], intr["cy"], intr["k1_removed"])
    out = cv2.remap(small, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst.name, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    cfg = scene_cfg()
    src_dir = cfg["source_dir"]
    cams = read_cameras(cfg["sparse_dir"] / "cameras.bin")
    imgs = read_images(cfg["sparse_dir"] / "images.bin")
    for s in a.scale or cfg["work_scales"]:
        out_dir = DATA / f"undistorted_s{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        intr = {cid: scaled_intrinsics(c, s) for cid, c in cams.items()}
        (out_dir / "intrinsics.json").write_text(json.dumps({"scale": s, "cameras": intr}, indent=1))
        jobs = [(src_dir / "images" / im.name, out_dir / (Path(im.name).stem + ".jpg"), s, intr[im.camera_id])
                for im in imgs.values()]
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(work, jobs))
        print(f"scale {s}: {sum(r == 'ok' for _, r in res)} written, {sum(r == 'skip' for _, r in res)} skipped")


if __name__ == "__main__":
    main()
