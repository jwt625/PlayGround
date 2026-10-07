"""Minimal reader for COLMAP binary sparse models (cameras.bin, images.bin, points3D.bin)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    name: str
    camera_id: int
    qvec: np.ndarray  # w, x, y, z (world-to-camera rotation)
    tvec: np.ndarray  # world-to-camera translation
    xys: np.ndarray = field(repr=False)
    point3d_ids: np.ndarray = field(repr=False)

    @property
    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec


@dataclass
class Points:
    ids: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    track_len: np.ndarray
    tracks: dict[int, np.ndarray] = field(repr=False)  # point id -> (n, 2) array of (image_id, point2d_idx)


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rd(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path: Path) -> dict[int, Camera]:
    out = {}
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            cid, mid, w, h = _rd(f, "iiQQ")
            name, k = CAMERA_MODELS[mid]
            out[cid] = Camera(cid, name, w, h, np.array(_rd(f, "d" * k)))
    return out


def read_images(path: Path) -> dict[int, Image]:
    out = {}
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _rd(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            (np2,) = _rd(f, "Q")
            arr = np.frombuffer(f.read(24 * np2), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            out[iid] = Image(iid, name.decode(), cid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]),
                             np.stack([arr["x"], arr["y"]], 1), arr["id"].copy())
    return out


def read_points(path: Path) -> Points:
    ids, xyz, rgb, err, tl, tracks = [], [], [], [], [], {}
    with open(path, "rb") as f:
        (n,) = _rd(f, "Q")
        for _ in range(n):
            pid, x, y, z, r, g, b, e, t = _rd(f, "QdddBBBdQ")
            tr = np.frombuffer(f.read(8 * t), dtype="<i4").reshape(-1, 2)
            ids.append(pid); xyz.append((x, y, z)); rgb.append((r, g, b)); err.append(e); tl.append(t)
            tracks[pid] = tr
    return Points(np.array(ids), np.array(xyz), np.array(rgb, np.uint8), np.array(err), np.array(tl), tracks)


def read_model(d: Path) -> tuple[dict[int, Camera], dict[int, Image], Points]:
    d = Path(d)
    return read_cameras(d / "cameras.bin"), read_images(d / "images.bin"), read_points(d / "points3D.bin")
