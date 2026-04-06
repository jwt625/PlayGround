from __future__ import annotations

import json
import io
import random
import subprocess
import shutil
import math
from pathlib import Path

from PIL import Image


WIDTH = 1920
HEIGHT = 800
RANDOM_SEED = 7
START_RANDOM = 402
END_RANDOM = 442
ICON_SCALE_UP = 1.2

AI_LABS = [
    "openai",
    "anthropic",
    "xai",
    "mistral-ai",
    "deepseek",
    "qwen",
    "z-ai",
    "moonshot-ai",
]

ALL_ICONS = [
    "openai",
    "anthropic",
    "xai",
    "mistral-ai",
    "deepseek",
    "qwen",
    "z-ai",
    "moonshot-ai",
    "reddit",
    "wikipedia",
    "arxiv",
    "new-york-times",
    "the-washington-post",
    "github",
    "stack-exchange",
    "hugging-face",
]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_bbox(box_a: list[float], box_b: list[float], t: float) -> list[float]:
    return [lerp(a, b, t) for a, b in zip(box_a, box_b)]


def load_tracks(metadata_path: Path) -> tuple[dict[int, dict[int, dict[str, object]]], int, int]:
    data = json.loads(metadata_path.read_text())
    track_frames: dict[int, dict[int, dict[str, object]]] = {}
    for frame in data["frames"]:
        frame_index = int(frame["frame_index"])
        for detection in frame["detections"]:
            track_id = int(detection["track_id"])
            track_frames.setdefault(track_id, {})[frame_index] = {
                "bbox": [float(v) for v in detection["bbox"]],
                "confidence": float(detection["confidence"]),
            }
    return track_frames, int(data["frame_step"]), int(data["frame_count"])


def build_dense_tracks(
    track_frames: dict[int, dict[int, dict[str, object]]],
    frame_count: int,
    interpolation_limit: int,
) -> dict[int, dict[int, dict[str, object]]]:
    dense_tracks: dict[int, dict[int, dict[str, object]]] = {}
    for track_id, frames in track_frames.items():
        sorted_items = sorted(frames.items())
        dense: dict[int, dict[str, object]] = {}
        for index, (frame_index, payload) in enumerate(sorted_items):
            dense[frame_index] = payload
            if index == len(sorted_items) - 1:
                continue
            next_frame_index, next_payload = sorted_items[index + 1]
            gap = next_frame_index - frame_index
            if gap <= 1 or gap > interpolation_limit:
                continue
            for between in range(frame_index + 1, next_frame_index):
                t = (between - frame_index) / gap
                dense[between] = {
                    "bbox": lerp_bbox(payload["bbox"], next_payload["bbox"], t),
                    "confidence": lerp(payload["confidence"], next_payload["confidence"], t),
                }
        dense_tracks[track_id] = dense
    return dense_tracks


def load_fixed_sizes(path: Path) -> dict[int, tuple[float, float]]:
    data = json.loads(path.read_text())
    sizes: dict[int, tuple[float, float]] = {}
    for track in data["tracks"]:
        sizes[int(track["track_id"])] = (
            float(track["median_width"]),
            float(track["median_height"]),
        )
    return sizes


def get_svg_dimensions(svg_path: Path) -> tuple[float, float]:
    result = subprocess.run(
        [
            "node",
            "-e",
            (
                "const fs=require('fs');"
                "const sharp=require('sharp');"
                "sharp(fs.readFileSync(process.argv[1]))"
                ".metadata()"
                ".then(meta=>console.log(`${meta.width} ${meta.height}`))"
                ".catch(err=>{console.error(err);process.exit(1);});"
            ),
            str(svg_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = result.stdout.strip().split()
    return float(width), float(height)


def render_icon(svg_path: Path, target_diagonal: float) -> Image.Image:
    base_width, base_height = get_svg_dimensions(svg_path)
    scale = target_diagonal / math.hypot(base_width, base_height)
    out_width = max(1, int(round(base_width * scale)))
    out_height = max(1, int(round(base_height * scale)))
    result = subprocess.run(
        [
            "node",
            "-e",
            (
                "const fs=require('fs');"
                "const sharp=require('sharp');"
                "const [svg,w,h]=process.argv.slice(1);"
                "sharp(fs.readFileSync(svg))"
                ".resize(Number(w), Number(h), {fit:'fill'})"
                ".png()"
                ".toBuffer()"
                ".then(buf=>process.stdout.write(buf))"
                ".catch(err=>{console.error(err);process.exit(1);});"
            ),
            str(svg_path),
            str(out_width),
            str(out_height),
        ],
        check=True,
        capture_output=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGBA")


def pick_icons(frame_index: int, rng: random.Random) -> dict[int, str]:
    if frame_index < START_RANDOM:
        return {1: "openai", 2: "reddit"}
    if frame_index <= END_RANDOM:
        return {
            1: rng.choice(AI_LABS),
            2: rng.choice(ALL_ICONS),
        }
    return {1: "deepseek", 2: "anthropic"}


def main() -> None:
    metadata_path = Path("face_tracks.json")
    fixed_sizes_path = Path("fixed_bbox_sizes.json")
    svg_dir = Path("svg")
    out_dir = Path("overlay_frames")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    track_frames, frame_step, frame_count = load_tracks(metadata_path)
    dense_tracks = build_dense_tracks(track_frames, frame_count, interpolation_limit=max(frame_step, 6))
    fixed_sizes = load_fixed_sizes(fixed_sizes_path)

    needed_icons = set(AI_LABS) | set(ALL_ICONS)
    cached_icons: dict[tuple[str, int], Image.Image] = {}
    for icon_name in needed_icons:
        svg_path = svg_dir / f"{icon_name}.svg"
        for track_id, size in fixed_sizes.items():
            target_diagonal = math.hypot(size[0], size[1]) * ICON_SCALE_UP
            cached_icons[(icon_name, track_id)] = render_icon(svg_path, target_diagonal)

    rng = random.Random(RANDOM_SEED)

    for frame_index in range(frame_count):
        canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        icons = pick_icons(frame_index, rng)
        for track_id in (1, 2):
            payload = dense_tracks.get(track_id, {}).get(frame_index)
            if payload is None:
                continue
            icon_name = icons[track_id]
            icon = cached_icons[(icon_name, track_id)]
            bbox = payload["bbox"]
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            x = int(round(cx - icon.width / 2))
            y = int(round(cy - icon.height / 2))
            x = max(0, min(WIDTH - icon.width, x))
            y = max(0, min(HEIGHT - icon.height, y))
            canvas.alpha_composite(icon, (x, y))

        frame_path = out_dir / f"frame_{frame_index:04d}.png"
        canvas.save(frame_path)


if __name__ == "__main__":
    main()
