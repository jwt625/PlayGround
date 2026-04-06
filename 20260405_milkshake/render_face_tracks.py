from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


COLORS = {
    1: (80, 220, 80),
    2: (80, 180, 255),
    3: (255, 180, 80),
    4: (255, 80, 180),
}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_bbox(box_a: list[float], box_b: list[float], t: float) -> list[float]:
    return [lerp(a, b, t) for a, b in zip(box_a, box_b)]


def load_tracks(metadata_path: Path) -> tuple[dict[int, dict[int, dict[str, object]]], int]:
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
    return track_frames, int(data["frame_step"])


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

        dense_tracks[track_id] = {k: v for k, v in dense.items() if 0 <= k < frame_count}

    return dense_tracks


def draw_box(frame: object, track_id: int, bbox: list[float], confidence: float) -> None:
    x, y, w, h = [int(round(v)) for v in bbox]
    color = COLORS.get(track_id, (255, 255, 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    label = f"id {track_id} {confidence:.2f}"
    text_origin = (x, max(24, y - 10))
    cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def render_video(
    video_path: Path,
    metadata_path: Path,
    output_path: Path,
    interpolation_limit: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    track_frames, frame_step = load_tracks(metadata_path)
    dense_tracks = build_dense_tracks(
        track_frames=track_frames,
        frame_count=frame_count,
        interpolation_limit=max(interpolation_limit, frame_step),
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output for writing: {output_path}")

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            for track_id, frames in dense_tracks.items():
                payload = frames.get(frame_index)
                if payload is None:
                    continue
                draw_box(frame, track_id, payload["bbox"], float(payload["confidence"]))

            cv2.putText(
                frame,
                f"frame {frame_index}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracked face bboxes onto a video.")
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--metadata", type=Path, default=Path("face_tracks.json"), help="Tracking JSON metadata")
    parser.add_argument("--output", type=Path, default=Path("milkshake_face_tracks_preview.mp4"), help="Output preview video")
    parser.add_argument(
        "--interpolation-limit",
        type=int,
        default=6,
        help="Only interpolate gaps up to this many frames. Default: 6",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video(
        video_path=args.video,
        metadata_path=args.metadata,
        output_path=args.output,
        interpolation_limit=args.interpolation_limit,
    )


if __name__ == "__main__":
    main()
