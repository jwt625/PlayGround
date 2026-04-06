from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_frame: int
    misses: int = 0
    history: list[dict[str, object]] = field(default_factory=list)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center_distance(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def match_detections_to_tracks(
    tracks: list[Track],
    detections: list[Detection],
    frame_index: int,
    next_track_id: int,
    max_center_distance: float,
    min_iou: float,
    max_misses: int,
) -> tuple[list[dict[str, object]], int]:
    assignments: list[tuple[float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            overlap = iou(track.bbox, detection.bbox)
            distance = center_distance(track.bbox, detection.bbox)
            if overlap >= min_iou or distance <= max_center_distance:
                score = overlap - (distance / max_center_distance) * 0.01
                assignments.append((score, track_idx, det_idx))

    assignments.sort(reverse=True)
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    frame_results: list[dict[str, object]] = []

    for _, track_idx, det_idx in assignments:
        if track_idx in used_tracks or det_idx in used_detections:
            continue
        track = tracks[track_idx]
        detection = detections[det_idx]
        track.bbox = detection.bbox
        track.last_frame = frame_index
        track.misses = 0
        used_tracks.add(track_idx)
        used_detections.add(det_idx)
        frame_results.append(
            {
                "track_id": track.track_id,
                "bbox": list(detection.bbox),
                "confidence": round(detection.score, 5),
            }
        )

    for track_idx, track in enumerate(tracks):
        if track_idx not in used_tracks:
            track.misses += 1

    tracks[:] = [track for track in tracks if track.misses <= max_misses]

    for det_idx, detection in enumerate(detections):
        if det_idx in used_detections:
            continue
        track = Track(track_id=next_track_id, bbox=detection.bbox, last_frame=frame_index)
        tracks.append(track)
        frame_results.append(
            {
                "track_id": track.track_id,
                "bbox": list(detection.bbox),
                "confidence": round(detection.score, 5),
            }
        )
        next_track_id += 1

    frame_results.sort(key=lambda item: item["track_id"])
    return frame_results, next_track_id


def detect_faces(
    video_path: Path,
    model_path: Path,
    output_json: Path,
    output_csv: Path,
    frame_step: int,
    confidence: float,
    max_faces: int,
    max_misses: int,
    min_track_hits: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        (width, height),
        score_threshold=confidence,
        nms_threshold=0.3,
        top_k=max(10, max_faces * 2),
    )

    tracks: list[Track] = []
    next_track_id = 1
    frames_payload: list[dict[str, object]] = []

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            detections: list[Detection] = []
            detector.setInputSize((width, height))
            _, result = detector.detect(frame_bgr)

            if result is not None:
                sorted_result = sorted(result, key=lambda row: float(row[14]), reverse=True)
                for detection in sorted_result[:max_faces]:
                    x, y, w, h = [round(float(value), 2) for value in detection[:4]]
                    score = float(detection[14])
                    detections.append(Detection(bbox=(x, y, w, h), score=score))

            frame_results, next_track_id = match_detections_to_tracks(
                tracks=tracks,
                detections=detections,
                frame_index=frame_index,
                next_track_id=next_track_id,
                max_center_distance=max(width, height) * 0.18,
                min_iou=0.2,
                max_misses=max_misses,
            )

            frames_payload.append(
                {
                    "frame_index": frame_index,
                    "time_sec": round(frame_index / fps, 6),
                    "detections": frame_results,
                }
            )
            frame_index += 1
    finally:
        capture.release()

    track_counts = Counter()
    first_seen: dict[int, int] = {}
    for frame in frames_payload:
        for detection in frame["detections"]:
            track_id = int(detection["track_id"])
            track_counts[track_id] += 1
            first_seen.setdefault(track_id, int(frame["frame_index"]))

    surviving_tracks = [
        track_id
        for track_id, hits in sorted(track_counts.items(), key=lambda item: (first_seen[item[0]], item[0]))
        if hits >= min_track_hits
    ]
    track_id_map = {track_id: index + 1 for index, track_id in enumerate(surviving_tracks)}

    for frame in frames_payload:
        filtered_detections = []
        for detection in frame["detections"]:
            mapped_track_id = track_id_map.get(int(detection["track_id"]))
            if mapped_track_id is None:
                continue
            filtered_detections.append(
                {
                    "track_id": mapped_track_id,
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                }
            )
        frame["detections"] = filtered_detections

    payload = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frame_step": frame_step,
        "tracking": {
            "method": "opencv_yunet + iou_center_tracker",
            "detector_model": str(model_path),
            "max_faces": max_faces,
            "max_missed_samples": max_misses,
            "min_track_hits": min_track_hits,
        },
        "frames": frames_payload,
    }

    output_json.write_text(json.dumps(payload, indent=2))

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "time_sec",
                "track_id",
                "x",
                "y",
                "width",
                "height",
                "confidence",
            ],
        )
        writer.writeheader()
        for frame in frames_payload:
            for detection in frame["detections"]:
                x, y, w, h = detection["bbox"]
                writer.writerow(
                    {
                        "frame_index": frame["frame_index"],
                        "time_sec": frame["time_sec"],
                        "track_id": detection["track_id"],
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": detection["confidence"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and track faces in a video clip.")
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("face_detection_yunet_2023mar.onnx"),
        help="Path to the YuNet ONNX model file.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Run detection every N frames. Default: 3",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence. Default: 0.5",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=2,
        help="Maximum faces to keep per sampled frame. Default: 2",
    )
    parser.add_argument(
        "--max-misses",
        type=int,
        default=2,
        help="How many sampled frames a track can survive without a match. Default: 2",
    )
    parser.add_argument(
        "--min-track-hits",
        type=int,
        default=3,
        help="Drop tracks that appear in fewer than this many sampled frames. Default: 3",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("face_tracks.json"),
        help="JSON metadata output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("face_tracks.csv"),
        help="CSV metadata output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detect_faces(
        video_path=args.video,
        model_path=args.model,
        output_json=args.output_json,
        output_csv=args.output_csv,
        frame_step=args.frame_step,
        confidence=args.confidence,
        max_faces=args.max_faces,
        max_misses=args.max_misses,
        min_track_hits=args.min_track_hits,
    )


if __name__ == "__main__":
    main()
