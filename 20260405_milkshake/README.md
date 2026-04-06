# Milkshake Meme Pipeline

This folder contains a small reproducible pipeline for turning a trimmed video clip into a face-tracked meme edit with SVG logo overlays.

## Included

- Face detection and lightweight tracking:
  - `detect_faces.py`
  - `render_face_tracks.py`
- Overlay generation:
  - `build_logo_overlay.py`
- Detection/tracking metadata:
  - `face_tracks.json`
  - `face_tracks.csv`
  - `fixed_bbox_sizes.json`
  - `fixed_bbox_sizes.csv`
- Face detector model:
  - `face_detection_yunet_2023mar.onnx`
- SVG asset library:
  - `svg/`
- Minimal Node dependencies used by the overlay builder:
  - `package.json`
  - `package-lock.json`

## Not Tracked

The following are intentionally not meant for git:

- Python virtual environment: `.venv/`
- Installed Node modules: `node_modules/`
- Downloaded source media and generated video outputs such as `.mp4` and `.mov`
- Temporary overlay frame directories

## Workflow Summary

1. Download the source clip locally.
2. Trim the clip to the target scene.
3. Detect faces and assign stable `track_id`s.
4. Compute median bbox sizes for fixed overlay sizing.
5. Prepare SVG icons.
6. Build per-frame overlays and composite them into a final video render.

## Key Files

- `face_tracks.json` is the main per-frame face metadata used for overlays.
- `fixed_bbox_sizes.json` stores the fixed median face sizes used for stable logo scaling.
- `build_logo_overlay.py` renders the logo overlay frames and is the main script for final meme composition.

## Notes

- The final render is intentionally generated locally rather than committed.
- The SVG set was normalized for overlay use, including white variants for monochrome brands and compact icon-only versions where needed.
