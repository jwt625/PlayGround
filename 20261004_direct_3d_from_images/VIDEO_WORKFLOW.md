# Demo video workflow (explicit model vs 3DGS, build montage)

For the agent that makes the demo videos. Everything is recorded from the real files (`.glb` of the explicit
model, `.ply` of the 3DGS) in a browser viewer; nothing is re-rendered offline. Total time under 5 minutes.

## Final cut spec

| Item | Value |
|---|---|
| Structure | 30 s build montage, then 30 s comparison (recorded at 60 s, played at 2x camera speed) = 60 s |
| Frame | 1080x1620 (2:3): two 1080x810 (4:3) panels stacked; top = explicit model, bottom = 3DGS |
| Video | H.264 High, level 4.2, yuv420p, tv range, 30 fps constant, CRF 20, `+faststart` |
| Audio | AAC 192 kb/s, 48 kHz stereo; a loopable section picked by `pick_loop.py`; tempo change at most 1.25x with pitch kept (`atempo`); 20 ms fades at both ends |
| Labels | top-left of each panel, white bold with black outline; montage step caption bottom-left |

## Viewer stack

- Same stack as `~/Documents/GitHub/3dgs-viewer`: three 0.181 + Spark 0.1.10, linked from that repo's
  `node_modules` (no downloads).
- Setup (after `camera_path.py`): `uv run python scripts/video/make_viewer_cfg.py --glb outputs/snapshots/<tag>/glb/crt_model_with_mat.glb`
  writes `outputs/video/viewer/cfg.json` (per-frame cameras in the glb frame, splat placement, horizontal field of
  view, panel size 1080x810) and the 7 links the pages need: `index.html`, `montage.html` (from
  `scripts/video/viewer/`), `model.glb`, `scene.ply`, `three` (node_modules/three), `spark`
  (node_modules/@sparkjsdev/spark), `montage` (outputs/video/montage).
- Serve: `python3 -m http.server <uncommon free port> --bind 127.0.0.1 --directory outputs/video/viewer` (check the
  port first; stop the server when done).
- Frames: the glb is Y-up (exporter); the ply is in the COLMAP frame (check a new capture's ply) and is placed by
  cfg "splat" (rotation, scale, position from `config/world.yaml` composed with the Z-up to Y-up swap).
- glb lighting: ambient white 3.0, no tone mapping, sRGB output (looked right against the Blender renders).

## Camera path

`scripts/video/camera_path.py`: keyframes (time, target, distance, azimuth, elevation) with Catmull-Rom
interpolation; 60 s at 30 fps. Keep distance and elevation inside the training cameras' range (CRT: 170-320 mm,
17-74 deg); outside it the 3DGS shows floaters (wide shots at 430 mm and a 12 deg pass were rejected). Shots used:
orbit, yoke label, flyback label, socket cap, external board, low pass along the case, window, top-down over the
bracket, pull-out. `cfg.json` holds per-frame camera centers and targets in the glb frame.

## Capture

```text
uv run python scripts/video/camera_path.py                                      # 60 s path -> outputs/video/path.json
uv run python scripts/video/make_viewer_cfg.py --glb <glb>                     # cfg.json + links (see Viewer stack)
uv run --with playwright python scripts/video/capture_viewer.py --port P        # comparison, real time
uv run python scripts/video/gen_montage.py                                      # montage plan, cards, splat stages
uv run --with playwright python scripts/video/capture_montage.py --port P       # montage, 150 images
```
- Headless installed Chrome (`channel="chrome"`, fresh temporary profile; never the user's profile).
- Comparison: the page draws both renderers into one canvas and records it with MediaRecorder (VP9, 40 Mb/s) in
  real time (60 s recording; 1:21-1:24 including page load and transcode), then `capture_viewer.py` transcodes to
  H.264 CRF 18 and converts full range to tv range (`scale=in_range=pc:out_range=tv,format=yuv420p`).
- Montage: 5 images per second, 150 images, page screenshots of the canvas piped to ffmpeg (about 55 s).
- Do not take per-frame screenshots for the 60 s comparison (that was the slow path), and do not write a custom
  splat renderer.

## Montage content

- Model panel: the glb revealed in steps over the 30 s with the same orbit: 1 block-out by part group (flat gray),
  2 materials (each part's mean texture color), 3 labels from photos, 4 photo textures by group, 5 final. The case
  keeps its final materials throughout (its textured panels carry the see-through look of the open window/tray;
  drawn flat they become lids). Transparent materials are never overridden.
- Cards: real artifacts of the agents' loop (worst-region sheets per agent, view sheets, pick crops, texture
  sheets, comparison sheets) and code flashes (rendered from the real scripts). Card registry
  `outputs/video/montage/cards.json` is append-only: cards are rendered once into `montage/cards/` and kept;
  rebuilds only append new ones (sources such as agent runs may be deleted later). 3 frames per card.
- 3DGS panel: "3DGS training (illustration)": 30 `.splat` stages interpolated from random Gaussians to the final
  ply (count grows to the full set by iteration 12k, per-Gaussian exponential convergence with time constants
  300-1500 iterations, exact final at 30k), shown in Spark with the same orbit. Label it as an illustration.

## Combine and soundtrack

```text
ffmpeg -i track.m4a -ac 1 -ar 22050 outputs/scratch/track.wav          # librosa cannot read m4a
uv run --with librosa python scripts/video/pick_loop.py outputs/scratch/track.wav 59.966667 --skip 10 --max-speed 1.25
# -> BEST start S, length L; tempo T = L / 59.966667 (CRT: S 132.934, L 61.672, T 1.028437)
cd outputs/video
ffmpeg -y -i crt_build_montage_<date>.mp4 -i crt_model_vs_3dgs_<date>.mp4 -ss S -t L -i track.m4a \
  -filter_complex "[1:v]setpts=0.5*PTS,fps=30[f];[0:v][f]concat=n=2:v=1:a=0,fps=30,format=yuv420p[v];[2:a]atempo=T,aresample=48000,afade=t=in:d=0.02,afade=t=out:st=59.94:d=0.02[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset slow -crf 20 -profile:v high -level 4.2 -pix_fmt yuv420p \
  -color_range tv -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart -shortest crt_build_and_compare_<date>.mp4
```
- `pick_loop.py` beat-tracks the track (decode m4a to wav first; libsndfile cannot read m4a), scores bar-aligned
  sections by loop similarity (chroma + MFCC + RMS right after start vs right after end) and loudness match.
  CRT cut: 2:12.93, 61.67 s, tempo 1.028.
- Keep the user's audio files untouched; temporary wavs go to scratch and are deleted.

## Verify before delivering

ffprobe (size, pix_fmt yuv420p, 30 fps, duration, audio stream), a 1 fps contact sheet of each panel, the join
between the two parts, and loudness (`volumedetect`). Remove temporary frames, splat stages and servers.
