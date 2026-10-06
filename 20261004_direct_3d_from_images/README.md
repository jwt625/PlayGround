# Direct 3D modeling from photos (instead of 3DGS)

Agents build an explicit, editable Blender model of a captured object directly from its photos, steered by code
that renders the model at the COLMAP cameras, compares it with the photos, and points the agents' visual
inspection at the worst regions. Test object: a small CRT display unit (137 iPhone photos, a COLMAP sparse model,
and an existing Brush 3DGS as baseline); source capture at `~/Documents/3DGS/20251004_CRT_display` (read-only).

## Current state (2026-10-05, snapshot v4)

- Model: 119 scored objects in 4 part groups (case, crt, pcb, wires) plus a photo-textured mat; 69 photo textures
  baked from train views (labels, every visible pcb component, CRT parts, case panels, board, external board).
  Built by scripts from parameters (`config/model/*.toml`), real size in meters, Z up.
- Outputs (generated, not tracked): `outputs/snapshots/v4/crt_model_v4.blend`, glTF binaries with embedded textures
  `outputs/snapshots/v4/glb/crt_model.glb` (5.2 MB) and `crt_model_with_mat.glb` (6.1 MB), orbit renders,
  comparison sheets; demo video `outputs/video/crt_build_and_compare_20261005.mp4` (see `VIDEO_WORKFLOW.md`).
- Scale: from the mat's molded mm ruler; confirmed by the mat's printed 450 x 300 mm and an independent audit. The
  case measures about 100 x 201 mm (not 12 cm).

## Results (holdout: 14 views never used for measuring, fitting or texturing)

| Metric | v1 rough | v2 | v3 | v4 | 3DGS (Brush, as is) |
|---|---|---|---|---|---|
| Silhouette IoU (known region, 2 px band) | 0.921 | 0.927 | 0.927 | 0.930 | - |
| Model edges to photo edges, mean px (1/4 scale) | 2.92 | 2.88 | 2.86 | 2.80 | - |
| Sparse COLMAP points: median mm to surface / within 2 mm | - | 0.38 / 92 % | 0.39 / 92 % | 0.39 / 92 % | - |
| PSNR, object region, per-view color fit (dB) | 12.6 | 13.3 | 14.1 | 14.6 | 22.1 |
| Same after a 4 px blur (alignment-tolerant) | - | 15.1 | 16.2 | 16.8 | 26.3 |
| SSIM, color fit | 0.36 | 0.38 | 0.41 | 0.42 | 0.79 |

v4 on the 12 probe (train) views: 15.0 dB color-fit, 17.1 dB blurred (3DGS 22.9 / 27.0). v4 holdout error budget:
case 58 percent (window 23.6, tray 10.4, screen_box 7.3, card 6.4), wires 13, crt 10, pcb 9.5, no model 9.5.

Reading the numbers:
- Geometry is close: half the sparse points lie within 0.4 mm of the surface; most of the rest are wires and small
  parts.
- The 3DGS trained on all 137 photos including the holdout views, so its holdout score is optimistic. A flat mean
  color scores 10.6 dB on the same region.
- "No model" error is mostly cast shadows labeled as object in the masks, not missing parts. The window and glossy black case show view-dependent reflections
  of a ceiling and ring lamp that were never photographed (3.5 percent of the upper hemisphere is ever seen), so
  they cannot be reproduced physically from this capture; baking them into textures made things worse.
- Self-texture test on the CRT yoke in IMG_1624 (texture from that view, render the same view): 29.9 dB vs 17.7 dB
  with multi-view textures, so the remaining error is view-dependent shine on tape and metal, not geometry (an
  earlier 16.2 dB version of this test, before the Phase 3 geometry fixes, pointed at edge misalignment).

## Methodology

1. Data prep: undistort SIMPLE_RADIAL to pinhole at 1/4 and 1/2 scale (raw pixel order, EXIF orientation ignored,
   as COLMAP did); world frame from a RANSAC fit of the mat plane (Z up) and case rim corners (axes, origin); metric
   scale from an FFT of the rectified ruler ticks in 75 views; holdout (14) / train (123) / probe (12) split by
   farthest-point sampling of view directions; 3-level masks (object, mat, unknown where the ray leaves the mat)
   from mat color, a mirror check for reflections in glossy walls, and sparse-point evidence.
2. Parts and agents: four part groups, one agent each, which also partitions the object into 3D regions; each group
   is a build script plus a TOML of dimensions. A coordinator owns the shared tools, the environment (photo-
   textured mat) and the evaluation; a last-known-good copy per group keeps everyone's renders whole while one agent
   edits.
3. Loop (about 12 s): build all groups, render the probe views (exact per-object ID pass in Workbench, RGB in EEVEE
   with screen-space ray tracing, point-to-surface distances via BVH), evaluate (silhouette, edge chamfer, color,
   sparse points, orphan clusters), and emit about 8 inspection crops of the worst regions (photo | render | edge
   overlay | error heat). Agents look only at those crops, not at whole photos.
4. Measure first, fit second: picks on labeled-grid crops, epipolar NCC matching with affine-warped patches in
   neighbor views, robust triangulation (about 1 px reprojection); then in-Blender pattern-search fits of TOML
   parameters on edge and excess-geometry loss (about 0.5 s per evaluation, repeatable to about 1 mm near the
   truth, local minima from far starts).
5. Appearance: a render rig calibrated so photo textures reproduce photo values (uniform white world, no lights,
   photo textures without added specular); labels and surfaces textured from train views with ID-owned visibility
   and a per-texel median of the best views.
6. Checks: a fresh-context audit of the measurement chain (confirmed scale, frame and the 3DGS renderer; found
   truncated per-part statistics and holdout leakage, both fixed); milestone bundles via `scripts/final_eval.sh`.

Lessons: rough block-out first, then details and textures; code-ranked crops keep visual inspection cheap; photo
textures with ID visibility gave the largest appearance gains (CRT parts 11.6 -> 15.0 dB inside their pixels,
heat-sink plate 14.1 -> 16.7, case panels +0.7 dB overall); materials alone moved little; reflections need a capture
of the surroundings (a few photos of the ceiling and lights, or a mirror ball).

## Docs for the next batch

| Doc | For |
|---|---|
| `PLAYBOOK.md` | coordinator of a new capture: capture choice, data prep and capture-specific values, phases, metrics, proven techniques, negative results, pitfalls, timing, brief template |
| `AGENT_GUIDE.md` | part agents: loop, measuring tools, render rig, texturing recipe, holdout rule, reports |
| `VIDEO_WORKFLOW.md` | demo videos: specs, viewer stack (three + Spark), camera path, capture, montage, soundtrack, checks |
| `DevLog/parts/` | per-group measurements and methods (e.g. the wire routing pipeline in DevLog-002-wires) |

Next capture: `20250413_coherent_laser` (181/181 registered, cm grid and ruler in frame, rigid machined parts; the
`_v2` capture of the same package is an independent test set). Expect the same appearance ceiling on its gold and
metal surfaces.

## Layout

| Path | Content |
|---|---|
| `DevLog/` | 000 kickoff and dataset pick, 001 plan/toolkit/findings/progress, 003 measurement audit, `parts/` per-group agent logs |
| `AGENT_GUIDE.md` | brief for modeling agents: world frame, loop, tools, holdout rule |
| `config/` | `scene.yaml`, manual picks, extracted world frame (`world.yaml`, `world_extra.yaml`), model parameters (`model/*.toml`), texture specs (`model/env_mat_texture_spec.json`, `textures/pcb/*.json`) |
| `scripts/prep/` | COLMAP reader, undistortion, world frame and scale, split/cameras/masks, edge maps, env map (tried, unused) |
| `scripts/blender/` | render at COLMAP views, parameter fit, orbit renders, save .blend, export .glb |
| `scripts/eval/` | evaluator and region finder, photometric comparison, error budget, 3DGS baseline renderer |
| `scripts/tools/` | measuring (pick, overlay, rayplane), textures from photos, ID-owned bakes |
| `scripts/model/` | `lib.py`, `build_all.py`, one folder per group (build, texture bakers, wire routing tools) |
| `scripts/video/` | camera path, viewer pages, capture, montage assets, soundtrack loop picker (see `VIDEO_WORKFLOW.md`) |

Not tracked (regenerated): `data/`, `outputs/`, `assets/textures/*.png`, `.blend`, `.glb`, images and arrays.

## Reproduce (full order with the two-pass mat bake: PLAYBOOK.md section 3)

```text
uv sync
uv run python scripts/prep/undistort.py
uv run python scripts/prep/world_frame.py             # writes config/world.yaml (tracked)
uv run python scripts/prep/export_points.py           # sparse points in the world frame (data/points_*.npy)
uv run python scripts/prep/split_cams_masks.py        # split, cameras, masks (mirror check uses the mat texture)
uv run python scripts/prep/edges.py
uv run python scripts/tools/texture_from_photos.py config/model/env_mat_texture_spec.json   # mat texture
# part textures: scripts/model/crt/texture_yoke.py, scripts/model/case/bake_case.py,
#   scripts/tools/bake_id_owned.py with config/textures/pcb/*.json, texture_from_photos.py with
#   scripts/model/crt/label_spec.json (each needs an ID render of the train views; see the part devlogs)
scripts/iterate.sh <run> probe [parts_prefix] [geom|full]  # one iteration, about 12 s
scripts/final_eval.sh <tag>                                # milestone bundle incl. .blend and .glb
uv run python scripts/eval/render_3dgs.py --views holdout  # baseline renders (CPU, about 6 s per view)
```

Blender 4.2 at `/Applications/Blender.app`, always through `scripts/bslot.sh` (at most 2 concurrent instances).
Known gap: the wires group's external-board and pot texture spec files were not kept; the board quad is in
`config/model/wires.toml` [ext_board] (texture_from_photos on that quad, 12 train views, ID occlusion; see
`DevLog/parts/DevLog-002-wires.md`).
