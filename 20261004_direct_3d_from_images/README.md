# Direct 3D modeling from photos (instead of 3DGS)

Agents build an explicit, editable Blender model of a captured object directly from its photos, steered by code
that renders the model at the COLMAP cameras, compares it with the photos, and points the agents' visual
inspection at the worst regions. Test object: a small CRT display unit (137 iPhone photos, a COLMAP sparse model,
and an existing Brush 3DGS as baseline); source capture at `~/Documents/3DGS/20251004_CRT_display` (read-only).

## Current state (2026-10-05, snapshot v3)

- Model: 129 scored objects in 4 part groups (case, crt, pcb, wires) plus a photo-textured mat. Labels, the board
  top, CRT parts and the large case surfaces use 60 photo textures baked from train views. Built by scripts from
  parameters (`config/model/*.toml`), real size in meters, Z up.
- Outputs (generated, not tracked): `outputs/snapshots/v3/crt_model_v3.blend`, viewer-friendly glTF binaries with
  embedded textures `outputs/snapshots/v3/glb/crt_model.glb` (4.7 MB) and `crt_model_with_mat.glb` (5.6 MB),
  new-viewpoint renders `orbit_sheet.jpg`, comparison sheets `compare_holdout/compare.png`.
- Scale: from the mat's molded mm ruler; confirmed by the mat's printed 450 x 300 mm and an independent audit. The
  case measures about 100 x 201 mm (not 12 cm).

## Results (holdout: 14 views never used for measuring, fitting or texturing)

| Metric | v1 rough | v2 | v3 | 3DGS (Brush, as is) |
|---|---|---|---|---|
| Silhouette IoU (known region, 2 px band) | 0.921 | 0.927 | 0.927 | - |
| Model edges to photo edges, mean px (1/4 scale) | 2.92 | 2.88 | 2.86 | - |
| Sparse COLMAP points: median mm to surface / within 2 mm | - | 0.38 / 92 % | 0.39 / 92 % | - |
| PSNR, object region, per-view color fit (dB) | 12.6 | 13.3 | 14.1 | 22.1 |
| Same after a 4 px blur (alignment-tolerant) | - | 15.1 | 16.2 | 26.3 |

Reading the numbers:
- Geometry is close: half the sparse points lie within 0.4 mm of the surface; most of the rest are wires and small
  parts.
- The 3DGS trained on all 137 photos including the holdout views, so its holdout score is optimistic. A flat mean
  color scores 10.6 dB on the same region.
- Error budget (v3 holdout): case 54 percent (window alone 19), wires 12, pcb 12, crt 10, "no model" 12 (mostly
  cast shadows labeled as object in the masks). The window and glossy black case show view-dependent reflections
  of a ceiling and ring lamp that were never photographed (3.5 percent of the upper hemisphere is ever seen), so
  they cannot be reproduced physically from this capture; baking them into textures made things worse.
- Self-texture test on the CRT yoke (texture from a view, render the same view): 16.2 dB. With explicit geometry,
  1-3 px edge misalignment at high-contrast borders caps PSNR; 3DGS, optimized per pixel on those photos, does not
  pay that penalty.

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

Not tracked (regenerated): `data/`, `outputs/`, `assets/textures/*.png`, `.blend`, `.glb`, images and arrays.

## Reproduce

```text
uv sync
uv run python scripts/prep/undistort.py
uv run python scripts/prep/world_frame.py             # writes config/world.yaml (tracked)
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
