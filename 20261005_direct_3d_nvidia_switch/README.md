# Direct 3D model of a CPO switch tray and its package from photos

Agents build an explicit, editable Blender model of an NVIDIA "Spectrum-X CPO Switch Tray" and the CPO package
displayed in front of it, directly from 44 iPhone photos, steered by code that renders the model at the solved
cameras, compares it with the photos, and points visual inspection at the worst regions. Second capture after
`../20261004_direct_3d_from_images` (CRT); unlike that run there is no prior COLMAP model, no 3DGS baseline and no
ruler in frame.

## Current state (2026-10-06)

- Phase 0 done (DevLog-001, audited in DevLog-003): own COLMAP model (44/44 registered, 1.31 px; IMG_5739
  excluded), tray frame and metric scale from the published SN6810-LD body width (438 mm; audit: factor 1.000,
  0.995-1.002), gravity and package frames, holdout split by session, SAM 2.1 masks, adapted CRT toolkit.
- Models v1-v4 (`outputs/snapshots/v1..v4/`, DevLog-004; audits DevLog-003 and DevLog-005): 4 groups (chassis,
  upper, lower, package) plus a basic environment, photo-textured. Holdout (5 views, never used for measuring,
  fitting or texturing):

| Metric (holdout) | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Silhouette IoU | 0.967 | 0.968 | 0.968 | 0.968 |
| Edges, mean px (1/4 scale) | 3.30 | 3.24 | 3.21 | 3.15 |
| Sparse points, median mm | 1.19 | 1.14 | 1.13 | 1.11 |
| PSNR color fit (dB); flat-color floor 11.28 | 12.94 | 13.82 | 13.97 | 14.00 |
| Same after 4 px blur (dB); floor 12.09 | 14.23 | 15.45 | 15.62 | 15.65 |
| SSIM color fit | 0.359 | 0.376 | 0.381 | 0.385 |

- v4 is the current model: `outputs/snapshots/v4/nvsw_model_v4.blend`, `glb/nvsw_tray_model.glb` (5.8 MB,
  gravity frame, JPEG textures) and `_with_env.glb`, orbit renders. Summary and lessons: DevLog-004 section 6.
- Known limits: sun shadows and shine differ between the three capture sessions (a single texture per surface
  cannot match all views); never-photographed faces (underside, package back) are untextured in the wall color;
  a washed-out patch on the rear-bay +X deck.

## Layout

| Path | Content |
|---|---|
| `DevLog/` | 000 kickoff and plan, 001 Phase 0; `parts/` per-group agent logs |
| `AGENT_GUIDE.md` | brief for part agents (frame, loop, tools, holdout rule) |
| `config/` | `scene.yaml` (paths, sessions, split rules), `picks.yaml` (frame inputs), `world.yaml` (generated), `frames.json` (scene/gravity and package frames), `mask_prompts.yaml`, `eval.yaml`, `model/*.toml` |
| `scripts/prep/` | `run_colmap.sh`, undistort, split, world_frame, masks_sam (SAM 2.1), cameras_masks, edges, export_points |
| `scripts/blender/`, `scripts/eval/`, `scripts/tools/`, `scripts/model/` | from the CRT run: render, fit, evaluate, measure, texture; model groups |
| `references/md/dimensions_research.md` | published dimensions (tray, package, standard parts) with sources |

Not tracked: `source/` (photo copies), `data/`, `outputs/`, images, arrays, .blend, .glb.
