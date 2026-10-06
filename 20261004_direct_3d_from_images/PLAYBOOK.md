# Playbook: explicit 3D model from a photo capture (coordinator guide)

For the coordinator (fresh context) of the next capture. Distilled from the CRT run (2026-10-04 to 10-05;
DevLog-000/001/003 and `DevLog/parts/`). Part agents read `AGENT_GUIDE.md`; demo videos follow
`VIDEO_WORKFLOW.md`.

## 1. What to expect

| Outcome on the CRT (14 holdout views) | Value |
|---|---|
| Geometry: sparse COLMAP points within 2 mm of the surface | 92 percent (median 0.39 mm) |
| Gain per phase (holdout, color-fit) | rough 12.6 -> details/labels 13.3 -> case textures 14.1 -> all components 14.6 dB |
| Silhouette IoU / model-to-photo edge distance | 0.930 / 2.80 px (1/4 scale) |
| Photometric, object region, per-view color fit | v1 12.6, v2 13.3, v3 14.1, v4 14.6 dB vs 3DGS 22.1 dB (blur 4 px: v4 16.8 vs 26.3) |
| Flat mean color on the same region (floor, one view: IMG_1522; DevLog-003) | 10.6 dB |

The ceiling is view-dependent appearance: glass, glossy plastic, tape and metal show reflections and shine
that a fixed texture cannot follow (self-texture test on the yoke in IMG_1624: 29.9 dB textured from that view vs
17.7 dB with multi-view textures; DevLog-002-crt). The 3DGS fits view dependence through its spherical harmonics, and it trains on every
photo, holdout included. Report both facts with every comparison.

## 2. Pick the capture

Prefer, in order of impact:
1. Registration near 100 percent and a sharp sparse model (median reprojection error 1.0-1.5 px).
2. A metric reference in frame: a ruler, grid mat or printed dimension. Ruler ticks give about 1 percent scale
   from an FFT over many views; a known object size needs an independent check (the stated "12 cm" CRT width
   was wrong by 21 percent; the ruler and the mat's printed size settled it).
3. Rigid, mostly matte parts; glossy and transparent areas cap photometric scores.
4. Some views of the surroundings and lights (ceiling): without them reflections cannot be modeled. Only
   3.5 percent of the upper hemisphere was ever seen in the CRT set.

Candidates surveyed in DevLog-000: `20250413_coherent_laser` (181/181 registered, cm grid + ruler, rigid
machined parts, and `20250414_coherent_laser_v2` as an independent second capture usable as a true test set;
gold surfaces are glossy). Avoid `20250502_blueFors` (32/154 registered, broken pose).

## 3. Phase 0: data prep (coordinator, about 2-4 h on the CRT)

Order matters (new project folder; copy the scripts, do not import across projects):
```text
uv sync
uv run python scripts/prep/undistort.py              # 1/4 and 1/2 scale pinhole images, EXIF ignored
uv run python scripts/prep/world_frame.py            # needs config/picks.yaml; writes config/world.yaml
uv run python scripts/prep/export_points.py          # data/points_world.npy, _rgb, _err, points_views.json
uv run python scripts/prep/split_cams_masks.py       # split, Blender cameras, masks (no mirror check yet)
uv run python scripts/prep/edges.py                  # edge distance maps for fit_params
# background plane texture, pass 1: spec with "exclude_object_mask": true (no "exclude_model_run" yet)
uv run python scripts/tools/texture_from_photos.py config/model/env_mat_texture_spec.json
uv run python scripts/prep/split_cams_masks.py --force   # masks again, now with the mirror check (needs env_mat.png)
# after the rough block-out (Phase 1A): ID-render the train views and bake the background again
scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- --out outputs/runs/coord_idtrain \
  --build scripts/model/build_all.py --views train --passes id
#   spec: "exclude_model_run": "outputs/runs/coord_idtrain", then rerun texture_from_photos.py on the spec
uv run python scripts/eval/render_3dgs.py --views holdout   # baseline renders for metrics (videos use the viewer)
```

Capture-specific values to change (all CRT-specific today; moving them into `config/` is a good first task):

| Where | What |
|---|---|
| `config/scene.yaml` | `source_dir`, holdout fraction and seed (`case_width_mm` is informational; nothing reads it) |
| `config/picks.yaml` | `ruler_seed` (two pixels along the ruler in one view), `case_rim_corners` (fixed names near_left, far_left, far_right define the axes) |
| `scripts/prep/world_frame.py` | mat-blue filter for the plane fit; ruler strip size, FFT band and corner depth range in COLMAP units (they assume about 52.8 mm per unit); hard-coded `stated_width_mm: 120.0` in the report |
| `scripts/prep/split_cams_masks.py` | background color model in `object_mask` and in `mat_rect`, the 20 mm mat-edge margin, `FOOTPRINT_MM` (mirror check), probe count 12 |
| `scripts/eval/evaluate.py` | `OBJ_BOX`; mat-blue filter and the separate footprint 0.107 / 0.055 m for sparse points (line about 292) |
| `scripts/eval/render_3dgs.py`, `scripts/video/make_viewer_cfg.py` | assume the 3DGS ply is the first `*.ply` in `source_dir` and in the COLMAP frame (DevLog-000 notes some Brush exports as y-up: check by sparse-point to Gaussian-center distance first) |
| `scripts/blender/render_orbit.py` | orbit target and close-up shot targets |
| `scripts/final_eval.sh`, `scripts/blender/export_glb.py` | `crt_model_*` output names |
| `config/model/groups.toml`, `config/model/env*.{toml,json}` | part groups, background plane texture spec |
| `scripts/video/camera_path.py`, `viewer/montage.html`, `gen_montage.py` | shot keyframes, group-name regex, card sources |

Checks before any modeling (each one caught a real problem on the CRT):
- Raw pixel order: COLMAP ignores EXIF orientation; load with `cv2.IMREAD_IGNORE_ORIENTATION`. Cameras may differ
  in size (CRT: cam 1 5712x4284, cam 2 4032x3024).
- Plane fit on background-colored sparse points only (the first unconstrained RANSAC plane was wrong).
- Overlay a box of the measured size in 6 views (frame and scale), and the masks in 6 views including low ones.
- Masks: three levels (object / background / unknown where the ray leaves the background plane). Glossy parts
  mirror the background (mirror check inside the footprint), mat-colored parts need sparse-point evidence,
  shadows get labeled object (accept, but know it inflates "missing geometry").
- Holdout split by farthest-point sampling of view directions (14 of 137 here); a 12-view diverse probe subset
  from the train views for the fast loop.

## 4. Phases and agents

Split by part group; each group is also a 3D region (CRT: case, crt, pcb, wires). Up to 4 part agents plus one
audit slot (5 concurrent max); Blender at most 2 concurrent through `scripts/bslot.sh`.

| Phase | Goal | Budget per agent | Coordinator duties |
|---|---|---|---|
| 1A rough | every visible part present within 2-3 mm, simple materials | about 8 iterations | fix shared tools from agent reports the same hour; relay cross-group findings |
| 1B detail | details, labels as photo textures, photo-sampled colors | about 15 | calibrate the render rig; combined snapshot and photometric compare |
| 2 appearance | photo textures on every visible surface | about 10-12 | error budget by group; environment (background plane texture) |
| 3 refine | worst parts by error budget, re-bakes after occluders change | about 10 | sequence re-bakes (wires before board); milestone eval; videos |

Move an agent to the next phase as soon as it reports; do not wait for the slowest. Resume finished agents with
SendMessage (context kept); a message to a running agent is queued and may arrive after it stops: check file
activity and resend.

After each wave: `scripts/final_eval.sh <tag>` (holdout and probe metrics, compare with 3DGS, error budget,
.blend, .glb, orbit renders). After significant work: a fresh-context audit agent of the measurement chain
(it found truncated per-part statistics and holdout leakage on the CRT).

## 5. Evaluation: what each signal is good for

| Signal | Use | Caveat |
|---|---|---|
| Sparse points vs surface (per part, untruncated median + share beyond 4 mm) | best 3D geometry signal | thin parts and mat relief need filtering |
| Edge chamfer (model edges to photo edges, px) | outlines and creases | texture edges in the photo add noise; 2-4 px is typical |
| Silhouette IoU (known region, 2 px band) | gross shape, missing parts | masks: reflections, shadows, mat-colored parts |
| Orphan clusters (points far from any surface) | missing geometry with 3D location | mat relief and background clutter show up too |
| Regions sheet (top tiles per agent) | where to look | the only images agents should read per iteration |
| PSNR/SSIM after per-view color fit; blur-4 PSNR | appearance; blur separates color from sub-pixel alignment | glare dominates on glossy parts |
| `error_budget.py` (share of squared error by part and group) | where effort pays | shares redistribute as other groups improve; compare absolute SSE |

Rules: tune on probe or train views only; holdout views are for milestones (pick.py, texture_from_photos.py,
bake_id_owned.py, fit_params.py and the wires tools refuse them; render_views/evaluate accept them for milestones).
Compare per part, not totals, while several agents edit at once.

## 6. Proven techniques (with measured effect)

- Measure first, fit second: picks on labeled-grid crops, `pick.py tri` (epipolar NCC with patches warped by the
  local affine of a fronto-parallel plane: inliers on top-down views 3-4 -> 8-11), `tri2` for silhouettes,
  `rayplane.py` / `pick.py ray` for points on known planes, `overlay.py` to check a hypothesis in 2-3 views.
  `fit_params.py` repeats to about 1 mm near the truth but lands in local minima from far starts (94 vs 110 mm).
- Sparse-point clusters by color class give component heights and footprints fast (pcb agent).
- Orthophotos: resample a near top-down view onto a plane z = h to read xy of features at that height.
- Wires: 2D trace in a top-down view, per-point depth along the rays by Viterbi over color-mask distance in 5-6
  side views, then a corridor-guarded 3D fit (ROI 8 px, max move 4 mm); see DevLog-002-wires "How to route".
- Textures: per-part UV atlases (box faces, cylinder side and top), each texel = median of its best K train views
  where the texel faces the camera and the ID pass says the part owns the pixel. K = 5 beat 3 (+0.05 to +0.11 dB:
  case and pcb devlogs).
  Model-exact texel grids (bake on the real mesh, not an idealized cylinder) removed parallax ghosting.
  Labels from a single sharp view can beat a multi-view median when registration is imperfect.
- Glossy surfaces: bake from a darker percentile (25th over 8 views) to drop view-dependent highlights (+0.2 dB on
  the screen box; DevLog-002-case).
- Background plane (mat): bake texels excluding only pixels the model covers (not the photo object mask), so cast
  shadows are baked in ("no model" error share 7.6 -> 6.0 percent).
- Render rig calibrated to the photos: uniform white world of strength 1, no lights, photo-textured materials
  without added specular (Specular IOR Level 0), matte photo-sampled materials at 0.2-0.3; EEVEE screen-space ray
  tracing on (+0.47 dB holdout: 13.30 -> 13.77; DevLog-001).
- Last-known-good copy per group in `build_all.py` (TOML parsing inside the try): one agent's half-saved edit
  never drops a group from the others' renders.

## 7. Negative results (do not repeat without a new idea)

| Tried | Result |
|---|---|
| Environment map from background pixels | upper hemisphere 3.5 percent seen; lower = nearby mat; unusable |
| Dim world for glossy rays only (Light Path) | EEVEE darkens all world light |
| Black plastic specular sweep 0.02 / 0.15 / 0.5 | 13.89 / 13.89 / 13.83 dB: flat |
| Opaque textured quad over the window | 12.65 dB vs 13.89 without it (case_022): card parallax plus baked lamp reflection |
| Refracting glass with ray tracing | 12.6 dB vs 14.3 for the textured shell |
| Color filters while baking tape/copper | dropped real highlights and folds (-0.8 dB) |
| Ring lamp as a highlight-only light | automatic lamp detection failed; unfinished (idea still open) |

## 8. Pitfalls seen

- Workbench ID colors can come back one 8-bit level off: decode to the nearest palette color.
- Label quads: corner order TL, TR, BR, BL; the normal is (down) x (right). A wrong winding buried a label.
- Holdout leakage happened at least four times (a wall height, wire depth solves, a test texture, early tri
  matches) before the tools refused holdout views.
- Truncated statistics flatter parts (white_pair 0.43 mm reported vs 2.66 mm real).
- Agents delete their old runs; anything that must survive (montage cards, figures) needs its own copy.
- zsh does not word-split `$var` (use `${=var}`); argparse needs `--` before negative numbers.
- Blender's Python has no PyYAML: model parameters are TOML (tomllib).
- Never hand-build an established workload (3DGS rendering, video capture): use the proper viewer or library and
  write only glue (see `VIDEO_WORKFLOW.md`).

## 9. Timing and cost on the CRT (M4 Pro Mac)

- Iteration (build + 12-view render + eval): about 12 s. Fit: about 0.5 s per evaluation (12 views at 50 percent).
- Milestone bundle (`final_eval.sh`): about 70 s (v3 snapshot file times). Demo videos: comparison capture 1:21-1:24,
  montage capture 52-65 s, combine about 30 s (DevLog-001).
- Agent phases (from the session's task records, logged in DevLog-001): 45 min to about 5 h of wall time each
  (case Phase 1B 4.5 h, wires Phase 1B 5.2 h); 150k-430k tokens per agent over the whole run.

## 10. Brief template for part agents

```text
You are the <GROUP> modeling agent in <project path>. Read AGENT_GUIDE.md first, then PLAYBOOK.md sections 5-8.
You own scripts/model/<group>/, config/model/<group>.toml and DevLog/parts/DevLog-002-<group>.md.
Scope: <parts, with world-mm hints and good train views>.
Phase <X> only: <goal>. Budget: about <N> iterations. Train/probe views only; holdout never.
Final report (under 250 words): parts built, key dimensions and how measured, before/after per part
(error_budget.py and parts.json on probe runs), sheet paths, tool problems, next steps. No emoji. No git.
```
