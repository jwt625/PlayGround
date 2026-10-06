# DevLog-001: CRT display - plan, toolkit and workflow exploration

| Field | Value |
|---|---|
| Date | 2026-10-04 |
| Status | Phase 0 done (data prep, toolkit, workflow pilot); Phase 1 part agents launched for a rough build |
| Scope | Explicit, editable Blender model of `~/Documents/3DGS/20251004_CRT_display` (137 photos, COLMAP sparse/0), built by agents from the photos, steered by code-driven evaluation that points visual inspection at the worst regions |
| Authors | Claude (coordinator), at Wentao Jiang's request |

## 1. Requirements (from DevLog-000 6.1)

- Subject: the CRT display set. Scale: the case's shorter edge (width) is about 12 cm.
- Limits: up to 5 subagents at once, up to 2 Blender instances at once (`scripts/bslot.sh`, 2 slots). No remote GPU.
- Labels are first-class: modeled as proper textures, judged as important as the structure.
- Detail floor: everything resolvable at about 0.3 mm (components, connectors, wires as curves).
- 3DGS baseline: the existing Brush ply as-is.
- Asset conventions from the film's `ASSET_SPEC.md` (1 BU = 1 m, real size, Z up, -Y front, `ASSET_`/`ROOT_` naming), but all code lives in this folder (no cross-project imports or shared files).
- Explore and measure the workflow itself, especially evaluation and feedback; balance speed against accuracy, and use code to pick the regions the built-in visual inspection should look at.

## 2. Facts about the data (verified 2026-10-04)

- 137 images, all registered. Two SIMPLE_RADIAL cameras: cam 1 is 5712x4284, f 4534.8, k1 -0.0296 (129 images); cam 2 is 4032x3024, f 3202.9, k1 -0.0323 (8 images). The principal point sits at the image center.
- COLMAP used raw pixel order, ignoring EXIF orientation (IMG_1505 has EXIF orientation 6, yet camera 1 is 5712x4284). Every image load in this project must ignore EXIF orientation: `cv2.IMREAD_IGNORE_ORIENTATION`, and no `ImageOps.exif_transpose`.
- Scene: an open black plastic case. One half holds the bezel and screen window; the other holds the electronics: CRT neck with socket board, yellow taped deflection yoke with a printed label, a boxed transformer with a red WARNING label, a PCB with electrolytics, chokes and connectors, and wire looms. Outside the case are a small brown PCB on wires and white two-conductor wires. Everything sits on a blue silicone repair mat with molded features.
- Disk: 12 GiB free at start. Data is stored at reduced resolution; full resolution is read from the originals only for label texture crops.

## 3. Plan

### Phase 0: data prep (coordinator)
- [ ] `scripts/prep/colmap_io.py`: read the binary COLMAP model.
- [ ] `scripts/prep/undistort.py`: undistort to a pinhole model at scale s (default 1/4, 1428x1071; also 1/2 for inspection crops), with the same f and principal point, written to `data/undistorted_s{N}/`.
- [ ] `scripts/prep/world_frame.py`: RANSAC fit of the mat plane to the sparse points to get Z up; origin at the case footprint; scale from the case width = 120 mm (triangulated case corners); written to `config/world.yaml` as the similarity transform from COLMAP to world.
- [ ] Cameras as JSON for Blender (`data/cameras.json`: K, R, t in world frame, sizes per scale); holdout split (about 10 percent, spread over viewing directions).
- [ ] Masks: object vs mat by color keying in an ROI around the projected case box; spot-check.

### Phase 0b: toolkit (coordinator, then tested)
- [ ] `scripts/blender/render_views.py`: load a model .blend (or build script), set up cameras, render RGB plus a part-ID pass, depth and alpha for a view subset at scale s. Lighting is a simple studio rig; color comparison is lighting-normalized in eval.
- [ ] `scripts/tools/triangulate.py`: pixel picks in two or more views to a 3D world point (DLT plus reprojection error). Epipolar-assisted matching (NCC along the epipolar line) so one pick in one view is enough. This is the main measuring tool for modelers.
- [ ] `scripts/tools/label_rectify.py`: label quad (3D corners) or a cylinder patch to a rectified texture from the best views (fronto-parallel, sharp, not occluded), using full-resolution originals.
- [ ] `scripts/eval/evaluate.py`: per view, silhouette IoU and an XOR map; edge chamfer (photo vs render edges, px); local-contrast-normalized color and structure error (per-view affine color fit, SSIM); sparse points vs model surface (mm, per part). Aggregated per part via the part-ID pass.
- [ ] `scripts/eval/regions.py`: rank error tiles across views and emit a few inspection crops (photo | render | edge overlay | error heat) so visual inspection only looks where code says the error is.

### Phase 0c: workflow exploration (coordinator)
- [ ] Build the case (simplest part) myself with the toolkit; time each step.
- [ ] Sensitivity test: perturb the case (translate 1/2/5 mm, scale 1/2 percent, rotate 0.5/1 deg) and measure which metrics respond and how strongly. That sets which metrics drive feedback and the noise floor.
- [ ] Decide the loop: views per iteration, resolution, renderer and samples, crop budget per round.

### Phase 1: part agents (up to 4 modelers plus coordinator eval)
Part groups (tentative): case + bezel + screen; CRT neck + yoke + socket board (with the yoke label); main PCB + components + transformer (with the WARNING label); wires and the external small PCB. Each agent owns `scripts/model/<group>/` plus its devlog; one assembly script composes the groups. The coordinator runs the global eval and routes per-part feedback.

### Phase 2: fresh-context audit and report

## 4. Findings (Phase 0, 2026-10-04)

- Scale: the mat's molded mm ruler, measured by FFT of the rectified tick profile in 75 views: 1 mm = 0.018948 COLMAP
  units (per-view std 0.95 percent) -> 52.776 mm per unit. Cross-checks: blue mat points fit a 451.6 x 292.9 mm
  rectangle vs the printed "450mmX300mm"; triangulated case rim corners are 90.15 deg apart.
- The case's short edge measures 97.5 mm at the rim (fit: 98-99 mm), not "about 12 cm" as stated. The ruler scale is
  used; this needs Wentao's confirmation.
- Masks: object = not mat-colored, only where the pixel ray meets the mat inside its rectangle; the rest is "unknown"
  (ignored). Mat color needs hue + chroma + brightness rules (glossy black case reflects the mat; shadowed mat is
  dark). Residual errors: thin mat-lip lines, dark fringes in deep shadow.
- Speeds (M-series Mac, 1/4 scale, 12 probe views): build + render ID and RGB (EEVEE 16 spp) + point distances 8-12 s;
  evaluation 3 s; one full iteration about 12 s. In-Blender fit: about 0.5 s per evaluation (two Workbench renders
  x 12 views at 50 percent).
- Fit reliability (pilot case box): length/width/heights repeat within about 1 mm from two starts; the screen-section
  length is multimodal (110.75 vs 94.0 mm from different starts). Rule: measure key positions by picks and
  triangulation, then fit locally with small steps.
- Region finder: color residual dominates on untextured models, so `--mode geom` (silhouette + edges only) is the
  default while blocking out; `--parts` restricts inspection crops to one agent's parts.
- Wentao (mid-run): "dont get lost in the details. Start with a rough build, and then keep iterating and add better
  textures and details"; "try parallelism ... two agents ... two 3d regions ... or by parts". Decision: four part
  agents (case, crt, pcb, wires), which also partition the object into 3D regions; rough block-out first (Phase 1A),
  then detail and texture waves (1B).

## 5. Toolkit (as built)

| Tool | Purpose |
|---|---|
| `scripts/prep/undistort.py` | 1/4 and 1/2 scale pinhole images (data/undistorted_s4, _s2) |
| `scripts/prep/world_frame.py` | mat plane, ruler scale, case-based axes -> `config/world.yaml` |
| `scripts/prep/split_cams_masks.py` | holdout (14) / train (123) / probe (12) split, Blender camera JSON, 3-level masks |
| `scripts/prep/edges.py` | photo edge distance maps for the fit loss |
| `scripts/blender/render_views.py` | ID (Workbench, exact) + RGB (EEVEE) + sparse-point distances (BVH) per view |
| `scripts/eval/evaluate.py` | silhouette, edges, color/SSIM, 3D points, orphans, per-part table, region crops |
| `scripts/blender/fit_params.py` | in-Blender pattern-search fit of TOML parameters |
| `scripts/tools/pick.py` | crops with pixel grids, triangulation (epipolar NCC + RANSAC), rays, projections |
| `scripts/tools/texture_from_photos.py` | rectified label/print textures from full-resolution originals |
| `scripts/iterate.sh` | one iteration: build, render, evaluate, print report |
| `scripts/eval/render_3dgs.py` | CPU (numba) EWA renderer of the Brush 3DGS baseline at our views, about 6 s per view at 1/4 scale |
| `scripts/eval/compare_photometric.py` | masked PSNR/SSIM of model vs 3DGS vs photo, raw and after a per-view affine color fit |
| `AGENT_GUIDE.md` | shared brief for part agents |

## 6. Progress log

- 2026-10-04: Plan written. Project skeleton, uv env (numpy, opencv-python-headless, scipy, pillow, pyyaml; Python 3.12) and `scripts/bslot.sh` (2 slots) created.
- 2026-10-04: Phase 0 complete: data prep, world frame (ruler scale), masks, render/eval/fit/pick/texture tools, case pilot. Launching part agents (case, crt, pcb, wires) for the rough build.
- 2026-10-04: Four part agents launched (Phase 1A rough block-out). Texture tool tested on the flyback WARNING label (quad at z 26.9 mm, 29 x 20 mm): clean rectified texture from 3 views; fixed a normal-orientation bug in the tool and in lib.label_quad.
- 2026-10-05: 3DGS baseline: the Brush ply (633,876 Gaussians, SH degree 3) is in the COLMAP frame (median sparse-point to center distance 0.55 mm). Rendered at 14 holdout + 12 probe views. Reference numbers on probe views (masked object region): 3DGS 22.6 dB / SSIM 0.82; pilot case box only: 10.8 dB / 0.26 after color fit.
- 2026-10-05: CRT agent Phase 1A done (10 parts, sparse-point medians 0.15-0.88 mm); PCB agent Phase 1A done (45 parts, most 0.2-1.2 mm). Both moved straight to Phase 1B (details, textures) while case and wires finish 1A.
- 2026-10-05: Shared tool fixes from agent reports: ID decode now nearest palette color within L1 6 (Workbench colors came back 1 LSB off for some objects, which counted whole parts as missing); epipolar matching warps the neighbor patch by the local affine of a fronto-parallel plane (inliers on top-down views 3-4 -> 8-11); masks treat pixels where sparse points (z > 5 mm, err < 2 px) project as object (mat-colored light-blue socket cap).
- 2026-10-05: Environment: group "env" (coordinator) with the mat as a plane textured from the photos (per-texel best-5-view median at 2 px/mm, object pixels excluded, holes inpainted; helper object "_env.mat", excluded from ID/fit/points). Render lighting recalibrated to a uniform white world of strength 1 + weak 3 W key, so photo-textured diffuse surfaces reproduce photo values. Combined snapshot (12 probe views): sparse-point median 0.62 mm, 61 percent within 1 mm; photometric 12.6 dB after color fit (3DGS 22.6 dB).
- 2026-10-05: All four part agents finished Phase 1A; crt and pcb finished 1B (yoke label, flyback WARNING and SY-F100C labels, board-top photo texture; nearly photographic in renders). Case (1B) and wires (1B), crt and pcb (1C) running.
- 2026-10-05: Rough build v1 snapshot (`outputs/snapshots/rough_v1/`: model .blend, holdout compare sheet, eval report). 96 objects. Probe (12): IoU 0.936, sparse-point median 0.48 mm, 70 percent <= 1 mm, 82 percent <= 2 mm. Holdout (14): IoU 0.921, edge 2.92 px. Photometric holdout (object region, color-fit): model 12.6 dB / SSIM 0.36 vs 3DGS 22.1 dB / 0.79. Largest visible photometric error: glossy black case renders gray (reflects the white world); a dim glossy-ray world darkens everything in EEVEE, so the fix goes into the case material (low specular).
- 2026-10-05: Mask mirror check (reflections in the glossy case walls), point threshold z > 2.5 mm, tri2 guard, render rig recalibration (no key light, photo textures without specular), and build_all last-known-good fallback per group (outputs/lkg/).
- 2026-10-05: CRT 1C (crt agent): per-part photo textures with its own texel grid on the exact yoke mesh + ID visibility over 123 train views; raw PSNR inside CRT pixels 11.6 -> 14.5 dB (3DGS 22.5 on the same pixels). Self-texture test (texture and render the same view) gives only 16.2 dB on the yoke: the residual is 1-3 px edge misalignment at high-contrast borders, which PSNR punishes and 3DGS (per-pixel optimized on those exact photos) does not pay.
- 2026-10-05: PCB 1C done (small parts, axial list, scripts/tools/bake_id_owned.py). Whole-model holdout (14 views): 13.3 dB color-fit (3DGS 22.1); alignment-tolerant psnr_fit_blur4 15.1 (3DGS 26.3), so colors/structure also lag, not only edges.
- 2026-10-05: Error budget (scripts/eval/error_budget.py, holdout, blurred, after color fit): case 50 percent (window 16.4, tray 10.0, screen_box 9.7, bracket 4.5), pcb 18, crt 11.4, wires 11.0, missing geometry 9.6. The window/glossy black/metal plate error is view-dependent reflection of the room.
- 2026-10-05: Tried a room environment map from background pixels (scripts/prep/env_map.py): the cameras look down, so only 3.5 percent of the upper hemisphere is ever seen and the lower hemisphere is the nearby mat/table (infinite-distance assumption fails). Not usable; the window's reflections (ceiling, ring lamp) are not recoverable from this capture. A few extra photos of the ceiling/lights or a mirror-ball probe would be needed. Not used in renders.
- 2026-10-05: Fresh-context audit of the measurement chain (DevLog-003): scale confirmed (ruler numerals +0.3 to +0.7 percent, mat outer edges 460.5 x 302 vs printed 450 x 300; a 12 cm case is ruled out), world frame/undistortion confirmed (0.3-0.4 px residuals incl. camera 2 and EXIF-rotated views; model silhouettes best shift (0, 0) px), 3DGS renderer confirmed (0.1-0.2 px). Problems fixed: per-part point medians were truncated at 4 mm (now untruncated + frac >4 mm); holdout leakage (case used IMG_1595; texture auto views could pick holdout) -> pick.py and texture tool refuse holdout views, mat texture rebuilt from 123 train views, agents told; eval now reports IoU without the band and the unscored model fraction. Context: a flat mean color scores 10.6 dB on the same region. Open: mask failure types (wall reflections partly, light-blue cap, mat lip lines), edge recall metric, 3DGS trained on holdout views (kept as is per Wentao).
- 2026-10-05: Case 1B done (IoU 0.873 -> 0.940, sparse median 0.62 -> 0.44 mm; rounded walls, ramps, lugs, slot, clips, bracket holes; black plastic Specular IOR Level 0.02). Case re-measuring the IMG_1595-based wall height on train views.
- 2026-10-05: Phase 1 complete for all groups (case 1B + holdout fix, crt 1D, pcb 1C + re-bake, wires 1B incl. internal cables; holdout leakage found by the audit was removed by each agent). Milestone v2 (`scripts/final_eval.sh v2` -> outputs/snapshots/v2/): holdout IoU 0.927 (0.898 without band), edges 2.88 px, sparse points median 0.38 mm, 78.9 percent <= 1 mm, 91.7 percent <= 2 mm; photometric 13.3 dB color-fit (3DGS 22.1), blur4 15.1 (26.3). Error budget unchanged in shape: case 50.7 percent (window 16.4), pcb 18.1, missing geometry 10.9, wires 10.8, crt 9.5.
- 2026-10-05: EEVEE screen-space ray tracing (glossy walls reflect the mat plane): holdout 13.30 -> 13.77 dB, blur4 15.12 -> 15.65; now default (EVAL_RAYTRACE=0 to disable). Black plastic specular sweep (0.02 / 0.15 / 0.5, probe views, ray tracing on): 13.89 / 13.89 / 13.83 dB, insensitive; kept 0.02.
- 2026-10-05: New tools: scripts/final_eval.sh (milestone bundle), scripts/blender/render_orbit.py (new viewpoints), scripts/eval/error_budget.py. README written.
- Next (Phase 2, appearance + completeness): photo-texture the large surfaces (case walls, rim, interior, window appearance, heat-sink plate) from train views with ID-owned per-texel medians; hunt the 10.9 percent missing-geometry pixels (remaining red loom wires, small parts).
- 2026-10-05: Phase 2 (appearance + completeness). Case: 35 photo-textured panels (0.25 mm texels, ID-owned median of best 5 train views, scripts/model/case/bake_case.py), window inner walls segmented and textured; probe 13.89 -> 14.58 dB; an opaque textured window top failed (12.65 dB: card parallax + baked lamp reflections). PCB: heat-sink plate 14.1 -> 16.7 dB, bake_id_owned.py ranks train views itself and refuses holdout; no missing board geometry found. Wires: remaining "missing geometry" is shadows and blur halos (mask), not unmodeled wires; added red_e, re-routed white singles; board re-baked.
- 2026-10-05: Milestone v3 (outputs/snapshots/v3/), holdout 14 views: photometric 14.09 dB color-fit (v2 13.30; 3DGS 22.1), blur4 16.19 (v2 15.12; 3DGS 26.3), SSIM 0.41; geometry unchanged (IoU 0.927, edges 2.86 px, points median 0.39 mm, 91.7 percent <= 2 mm). Error budget: case 53.7 (window 19.2), wires 12.3, pcb 12.3, no-model 11.6 (mostly shadow pixels in the masks), crt 10.2.
- 2026-10-05: Wentao asked for a viewer-friendly output like the 3DGS ply. Added scripts/blender/export_glb.py (wires converted to meshes, textures embedded, Y-up meters) and hooked it into final_eval.sh. v3 exports: crt_model.glb 4.7 MB, crt_model_with_mat.glb 5.6 MB; verified by re-import in a fresh Blender (129 meshes, 58 images, bbox 239 x 177 x 50 mm) and a render.
- 2026-10-05: README rewritten with methodology, results table (v1-v3 vs 3DGS), lessons and a reproduce recipe. Agent texture specs from scratch copied to config/textures/pcb/ (hand-written specs; case specs are regenerated by bake_case.py). Project .gitignore: no images, arrays, .blend, .glb, data/ or outputs/. Committed scripts, configs and docs.
- 2026-10-05: Videos (scripts/video/): comparison and build montage recorded from the real glb and ply in a three 0.181 + Spark 0.1.10 page (same stack as the 3dgs-viewer repo) with headless Chrome; comparison recorded in real time via MediaRecorder (1:24 for 60 s), montage at 5 images/s. Lesson recorded in memory: never hand-build established workloads (the numba 3DGS renderer and per-frame CDP screenshots were far too slow). Final cut outputs/video/crt_build_and_compare_20261005.mp4: 30 s montage + comparison at 2x camera speed, 1080x1620, soundtrack section picked by beat tracking and loop similarity (scripts/video/pick_loop.py), tempo 1.028x with pitch kept.
- 2026-10-05: Phase 3 started (all four agents; Wentao: "many of the PCB components also lack proper textures" -> pcb agent: texture every visible component). Coordinator: mat re-bake that excludes only model-owned pixels (ID pass of the 123 train views, outputs/runs/coord_idtrain) instead of the photo object mask, so cast shadows next to the case are baked in; probe photometric 14.57 -> 14.69 dB (agents' edits also in flight), no-model share 7.6 -> 6.0 percent.
- 2026-10-05: Phase 3 results (probe views, error_budget.py): pcb: all 47 visible components photo-textured via per-part UV atlases (scripts/model/pcb/uvparts.py, bake_parts.py; ID-owned per-texel median of 5 train views), pcb share 15.6 -> 11.5 percent, worst parts +4 to +10 dB (trim_6 9.9 -> 19.5, capB_i 9.9 -> 19.5, conn_b 11.0 -> 17.5); case: card re-measured (17.4 -> 19.0 dB), screws textured (9.1 -> 16.4), refracting glass scored below the textured shell (12.6 vs 14.3 dB) so the shell stays; crt: yoke 18.1 -> 19.2 dB, self-texture test shows the remaining error is view-dependent shine, not geometry; wires: paths final, white_single_a points median 1.33 -> 0.34 mm, pots raised to 8 mm, missing-geometry error 1774 -> 1079 (1e6 units). Whole model on probe: 14.57 (v3) -> 14.89 dB color-fit (holdout milestone pending).
- 2026-10-05: build_all: TOML loading moved inside the try so a half-saved TOML also falls back to the last-known-good copy. Ring-lamp highlight-only light (to reproduce window reflections) tried briefly: automatic lamp detection failed; paused.
- 2026-10-05: Video refresh (v4 glb): montage gained build steps (block-out by group, materials as mean texture colors, labels, photo textures by group, final; the case keeps its final materials because its textured panels carry the see-through look of the open window and tray) and an append-only card registry (outputs/video/montage/cards.json: cards are kept forever and new ones appended; 16 cards). Combined cut re-made with the same soundtrack section.
- 2026-10-05: Docs for the next batch of fresh-context agents: PLAYBOOK.md (coordinator), AGENT_GUIDE.md augmented (render rig, texturing recipe, reports, pitfalls), VIDEO_WORKFLOW.md (demo video specs and pipeline), README links and next-capture recommendation.
- 2026-10-05: Session task records (for PLAYBOOK section 9): agent wall times per resume 45 min to 5.2 h (case Phase 1B 4.5 h, wires Phase 1B 5.2 h), 150k-430k tokens per agent. Video timings: comparison capture 1:21-1:24, montage capture 52-65 s, combine about 30 s.
- 2026-10-05: Fresh-context review of the next-batch docs: 15 fixes applied (missing points export -> scripts/prep/export_points.py; viewer cfg and links -> scripts/video/make_viewer_cfg.py; Phase 0 order with the two-pass mat bake; capture-specific constants table completed; fit_params refuses holdout views; capture_viewer converts to tv range; corrected numbers with sources; stale self-texture claim in README; LKG wording in AGENT_GUIDE).
- 2026-10-05: Milestone v4 (`scripts/final_eval.sh v4`, all groups built from current code, no LKG fallback; 119 scored objects, 69 textures): holdout 14 views: IoU 0.930 (0.901 without band), edges 2.80 px, sparse points median 0.39 mm, 78.4 percent <= 1 mm, 92.0 percent <= 2 mm; photometric color-fit 14.58 dB (v3 14.09; 3DGS 22.11), blur4 16.77 (v3 16.19; 3DGS 26.33), SSIM fit 0.424 (3DGS 0.788). Probe 12 views: 15.02 dB, blur4 17.13 (3DGS 22.87 / 27.02). Holdout error budget: case 57.7 (window 23.6, tray 10.4, screen_box 7.3, card 6.4, window_edge 4.7), wires 12.9, crt 10.3, pcb 9.5, no model 9.5. README results table and PLAYBOOK section 1 updated.
