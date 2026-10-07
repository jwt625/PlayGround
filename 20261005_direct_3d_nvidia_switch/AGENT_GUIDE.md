# Agent guide: direct 3D modeling of the CPO switch tray and its package

Part agents read this first, then sections 5-8 of the CRT run's playbook
(`../20261004_direct_3d_from_images/PLAYBOOK.md`: what each metric is good for, proven techniques with measured
effect, negative results, pitfalls). That project is read-only reference: never edit or import from it.

Goal: an explicit, editable Blender model of an NVIDIA SN6810-LD "Spectrum-X CPO Switch Tray" (2RU, liquid
cooled, top cover removed) and the Spectrum-6 CPO package on a post stand in front of it, built from 43 usable
iPhone photos (`source/images/`, read-only) and checked by rendering at the solved cameras. Labels and printed
text are as important as structure (image textures from the photos). Rough first, then details.

## Capture facts that differ from the CRT run
- Own COLMAP model (`data/colmap/sparse/0`): 44/44 registered, 28,775 points, 1.31 px mean reprojection error.
  IMG_5739 has a wrong pose and is excluded everywhere (`data/split.json` "excluded"; loaders skip it).
- Outdoors in direct sun: hard, dappled tree shadows that moved between three sessions (S1 12:31-12:37,
  S2 12:47, S3 13:36; see `config/scene.yaml`). Photo textures will carry shadows; prefer views from one
  session per surface, and expect per-session differences in color metrics.
- No ruler: scale from the published 438 mm body width (screw holes on both outer side walls), cross-checked
  against the 776 mm body length; audit: scale factor 1.000 (0.995-1.002). Report any dimension that disagrees.
- Masks: SAM 2.1 per object (`data/sam_s2/<tray|package|sign>/`), assembled into 3-level masks
  (`data/masks_s4/`). Inside the projected tray-body and package hulls, background is "unknown" (128), so
  silhouette scores judge only the outer outline (this hides excess geometry inside the outline: 3-15 percent of
  the hull area, audit); the acrylic sign, the package stand (black on black) and SAM spill outside the tray's
  projected boxes are unknown too.
  Inner geometry is judged by edges, sparse points and (later) color.

## World frame (meters in Blender, mm in configs and tools) = tray frame (with two known offsets)
- X: tray width, 0 midway between the outer side-wall faces (walls at X = -219 and +219; audit: 437.8 mm).
- Y: toward the rear. Y = 0 is NOT the front panel: it landed on the acrylic sign's printed text (Y -10..+3).
  The front port plate face is at Y = 23.5, the rear lip at Y = 802.5 (port plate to rear lip 779 +- 2.5 mm vs
  776 published). Coolant fittings extend beyond the rear lip.
- Z: out of the open top; Z = 0 at the crossbar top face at X = 0. The world frame is rolled about Y: the tray's
  +X side is higher, world z = tray z + 0.0148 x (0.85 deg, +-0.1). For tray-aligned rigid parts, build level and
  call `lib.apply_frame(objs, "tray")` (config/frames.json "tray"); parts placed directly from measurements in
  world coordinates need no correction. Do not add private roll/tilt corrections.
- Gravity is not Z: the tray stands front panel down, rear up, leaning back; the "scene" frame in
  `config/frames.json` has Z up. Package parts are built in the "package" frame (Z out of the package top, origin
  at the substrate-top center) and moved with `lib.apply_frame(objs, "package")`.
- Bays (measured by the agents, current frame): front bay up to the crossbar (crossbar top Y 331-371 at Z 0),
  middle bay to the divider (Y 578), rear bay to the rear lip (Y 802.5).
- Scale (audit DevLog-003): factor 1.000 (0.995-1.002), from the 438 mm published width; checked by the length
  and the package (108 mm measured; the 110 mm analyst figure is rounded).

## Files you own (one agent per group)
- `scripts/model/<group>/build.py` with `build(P, coll)`, and `config/model/<group>.toml` (all dimensions here,
  mm). Helpers: `scripts/model/lib.py` (box, open_box, cylinder, tube_path, boolean, bevel, place, mat_pbr,
  mat_image, label_quad, frame_matrix, apply_frame). Object names `<group>.<part>`; one object per part you want
  feedback on. The coordinator left a stub in your files: replace it.
- Your devlog `DevLog/parts/DevLog-002-<group>.md` (TODO checklist + timestamped progress + measurements).
- Never edit other groups' files, shared scripts, `data/`, `config/world.yaml`, `config/frames.json`,
  `config/picks.yaml` or the mask configs. Need a shared change? Write it in your devlog under "Requests to
  coordinator" and work around it. No git. No emoji anywhere.
- Keep your build always runnable: a broken build or TOML makes `build_all.py` fall back to your last-known-good
  copy (`outputs/lkg/<group>/`).

## Loop (about 10 s per iteration for 8 probe views)
1. `scripts/iterate.sh <group>_NNN <views> <group>. geom` builds all groups, renders, evaluates, prints
   `outputs/runs/<run>/eval/report.md`. Views: `probe` (8 diverse), `train` (38), a comma list of image stems,
   or `holdout` (5; final checks only). Use `full` instead of `geom` once you have materials/textures.
2. Look at `outputs/runs/<run>/eval/regions.png` (worst regions of YOUR parts: photo | render | edges green
   photo / red model / yellow both | error heat: magenta excess geometry, cyan missing geometry) and
   `views.png`. Read only these sheets and targeted crops, not whole photos, unless you need context.
3. `orphans.json`: clusters of sparse points far from any model surface (world mm) = missing geometry.
4. `parts.json`: fp_frac (excess), fn_px (missing nearby), edge_mean (px), pts_median_mm (best 3D signal).

## Holdout rule
Holdout views (IMG_5715, 5728, 5737, 5747, 5750) are for evaluation only: never measure, fit, depth-solve,
texture or inspect crops with them. pick.py, overlay.py, rayplane.py, texture_from_photos.py, bake_id_owned.py
and fit_params.py refuse them. Accepted limitation: the COLMAP poses and points were solved jointly with all
views (as in the CRT run).
Also never open holdout photos, holdout crops, holdout runs (`outputs/runs/*holdout*`) or the holdout files of
milestone snapshots (`outputs/snapshots/*/eval_holdout*`, `compare_holdout`, `error_budget_holdout*`), and never
steer work by holdout numbers: diagnose on probe/train views. render_views.py refuses holdout views (only
`scripts/final_eval.sh` renders them). A Wave 4 agent read holdout crops and its changes were rolled back (audit
DevLog-005).

## Measuring (do this instead of guessing)
- `uv run python scripts/tools/pick.py views X Y Z` best views of a world point and its pixel there.
- `... pick.py crop VIEW U V --half 150` labeled grid crop (1/2-scale pixel coords) -> Read it -> read coords.
- `... pick.py tri VIEW U V --depth 200,1500` one pick -> epipolar NCC match in neighbor views -> 3D (mm).
  Check n_inliers >= 3. It fails often on smooth metal and dark plastic here (few, oblique views); then use
- `... pick.py tri2 V1 U1 V1 V2 U2 V2 ...` manual correspondences read from crops of 2-3 views (find the same
  screw hole, corner or label feature; patterns of holes disambiguate). Expect 1-4 px residuals.
- `... pick.py ray VIEW U V --z Z` pixel ray hits plane z = Z; `scripts/tools/rayplane.py VIEW x|y|z VALUE U V`
  any axis plane (wall faces X = +-219, front port plate Y = 23.5, crossbar top Z = 0 at X = 0; mind the 0.85 deg roll).
- `uv run python scripts/tools/overlay.py VIEW '[{"box":[x0,x1,y0,y1,z0,z1]}]'` draws world-mm hypotheses on the
  photo with a pixel grid: check a dimension in 2-3 views before editing the TOML.
- Sparse points: `data/points_world.npy` (meters), `points_rgb.npy`, `points_err.npy`,
  `points_views.json`. For measuring use only `data/points_train_mask.npy` (points with >= 2 train-view
  observations; `tools.geom.train_point_mask()`): 9.8 percent of tray points exist only because COLMAP also saw
  the holdout views (audit). The full cloud is for evaluation. Clusters by color class and position give heights
  and footprints fast.
- Photos: `data/undistorted_s2/<stem>.jpg` (2856x2142) and `_s4` (1428x1071), raw pixel order (EXIF ignored on
  purpose, as COLMAP did). Most photos are portrait (EXIF 6) and IMG_5735-5740 are upside down (EXIF 3) in raw
  order: crops look rotated; never apply EXIF rotation.

## Fitting continuous parameters
`scripts/bslot.sh -b --factory-startup --python scripts/blender/fit_params.py -- --group <g>
 --params a.b:step,c.d:step --targets <g>.part_prefix --max-evals 80 --out outputs/fits/<g>_<name>`
Pattern search on edge + excess-geometry loss. Reliable within a few mm of the truth; measure first, fit second.
Copy accepted values from best.json into your TOML yourself.

## Textures (labels, printed parts, photo textures)
Same tools and recipe as the CRT run (its AGENT_GUIDE "Textures" and "Texturing recipe"):
`scripts/tools/texture_from_photos.py spec.json` (quad or cylinder patch -> `assets/textures/<name>.png` from the
full-resolution originals), `scripts/tools/bake_id_owned.py` (ID-owned per-texel median of the best train views).
Here, also restrict to views of one session where shadows differ (spec field or view list), and note it.

## Render rig and materials (calibrated in the CRT run; do not change per part)
- RGB renders: uniform white world of strength 1, no lights, EEVEE screen-space ray tracing on. A diffuse
  surface renders at about its base color, so photo textures reproduce photo values.
- Photo-textured materials: `lib.mat_image` (Specular IOR Level 0). Photo-sampled flat materials: base color =
  median of a well-lit photo patch, Specular IOR Level 0.2-0.3; metal stays metallic.

## Reports and devlogs
- Final report (under 250 words): what changed, how it was measured, before/after per part (`parts.json` on
  probe runs, or your own view list), sheet paths, tool problems, next steps.
- Compare per part, not totals: other agents edit at the same time.
- Runs you delete may be referenced elsewhere: copy anything meant to last out of `outputs/runs`.

## Pitfalls
- zsh: `$var` does not word-split (use `${=var}`); argparse needs `--` before negative numbers.
- Blender's Python has no PyYAML: parameters are TOML, frames are JSON.
- Label quads: corners TL, TR, BR, BL; normal = (down) x (right).
- Repeated parts (connector blocks, fingers, pins, ports): measure the pitch once, instance in code.

## Machine rules
- Blender only through `scripts/bslot.sh` (at most 2 Blender processes machine-wide; it queues). Never kill
  other Blender processes. Prefer `probe` or short view lists; `train` (38 views) only for occasional checks.
- Disk: keep run folders few (reuse a run name or delete your own old runs), no full-resolution renders.
- Scratch files: `outputs/scratch/<group>/`. Delete your own scratch when done.
