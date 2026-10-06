# Agent guide: direct 3D modeling of the CRT capture

Part agents read this first. The coordinator's playbook is `PLAYBOOK.md` (sections 5-8: what each metric is good
for, proven techniques with measured effect, negative results, pitfalls). For a new capture the coordinator
rewrites the CRT-specific parts: Goal, World frame, the view counts (14 holdout / 123 train / 12 probe), image sizes
(1/2 scale: cam 1 2856x2142, cam 2 2016x1512) and the disk figure; the rest applies as is.

Goal: an explicit, editable Blender model of the CRT display unit in `~/Documents/3DGS/20251004_CRT_display`
(137 photos), built from the photos and checked by rendering at the COLMAP cameras. Labels and printed text
are as important as structure (image textures from the photos). Work rough first, then iterate on details.

## World frame (meters in Blender, mm in configs and tools)
- Z up, z = 0 on the mat surface. Origin at the case footprint center. +X toward the electronics end, -X the
  screen end, -Y the near long side. Scale from the mat's mm ruler (checked against its printed 450 x 300 mm).
- Case: about 200.7 x 99 mm outer; screen section (-X) rim about 36 mm high, electronics tray (+X) about 28 mm.

## Files you own (one agent per group)
- `scripts/model/<group>/build.py` with `build(P, coll)`, and `config/model/<group>.toml` (all dimensions here,
  mm). Helpers: `scripts/model/lib.py` (box, open_box, cylinder, tube_path for wires, boolean, bevel, place,
  mat_pbr, mat_image, label_quad). Object names `<group>.<part>`; one object per part you want feedback on.
- Your devlog `DevLog/parts/DevLog-002-<group>.md` (TODO checklist + timestamped progress + measurements).
- Never edit other groups' files, shared scripts, `data/`, or `config/world*.yaml`. Need a shared change? Write
  it in your devlog under "Requests to coordinator" and work around it. No git. No emoji anywhere.
- Keep your build always runnable: a broken build or TOML makes `build_all.py` fall back to your last-known-good
  copy (`outputs/lkg/<group>/`), so others keep your occluders but your new work is invisible until it builds.

## Loop (about 12 s per iteration for 12 views)
1. `scripts/iterate.sh <group>_NNN probe <group>. geom` builds all groups, renders the 12 probe views, evaluates,
   prints `outputs/runs/<run>/eval/report.md`. Use `full` instead of `geom` once you have materials/textures.
   Views: `probe` (12 diverse), `train`, `holdout` (14, only for final checks), or a comma list of image stems.
2. Look at `outputs/runs/<run>/eval/regions.png` (worst regions of YOUR parts: photo | render | edges green
   photo / red model / yellow both | error heat: magenta = excess geometry, cyan = missing geometry) and
   `views.png` (all probe views). Read only these sheets, not whole photos, unless you need context.
3. `orphans.json` lists clusters of sparse 3D points far from any model surface (world mm): missing geometry.
4. Fix and repeat. Per-part numbers are in `parts.json`: fp_frac (excess), fn_px (missing nearby), edge_mean
   (px, model edges vs photo edges), pts_median_mm (sparse points on that part vs surface; best 3D signal).

## Holdout rule (audit 2026-10-05)
Holdout views (`data/split.json` "holdout", 14 views) are for evaluation only: never measure, fit, depth-solve or
texture with them. pick.py refuses them unless --allow-holdout; texture_from_photos.py auto mode skips them;
fit_params uses probe views (all train). Per-part point medians in eval are untruncated (all points whose nearest
surface is the part, within 50 mm) with "pts >4mm" as the share of far points: watch both.

## Measuring (do this instead of guessing)
- `uv run python scripts/tools/pick.py views X Y Z` best views of a world point and its pixel there.
- `... pick.py crop VIEW U V --half 150` labeled grid crop (1/2-scale pixel coords) -> Read it -> read coords.
- `... pick.py tri VIEW U V` one pick -> matched in neighbor views -> 3D point (mm). Check n_inliers >= 3.
- `... pick.py tri2 V1 U1 V1 V2 U2 V2 ...` manual correspondences when tri fails (silhouettes, dark areas).
- `... pick.py ray VIEW U V --z Z` pixel ray hits plane z = Z (good for things on the mat or the floor).
- `uv run python scripts/tools/overlay.py VIEW '[{"box":[x0,x1,y0,y1,z0,z1]},{"poly":[[x,y,z],...]}]'` draws world-mm
  hypotheses on the photo with a pixel grid: check a dimension in 2-3 views before editing the TOML.
- `uv run python scripts/tools/rayplane.py VIEW x|y|z VALUE U V [U V ...]` pixel rays hit any axis plane (wall faces,
  rim heights): for silhouette/textureless points where tri/tri2 cannot match.
- Photos: `data/undistorted_s2/<stem>.jpg` (2856x2142) and `_s4` (1428x1071), EXIF orientation ignored on
  purpose (match COLMAP). Never apply EXIF rotation.

## Fitting continuous parameters
`scripts/bslot.sh -b --factory-startup --python scripts/blender/fit_params.py -- --group <g>
 --params a.b:step,c.d:step --targets <g>.part_prefix --max-evals 80 --out outputs/fits/<g>_<name>`
Pattern search on edge + excess-geometry loss, about 0.5 s per evaluation. Reliable for sizes/positions
within a few mm of the truth (about 1 mm repeatability); it finds local minima from far starts, so measure
first, fit second. It writes best.json; copy accepted values into your TOML yourself.

## Textures (labels, printed parts)
`uv run python scripts/tools/texture_from_photos.py spec.json` (see the docstring): quad (4 world corners in
mm, image orientation TL, TR, BR, BL) or cylinder patch -> `assets/textures/<name>.png` sampled from the
full-resolution originals (best views, per-texel median). Check `outputs/textures/<name>_views.jpg`.
Use with `lib.mat_image` + `lib.label_quad` (or UV-mapped curved meshes).
Crowded planar surfaces (board tops under parts/wires): `scripts/tools/bake_id_owned.py` bakes only from pixels the
object owns in an ID render of the current model (see its docstring; re-run after occluders change). This tool is new: verify its output
visually the first time and report bugs in your devlog.

## Render rig and materials (calibrated, do not change per part)
- RGB renders: uniform white world of strength 1, no lights, EEVEE screen-space ray tracing on. A diffuse
  surface renders at about its base color, so photo textures reproduce photo values.
- Photo-textured materials: `lib.mat_image` (Specular IOR Level 0; the photo already holds the highlights).
  Photo-sampled flat materials: base color = median of a well-lit photo patch, Specular IOR Level 0.2-0.3; metal
  stays metallic. Truly glossy plastic: low specular (the case uses 0.02) since the white world would gray it.
- Glass/clear parts: a textured shell scored better than refracting glass (14.3 vs 12.6 dB on the window).

## Texturing recipe that worked (pcb, crt, case agents)
1. Give each part UVs that follow its real surfaces (box faces, cylinder side and top; one atlas per part, or one
   panel per flat face). Bake on the real mesh, not an idealized shape (removes parallax ghosting).
2. ID-render all train views with the current model (about 7 s):
   `scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- --out outputs/runs/<g>_idtrain
   --build scripts/model/build_all.py --views train --passes id`
3. Each texel = median of its best K = 5 train views where the texel faces the camera and the ID pass says the part
   owns the pixel (`scripts/tools/bake_id_owned.py`, `scripts/model/pcb/bake_parts.py`,
   `scripts/model/crt/texture_yoke.py`, `scripts/model/case/bake_case.py` are working examples).
4. Glossy surfaces: a darker percentile (25th over 8 views) instead of the median drops moving highlights.
5. Re-bake after occluders change (wires over a board); the coordinator sequences these.
Diagnose with a self-texture test: texture a part from one view and render that view. A high score there and a
low multi-view score means view-dependent shine, not geometry.

## Reports and devlogs
- Final report (under 250 words): what changed, how it was measured, before/after per part (`error_budget.py`
  and `parts.json` on probe runs), sheet paths, tool problems, next steps.
- Compare per part, not totals: other agents edit at the same time, and group shares redistribute.
- Runs you delete may be referenced elsewhere: copy anything meant to last (figures, cards) out of `outputs/runs`.

## Pitfalls
- zsh: `$var` does not word-split (use `${=var}`); argparse needs `--` before negative numbers.
- Blender's Python has no PyYAML: parameters are TOML. Keep TOML valid at every save (the builder falls back to
  your last-known-good copy, but your new work is then invisible to everyone).
- Label quads: corners TL, TR, BR, BL; normal = (down) x (right).

## Machine rules
- Blender only through `scripts/bslot.sh` (at most 2 Blender processes machine-wide; it queues). Never kill
  other Blender processes. Prefer `probe` views; `train` (123 views) only for occasional checks.
- Disk is tight (about 11 GiB free): keep run folders few (reuse a run name or delete your own old runs), no
  full-resolution renders.
- Scratch files: `outputs/scratch/<group>/`. Delete your own scratch when done.
