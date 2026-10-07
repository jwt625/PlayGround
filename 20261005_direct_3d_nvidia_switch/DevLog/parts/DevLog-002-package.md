# DevLog-002-package: Spectrum-6 CPO package and stand

Owner: package agent. Files: `scripts/model/package/build.py`, `config/model/package.toml`,
`scripts/model/package/measure/` (measuring scripts, see below). Views: train IMG_5711, 5712 (close-ups), 5717,
5736, 5740. Holdout 5737/5747 never used. IMG_5739 excluded, IMG_5738 (other package) out of scope.

## TODO
- [x] Phase 1A block-out: substrate, ring with notches and plus-shaped opening, 4 x 8 OEs, lid, stand
- [x] Frame correction measured (translation only), applied in build
- [x] Phase 1B: ring inner lips, OE split (gold bracket vs gray chip), heights per part
- [x] Textures: one planar photo texture over all up-facing package faces (lid markings, OE tops, ring QR)
- [x] Notch inner fillets and corner-block rounding (Phase 2, interrupted agent; r_notch 1.5, r_block 1.4, r_open 0.5)
- [x] Stand base re-modeled as an inclined easel panel, post removed (Phase 2, interrupted agent)
- [x] Wave 2: fillets kept (corner crop matches the rounded notch and block corners; ring edge neutral)
- [x] Wave 2: outer-wall atlas (ring + substrate edge) and inner-wall atlas (ring opening/lip/notch walls, field,
  gold OE sides, lid sides); OE chip sides stay flat (leave-one-out worse)
- [x] Wave 2: stand panel checked by overlays; kept (see Wave 2 notes: the 5712 "background" band is the panel side face)
- [x] Wave 2: field caps relief tested (0.35 mm boxes from the texture); texture only kept
- [x] v4 wave: top texture 20 px/mm from the full-resolution originals, views aligned (affine ECC) to their median
- [x] v4 wave: stand rod re-measured (runs to the back plate), ledge fitted and given stadium ends
- [ ] Next: OE relief (connector slot, bracket frame) would need the top bake at the relief heights (caps test);
  OE chip sides; stand panel lower edge in IMG_5717 (render shows background where the photo shows panel)

## Phase 1B (2026-10-06)
Coordinator folded the 1A correction into frames.json (translate now 0). Stand masks now unknown.
### Geometry (0.06 mm/px single-view orthos, NCC heights, edge agreement; `measure/edgeh.py`, `pproj.py`)
- Ring: rim and corner blocks top 2.7; inner lips top 1.3 (NCC 1.0-1.25 on the bands, best NCC 0.83 at
  (0, -51.5)); lips span the opening edge to x = +-51.0 (left/right, step lines -50.7 / +51.4) and y = +-52.4
  (top/bottom, line -52.35), between the corner blocks (|x|, |y| < 34.7). Lip height is the least certain number
  (smooth metal, about +-0.4 mm); ring edge metric 1.60 -> 1.66-1.77 px after adding the lips.
- OEs: pitch 7.88 (autocorrelation, 4 sides x 2 views, 7.86-7.91). Gold bracket 6.9 wide, top 1.3; gray chip
  (QR) 7.4 wide, top 0.85. Radial: rows chip 34.0-39.2 / gold 39.2-45.7; columns chip 31.8-36.6 / gold
  36.6-42.6 (rows sit about 3 mm further out, matching the wider top/bottom ring opening).
- Field (bump/cap area) top 0.2; lid top 0.9. No lid step resolvable (die tiles are reflectance, not relief).
### Texture (`scripts/model/package/bake_top.py`)
- Planar image over the substrate square (UV = package xy, 10 px/mm, 1083 x 1078), applied to faces with
  normal z > 0.9 of all package parts (build `_planar_uv`). Each texel sampled at its part's top height in every
  view whose ID render says the part owns the pixel (1 px erosion at 1/4 scale); unseen 4.8 percent, inpainted.
- Views: IMG_5711, 5712, 5717, 5740, 5736 (all session S1). Shadow: a curved soft shadow covers the upper-left
  ring and OEs in 5711/5712/5717; 5740 fully sunlit and overexposed on the ring. Lid shows view-dependent sky
  reflections (bright arc in 5740/5736) that a diffuse texture cannot reproduce (worst regions now).
- Median vs darker percentile, leave-one-out (bake without the test view, color residual of the whole view):
  test 5717: med 39.8, p25 38.0, p10 37.8; test 5740: med 41.3, p25 42.6, p10 43.7. Sum: p25 best (80.6 vs
  81.1 med, 81.5 p10). In-sample 5 views: p25 34.0 vs med 34.5. Chosen: `assets/textures/package_top_p25.png`
  (median kept as `package_top_med.png`).
- Lid markings legible in the texture (NVIDIA logo, T TW 2609, E9X717.001, e1); they read rotated 180 deg in
  +Z view of this frame.
### Results 1B (5 package views, full mode; flat colors package_003 -> textured package_005)
| part | edge px | pts med mm | color res | ssim |
|---|---|---|---|---|
| ring | 1.74 / 1.66 | 0.38 | 45.2 / 22.2 | 0.561 / 0.674 |
| substrate | 2.06 / 1.98 | 0.16 | 50.7 / 28.1 | 0.491 / 0.655 |
| field | 1.63 / 0.99 | 0.17 | 53.1 / 27.8 | 0.164 / 0.568 |
| lid | 3.57 / 2.07 | 0.07 | 40.3 / 30.2 | 0.347 / 0.610 |
| oe_gold (t/b/l/r) | 1.78/0.98/0.92/1.11 -> 1.17/1.07/0.71/0.89 | 0.27-0.40 | 50.6-70.1 -> 23.6-27.9 | 0.16-0.19 -> 0.41-0.74 |
| oe_chip (t/b/l/r) | 2.13/1.60/1.29/1.12 -> 2.24/1.20/1.55/1.14 | 0.12-0.24 | 37.7-59.8 -> 28.5-39.2 | 0.17-0.22 -> 0.35-0.58 |
In-sample (texture baked from these views); leave-one-out numbers above are the fair comparison.
Sheets: `outputs/package_1b/` (regions.png, views.png, parts.json, parts_flat_003.json, package_top_views.jpg).

## Requests to coordinator
1. Fold the frame correction into `config/frames.json` "package": local translation (-4.0, -3.1, -0.7) mm,
   rotation 0. New origin (substrate top center) in world mm: (112.03, -150.52, 77.84); R unchanged. After
   folding, set `[frame_correction]` in package.toml to zeros (the build applies it on top of frames.json).
2. Package masks exclude the stand, so stand parts score as excess geometry (fp 0.5-1.0) in every view. Add the
   stand (ledge, rod, base plate) to the package SAM object or to the unknown hull, if stand scores matter.
3. `config/eval.yaml` point region and `hull_unknown` are in the provisional frame (box centered on the old
   origin, 4 mm off the package center): still cover the package (outer edges at -58.2/+50.1 x, -57.0/+50.8 y).

## Measurements (corrected frame: origin substrate top center, X/Y along ring edges, Z out of top)
Method: orthophotos of the package plane (`measure/ortho.py`, `ana.py` two-view anaglyph), edge profiles along
lines (`prof.py`, `edge1.py`), plane-height by multi-view patch NCC over the 5 train views (`hfit.py`).
- Ring outer edges at h = 2 (old frame), 5 positions per side, 3 views agree within 0.3 mm:
  x -58.15 / +50.10, y -56.95 / +50.80 (old frame) -> 108.25 x 107.75 mm, center (-4.0, -3.1).
  In-plane rotation below 0.3 deg (edge slopes mixed sign), no correction.
- Substrate edge (seen in the notches) coincides with the ring edge within 0.5 mm: substrate about 108 x 108 mm.
  Analyst figure 110 mm (SemiAnalysis, check only): measured 1.6-1.8 percent smaller, beyond the about 1 percent
  scale uncertainty. Flag; not resolved.
- Heights (NCC, old frame): ring top 1.0-3.0 (median 2.0, 11 patches), substrate in notches/gaps -1.5..-0.25
  (about -0.7), OE tops 0.5, lid 0.25-0.5. Corrected: ring 2.7, OE 1.2, lid 1.0 above the substrate top.
  Ring top tilt below 0.4 deg. Substrate thickness not observable: 2.5 mm assumed.
- Notches 10.2 mm square. Opening: |x| < 45.1 at the OE columns, |y| < 48.25 at the OE rows, ring corner blocks
  fill |x|, |y| > 35.
- OEs: 8 per side, 32 total (counted on IMG_5711 orthophoto), pitch 7.85 mm (65-66 px at 0.12 mm/px, all four
  sides), footprint 6.8 mm along the side, radial 33.8-45.0 mm.
- Lid (gray surface incl. light rim): 49.5 x 54.0 mm at (-1.0, 0.35); 0.05 mm/px anaglyph at both corners.
- Colors: photo medians over 5711/5712/5717 (sun and shade mixed): ring 0.58/0.55/0.46, field (light-blue
  substrate) 0.21/0.41/0.49, notch substrate dark 0.02/0.04/0.045, OE 0.39/0.33/0.20, lid 0.22/0.21/0.18 (linear).
- Stand (black, faint; brightened crops, epipolar lines `epi.py`, contour overlays `idov.py`):
  - Base plate: front-edge corners from IMG_5711 x IMG_5712 corner picks (depth 270-275 mm): (-39, 96, -38) and
    (33, 99, -42); plate is horizontal (gravity up from the scene frame); its outline then fits 5711, 5712, 5717.
    72 x 72 x 5 mm; depth 72 assumed (back edge hidden), thickness from the 5711 front face (about 5 mm).
  - Ledge: thin black bar under the package +Y (gravity bottom) edge, X -28..23 (5740), 51 x 4 x 10 mm.
  - Rod: from the ledge backward; IMG_5711 sees it end-on (blob), IMG_5717 side-on with an end cap.
    p0 (-9, 59, -10), p1 (0, 72, -67), r 2.75.
  - Post from the rod end down to the base top (about 16 mm): assumed, not visible in any photo.
  - Back plate 100 x 100 x 4 under the substrate: fake, hides the unseen underside.

## Progress
- 2026-10-05 23:20 Baseline (coordinator stub slab, run package_000, 5 package views): substrate fn 39552 px,
  edge 5.58 px, pts median 0.43 mm.
- 2026-10-05 23:33 package_001: full block-out. Package parts edge 1.05-2.04 px, pts median 0.12-0.49 mm,
  substrate fn 162. Stand fp high (mask excludes stand).
- 2026-10-05 23:45 Stand from ID-contour overlays on brightened photos: ledge shortened/centered, rod start moved
  onto the 5717/5711 cues, Specular IOR Level 0.25 for all package materials (black stand rendered gray at 0.5).
- 2026-10-05 23:55 package_002: lid enlarged to the outer rim (45.5 x 52 -> 49.5 x 54). Lid edge metric
  3.07 -> 3.64 px (strong inner die-tile edges now unmatched) but the contour follows the rim in 5711/5717.

## Results (package_002, views IMG_5711,5712,5717,5736,5740; `outputs/package_1a/`)
| part | fp | fn px | edge px | pts n | pts med mm |
|---|---|---|---|---|---|
| substrate (stub before) | 0.000 | 39552 | 5.58 | 664 | 0.43 |
| substrate | 0.010 | 340 | 2.03 | 35 | 0.18 |
| ring | 0.000 | 964 | 1.60 | 49 | 0.43 |
| field | 0.000 | 0 | 1.85 | 246 | 0.30 |
| oe_top / bottom / left / right | 0 | 0 | 1.60 / 1.29 / 1.05 / 1.87 | 29/80/77/28 | 0.49/0.42/0.29/0.31 |
| lid | 0.000 | 0 | 3.64 | 117 | 0.12 |
| stand_ledge | 0.054 | 262 | 4.63 | - | - |
| stand_rod / post / base | 0.50 / 0.94 / 0.97 | 0 | 9.9 / 10 / 8.4 | - | - |
Sheets: `outputs/package_1a/{regions.png,views.png,parts.json,report.md,contours_5711_5712_5717.jpg}`.

- 2026-10-06 1B: lips, OE split, heights (package_003); bake + full-mode med/p25 (package_004/005); LOO tests
  on 5717 and 5740 (package_loo, deleted); final package_005. About 10 render runs.

## Phase 2 (interrupted, 2026-10-06 00:00-00:03; reconstructed by the Wave 2 agent from files and runs)
The previous agent's session ended before it updated this devlog. From package.toml, build.py (00:02),
bake_top.py (00:03), measure/fit_base.py, outputs/scratch/package/ and runs package_006/007:
- Stand: the horizontal 72 x 72 base plate and the assumed post were replaced by an inclined black easel panel
  (`[stand] panel_fl/fr/bl`, `panel_r` 7, `panel_thickness` 5; build object `package.stand_base`, rounded-rect
  prism). `measure/fit_base.py`: 7-parameter fit (yaw, shift x/h, width, depth, 2 tilts) to 13 outline points
  read in brightened s4 crops of IMG_5711/5712: rms 5.1 px vs 21.5 px for a horizontal plate; 33 deg from
  horizontal. `stand_post` removed (185 -> 184 objects); `rod_p1` extended to the panel front face
  (-1.5, 69.8, -57.2). Check sheet: `outputs/scratch/package/idov_package_006.jpg` (ID contours on brightened
  photos): the panel outline fits 5711; in 5712 it overshoots the lower-right corner onto background.
- Ring fillets: `_fillet_poly` in build.py; notch inner corners r_notch 1.5, corner-block convex corners into
  the opening r_block 1.4, opening corners at the lips r_open 0.5 (0.06 mm/px ortho, about 25 px radius; check
  crop `outputs/scratch/package/corner.jpg`, photo ortho | render with outlines).
- bake_top.py: `--sides` mode added (outer-wall atlas: 4 bands +y, -y, +x, -x, each from the ring top down to
  the substrate bottom; ring rows above z 0, substrate rows below, notch spans of the ring rows empty). Not yet
  run (no package_side textures) and not wired into build.py.
- Runs (5 package views, full): package_005 (1B final) -> 006 (panel, no post) -> 007 (fillets).
  | part | 005 | 006 | 007 |
  |---|---|---|---|
  | stand_base fp / edge px | 0.048 / 7.97 | 0.060 / 6.87 | 0.052 / 6.77 |
  | stand_post edge px | 9.99 | removed | removed |
  | stand_ledge fn px / edge | 153 / 4.64 | 24 / 4.81 | 24 / 4.81 |
  | ring fn px / edge | 1493 / 1.66 | 1425 / 1.66 | 1425 / 1.66 |
  | substrate fn px / edge | 218 / 1.98 | 206 / 1.98 | 206 / 2.04 |
  Other package parts unchanged within 0.01 px. The fillets are neutral on the ring and cost 0.06 px on the
  substrate edge (substrate visible in the notches is now rounded).

## Wave 2 (2026-10-06 00:10-00:40, Wave 2 package agent)
Views: train IMG_5711, 5712, 5717, 5736, 5740 (full mode). Holdout never used.
- Baseline package_008 (state after the interrupted Phase 2, other groups current).
- Side visibility (`outputs/scratch/package/sidevis2.py`, wall normal . view direction): among all train views only
  IMG_5712 sees the -x outer wall (12 px tall at 1/4 scale), IMG_5717 the +y wall (11 px, lower part hidden by the
  ledge), IMG_5740 the +x wall (6.5 px, owned texels only in aliasing slivers: dropped); -y unseen.
- Outer-wall atlas: `bake_top.py --sides` now has a facing test (`--min-cos` 0.03), a scale-2 ID pass
  (outputs/runs/package_id2; the 1/4-scale ID aliased the 2-5 px walls into sawteeth), unseen texels filled by
  the median of the same atlas row over the seen bands (instead of inpainting across bands), and a per-view debug
  sheet. `assets/textures/package_side_p25.png` from IMG_5712 + 5717; build `_side_uv` (texture.sides).
  package_009: ring color 22.3 -> 21.6, SSIM 0.674 -> 0.696; substrate 28.1 -> 27.5 (in-sample: one view per wall).
- Inner-wall atlas: build `_wall_charts` gives every near-vertical face (|n.z| < 0.5) of ring, field, gold OEs,
  lid its own chart (20 px/mm, shelf-packed, 2048 x 832); `PACKAGE_WALL_CHARTS=<json>` dumps the charts during an
  ID render; new `bake_walls.py` samples each texel at its 3D point in views where the object owns the pixel
  (scale-2 ID, 1 px erosion) and the face faces the camera (cos > 0.2); median; unseen texels = median of the
  object's seen texels with the same outward direction (8 sectors). Seen: ring 69 percent, gold OEs 16-30, lid 18.
  `assets/textures/package_walls_p50.png` (texture.walls; texture.walls_skip = ["oe_chip"]).
  Leave-one-out (bake without the test view; flat walls vs textured, whole view color residual / SSIM):
  test 5717 33.4 / 0.474 -> 33.1 / 0.478; test 5712 26.9 / 0.578 -> 26.9 / 0.577. With chip sides included,
  5717 chip_bottom went 35.2 -> 43.0, so chips stay flat. In LOO: ring 19.2 -> 17.9 (5717), 16.5 -> 16.6 (5712);
  gold_top 29.7 -> 28.4 and 20.5 -> 18.4.
- Stand: ID contours on brightened photos (`idov_package_009.jpg`) and the 13 fit targets
  (`outputs/scratch/package/tgt.py`): in IMG_5712 the 10-15 px black band outside the panel's top-face outline is
  the panel's 5 mm side face (consistent width along both visible edges and around the corner); the model's
  silhouette ends at its outer boundary. The mask labels that band background, which is the 5-9 percent stand_base
  fp. Added `panel_dx/df/dz/dw/dd/yaw` adjustments (default 0) and ran fit_params (3 views, w_fp 10, 80 evals,
  `outputs/fits/package_panel1`): edge 7.06 -> 6.96 px only, by shrinking the panel 4 mm and pushing it 5 mm back
  (fp is biased toward shrinking here). Rejected; panel unchanged. Rod matches the 5717 side view; its edge metric
  (9.9 px) reflects missing photo edges (black on black).
- Field caps: `measure/caps.py` thresholds the top texture (gray above a 31 px median background + 25) inside the
  eroded field: 791 components, `scripts/model/package/caps.json`; build option `[field] cap_height`.
  Relief 0.35 mm (package_011, chip-colored sides; package_012 field-colored sides), pixel-weighted over package
  parts vs package_010: color 26.53 -> 26.45 / 26.56, SSIM 0.635 -> 0.622 / 0.623, edge 1.48 -> 1.52 / 1.54 px,
  sparse points 0.199 -> 0.177 mm. Appearance worse (the top texture was baked at the field height, so the cap tops
  carry 0.35 mm parallax): cap_height = 0, texture only.
### Results Wave 2 (package_008 -> package_013, 5 views, full; pixel-weighted package parts without stand:
color 27.08 -> 26.65, SSIM 0.610 -> 0.625, edge 1.516 -> 1.516 px, points 0.199 mm unchanged)
| part | color res | ssim | edge px |
|---|---|---|---|
| ring | 22.3 -> 21.5 | 0.674 -> 0.718 | 1.66 -> 1.68 |
| substrate | 28.1 -> 28.2 | 0.652 -> 0.664 | 2.04 -> 2.06 |
| field | 27.8 -> 27.7 | 0.568 -> 0.568 | 0.99 -> 0.98 |
| lid | 30.2 -> 29.6 | 0.610 -> 0.610 | 2.07 -> 2.06 |
| oe_gold t/b/l/r | 26.4/28.0/23.7/27.7 -> 24.1/28.0/23.6/27.5 | 0.410/0.682/0.735/0.598 -> 0.460/0.692/0.745/0.602 | 1.17/1.07/0.71/0.89 -> 1.11/1.05/0.69/0.88 |
| oe_chip t/b/l/r | unchanged within 0.1 | 0.345 -> 0.351 (top) | unchanged |
| stand_base / ledge | 19.0 / 60.7 -> 20.1 / 63.2 | 0.733 / 0.289 -> 0.698 / 0.268 | 6.77 / 4.81 -> 6.80 / 4.83 |
Stand geometry unchanged; its color shift appeared between 008 and 009 (per-view color fit and neighbors).
Sheets: `outputs/package_w2/` (regions.png, views.png, parts.json, parts_before_008.json, report.md,
package_side_perview.jpg, package_walls_views.jpg, idov_package_009.jpg).
Cleanup not done (deletion was refused by the permission system): my runs package_009-012, package_loo_flat12/17,
package_loo12/17 and assets/textures/package_walls_loo12_p50.png, package_walls_loo17_p50.png can be deleted;
package_008 (before), package_013 (after) and package_id2 (scale-2 ID pass used by the bakes) should stay.

## v4 detail wave (2026-10-06 02:30-03:20)
Views: train IMG_5711, 5712, 5717, 5736, 5740 (full). Holdout photos, crops and runs not opened.
Baseline drift from other agents between back-to-back identical runs: package color 26.68 (103) vs 27.16 (107),
so texture variants are compared to package_107 (same config as 103, run next to them).
- Top texture density: full-resolution originals give 20.4 (5711), 22.7 (5712), 20.2 (5717), 16.3 (5736),
  15.3 (5740) px/mm at the package (`outputs/scratch/package/dens.py`). `bake_top.py` gained `--full`
  (texture_from_photos.sample_view, Lanczos on the originals), `--cache` (per-view owned samples, npz),
  `--region-view`, `--erode`; ID ownership from a fresh scale-2 ID pass (`outputs/runs/package_id2`).
  20 px/mm bake: 2166 x 2156, 6.5 min. New `align_top.py`: per-view affine ECC registration of the cached samples
  to a reference, then the 25th percentile. Dense DIS flow was unusable (p95 80-157 texels on the smooth ring and
  lid reflections); affine shifts are 1-4 texels (0.05-0.2 mm), scale 0.1-0.4 percent; IMG_5736 9 texels in y.
  | variant (pixel-weighted package parts, no stand) | color | SSIM | edge px |
  |---|---|---|---|
  | 10 px/mm p25 (package_107, baseline) | 27.16 | 0.624 | 1.488 |
  | 20 px/mm p25 unaligned (106) | 26.81 | 0.637 | 1.435 |
  | 20 px/mm aligned to IMG_5717 (104) | 27.76 | 0.597 | 1.526 |
  | same + lid from IMG_5717 only (105) | 28.58 | 0.612 | 1.563 |
  | 20 px/mm aligned to the median (108, kept) | 27.05 | 0.633 | 1.498 |
  Kept 108 (`assets/textures/package_top20m_p25.png`): within noise of the best on metrics and visibly the sharpest
  (OE connector slot lines, bracket screw bosses, chip QR codes, lid text "E9X717.001", "T TW 2609", "e1",
  NVIDIA logo; sheets `outputs/package_v4/tcmp_oe.jpg`, `tcmp_lid.jpg`). Single-view lid rejected: lid color
  30.3 -> 35.3, edge 2.11 -> 2.39 (5717 carries its own reflections; per-view lid samples in `tc_lid.jpg`:
  5711 camera/phone reflection, 5712 tree reflections, 5740/5736 sky arc).
- Stand rod: IMG_5740 shows the rod as a vertical band at s2 u 962 above the ledge, not the diagonal the old
  "end-on in 5711" endpoints gave. Line fit to 4 observations (5740 band, 5717 rod line v 932) with p1 on the
  panel plane: p0 (4.5, 24.4, -6.5) behind the back plate, p1 (-3.5, 73.0, -54.2); residuals below 1.3 px; 5711
  blob consistent (2030, 1040 vs 2052, 1042 s2). The rod passes 30 mm behind the ledge (hidden by it in 5740).
  stand_rod edge 9.89 -> 8.19 px (102).
- Ledge: stadium ends (ledge_r) and a fit_params fit of y, z, thickness, height (`outputs/fits/package_ledge1`,
  4 views, edge 4.71 -> 2.19 px at 50 pct): 54.5 / -7.1 / 3.0 / 8.0 from 56.5 / -4.0 / 4.0 / 10.0. Eval: ledge
  fp 0.020 -> 0.000, fn 24 -> 3 px, color 64.0 -> 53.1, edge 4.83 -> 4.91 (neutral). The thinner bar matches
  IMG_5717; in IMG_5740 the photo bar is somewhat thicker than the model (`pr_stand.jpg`).
  stand_base color 20.3 -> 31.1 and SSIM 0.69 -> 0.33 changed with the ledge only (panel unchanged): its scored
  pixels are the few outside the unknown mask region next to the ledge.
- Also added (not used): `stand_specular` hook.
### Results v4 wave (package_100 start -> package_108, 5 views, full)
Pixel-weighted package parts without stand: color 26.50 -> 27.05 (same-time baseline 107: 27.16), SSIM 0.632 ->
0.633, edge 1.493 -> 1.498 px, points 0.199 mm. Per part 100 -> 108: oe_chip_bottom edge 1.20 -> 1.05,
oe_chip_left 1.56 -> 1.45 (SSIM 0.535 -> 0.574), field SSIM 0.573 -> 0.583, substrate edge 2.05 -> 1.94, lid
color 29.6 -> 30.0; stand_rod edge 9.89 -> 9.09, ledge fn 24 -> 3 px.
Sheets: `outputs/package_v4/`. Scratch: `outputs/scratch/package/top20_samples.npz` (140 MB per-view sample cache,
reusable for re-blends; delete when done). Superseded textures (not referenced): package_top20_med/p25,
package_top20a_p25, package_top20aL_p25. Runs 101-107 are mine and can go; keep 100, 108, package_id2.

## Notes
- Package frame orientation: printed text on the lid reads rotated 180 deg in the +Z orthophoto (not mirrored).
- Measuring scripts in `scripts/model/package/measure/` write to `outputs/scratch/package/` (mkdir first); they
  use the provisional frame from frames.json (ovl.py adds the correction offset).
