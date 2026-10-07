# DevLog-002-lower: front bay contents, braided cables, front-bay copper tubes

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | Wave 5 (details toward v4) done 2026-10-06; Wave 4 (clean restart) done: fiber exit ribbons, per-region floor/board sessions, SMD texture, connector |
| Scope | Front bay Y 0-330 contents (control board with NVIDIA logo, connector, clips, 8 copper cold-plate fingers, OE frames, copper loops and long tubes, center manifold, fiber organizers, white strips, grouped blue fibers, floor), braided cables over their full length, copper tubes up to Y = 350 |
| Files | `scripts/model/lower/build.py`, `config/model/lower.toml`; measuring helpers `scripts/model/lower/ortho.py` (orthophoto on z = const), `crop.py` (s2 grid crop), `tri_poly.py` (curve from polylines traced in 2-3 views), traces in `scripts/model/lower/traces/` |
| Author | Claude (lower agent) |

## 1. TODO (Phase 1A)

- [x] Read AGENT_GUIDE, CRT PLAYBOOK 5-8, wires "How to route", DevLog-001
- [x] Layout from orthophotos (planes z = 0, -30, -32, -35; train views IMG_5725, 5751, 5749) and sparse-point Z
      histograms per region
- [x] Braided cable A path by polyline triangulation (IMG_5749 + 5752 + 5751), checked by overlay on IMG_5758, 5753
- [x] Braided cable B path (IMG_5735 + 5749 + 5725; weaker, 4-8 px residual in 5749)
- [x] First build (all parts present)
- [x] Iterate on probe + own front-bay list (9 probe iterations + 2 FB runs)
- [x] Final per-part numbers (`DevLog/parts/DevLog-002-lower_parts_20261005.txt`: lower_001 | lower_009 | lower_fb)
- [x] 1B: regressions fixed (board_bracket 5.34 -> 2.16 mm, ribbon_a_front 5.45 -> 0.47 mm)
- [x] 1B: flange identified (gray frame under the board, band Y 110-116 at Z about -6); finger split (3 blocks +
      silver straps); braid B offsets fitted
- [x] 1B: photo textures (board top, finger tops, OE tops, floor, organizers, strips, manifold, left plate)
- [ ] Next: braid A/B textures (cylindrical bake along the curve), fiber count/route detail, OE port boots,
      connector latch, clip strap shape, color of white clips/black loop (shading-limited)

## 2. Plan

1. Measure: orthophotos (own scratch tool `ortho.py`: resample one train view onto a plane z = const, mm grid);
   features at that height read directly in XY. Heights from sparse points clustered by color class.
2. Build block-out: boxes for board/connector/clips/fingers/OE frames/organizers/strips/manifold; filleted POLY
   tubes for copper; Bezier tubes for cables; grouped fiber tubes (2 per OE + 6 exit bundles per organizer).
3. Iterate with `scripts/iterate.sh lower_NNN <views> lower. geom`; views: probe, and front-bay list
   FB = IMG_5713,5721,5722,5723,5724,5725,5726,5735,5736,5748,5749,5752,5753,5757,5758.

## 3. Measurements (mm, tray frame)

- Control board: X -100..101, Y 52..108 (orthos z = 0 in IMG_5725 and 5751 agree to about 1 mm); top Z 0.0
  (1656 sparse points in Z -2..2; median 0.1). Six screw holes at X -88/0/96, Y 63/101.
- Connector top about Z +12..15 (sparse 75th-90th pct); footprint X 22..88, Y 90..113 (ortho).
- Fingers: 8 at 24 mm pitch, centers X -68, -44, -20, 4, 28, 52, 76, 100 (ortho z = -30 at 5 px/mm); copper
  Y 113..170 (3 blocks, the last with a hole and tab); copper top Z -27 (median of 386 copper points -27.2).
- Copper legs: 2 per finger at center +- 4.5, tube r 2.75 (5.5 mm wide in the ortho), center Z -32 (points
  median -30 on the visible upper surface). U-bends join f1-f2, f2-f3, f3-f4, f5-f6, f6-f7, f7-f8 at Y 200-215.
  Long legs f1L, f4R, f5L, f8R run to four compression fittings at X -2.5, 7.5, 20.5, 34.5, Y 284-299, Z -33.
- Center manifold block X -9..42, Y 298..326, top about Z -28 (sparse median -28.9).
- Organizers: L X -57..-17, Y 266..300, top -27; R X 50..92, Y 268..300, top -25. White strips Y 237.5..242.5,
  L X -72..3, R X 24..88, top -36.5 (sparse -37).
- OE floor top about Z -39 (white strips and fibers lie on it; gaps -38..-42).
- Braided cable A: 30 triangulated points (unc 1-2 mm front part, 3-15 mm at the crossbar clip); crossbar section
  set by overlay checks; ends: front at the connector ribbon (Y 128), rear at Y 592 (sparse cluster X 61, Z 6,
  n 19), then a ribbon fan to Y 614. Diameter about 10-11 mm (51 px at 480 mm in IMG_5758).
- Braided cable B: X -116 -> -105 over Y 110 -> 288, Z -33 -> -14 (polyline triangulation, residual 4-8 px in
  IMG_5749), then under the crossbar (sparse Y 430 Z -38, Y 490 Z -34).

## 4. Interface with upper: tubes at Y = 350

- No copper tube crosses Y 320-410 in the front bay or under the crossbar (no copper-colored sparse points there
  except outliers; orthos show the four front-bay tubes ending in the center manifold block at Y 298-326).
  So lower has no tube ends at Y = 350. Upper's wall tubes (X +-196, +-206, Z -28 / -44) end at Y = 350 under the
  crossbar; in the front bay X +-196..206 is inside the black ducts (chassis) and no copper is visible there,
  so lower does not continue them.

## 5. Progress log

- 2026-10-05: measured layout (orthos, sparse clusters, cable triangulation); first build written (lower_001:
  all parts present, every lower part pts median <= 6.5 mm, most 0.4-2 mm).
- 2026-10-05: lower_002: fiber exits bend down right after the organizers (regions: fibers pass under the
  crossbar edge); crossbar clip moved to rayplane z = 0 base picks; bracket as a flange.
- 2026-10-05: lower_003: bay tilt. Plane fits show the board (z = 0.0119 x + 0.0067 y - 0.09, 1640 pts) and the
  crossbar top (z = 0.0111 x + 0.0051 y - 0.95, 754 pts) are not level in the world frame (about 0.65 deg about Y);
  finger tops go from -28.5 (f1) to -24.2 (f8). The rigid front-bay assembly is built level and sheared with
  [tilt] (dzdx 0.0115, dzdy 0.006 about X 0, Y 200); measured cables are not sheared. oe_7 pts median
  3.91 -> 0.76, finger_7 2.49 -> 1.60, board 0.49 -> 0.32.
- 2026-10-05: lower_004: cable loop from 29 dark sparse points (pts median 4.23 -> 2.01).
- 2026-10-05: lower_005/006: wire fans (9 tubes) for both ribbon ends; front fan and braid start moved -X 3-7 mm
  (edges 2.02 -> 1.51 px; pts median on 11 points worse, 1.7 -> 5.5 mm: small n, left for 1B).
- 2026-10-05: lower_007: crossbar clip by tri2 on clip corners (IMG_5751 + 5753, 0.5-2.9 px): about X 36..66,
  Y 343..362, Z 0..24 (top face is not level; strap over the cable). Cable A rerouted through the clip and over the
  crossbar using sparse teal clusters (Y 390 X 46 Z 13; Y 430 X 48 Z 9): braid_a pts median 2.79 -> 2.12, edge
  2.60 -> 2.35 px. The polyline triangulation was ill-conditioned there (cable along the epipolar lines).
- 2026-10-05: lower_008: braid B nudged +X/+Z from point offsets: no gain (edge worse, rod edge worse): reverted.
- 2026-10-05: lower_009 (final 1A, probe): every lower part present; pts median <= 2.5 mm except board_bracket
  5.3, ribbon_a_front 5.5 (n 11), fibers_exit_r 3.8. Edge mean 1.1-2.4 px for fingers, OE frames, loops, tubes,
  fittings, strips, organizers; 3.4-4.1 px for board (front edge under the chassis lip), braid B, clip, fiber exits.

## 5b. Phase 1B (2026-10-05)

Baseline lower_100 (1A geometry, full mode, probe) -> final lower_106. Per-part table (edge px, pts median mm,
color residual, SSIM): `DevLog/parts/DevLog-002-lower_parts_1B_20261005.txt`.

- Geometry: bracket = gray frame under the board (X -100..101, Y 60..117, local Z -8..-1.6; orthos on z = -6 in
  IMG_5725/5751/5722 agree within 3 mm; local points cluster Y 112-116, Z -8..-4): pts median 5.34 -> 2.16.
  Connector from dark points (top edge dense Y 92-96, top Z 12-18): Y 89..107, top 15. Front wire fan from 39
  light-blue points (X 29-52, Y 106-118, Z 11-14): fan line Y 107, Z 12.5 -> braid start (48, 131, 7.5):
  ribbon_a_front pts 5.45 -> 0.47, edge 1.51 -> 1.12. Fingers split into 3 copper blocks (Y 116-132, 134-150,
  152-170) + silver straps: finger pts medians 0.98-2.47 -> 0.45-1.06.
- Braid B: offsets (dx, dz, dz_slope with weight 1 to Y 290, 0 from Y 345) by fit_params on 8 train views
  (`outputs/fits/lower_braid_b/best.json`, edge loss 1.747 -> 1.692): dx 0.75, dz 2.06, dz_slope -2.88 per 100 mm.
  Small gain (edge 3.36 -> 3.24 px, color residual 52.2 -> 49.9). Real depth still weak (see TODO).
- Textures: `scripts/model/lower/bake_lower.py` (ID-owned texels; train views ranked by cos x px/mm, filtered by
  session; median of the first 5 visible views; 8 px/mm). ID run of all 38 train views with the current model.
  Quads `lower.tex_<name>` lie 0.15 mm above each top face and are sheared with [tilt]; without a texture file
  they fall back to the part's flat material. Session chosen per surface by full-mode probe scores
  (S1 = 12:31-12:37, S3 = 13:36):

  | texture (assets/textures/) | session | why |
  |---|---|---|
  | lower_board_top.png | S3 | color res 56.8 (S1) vs 52.9 (S3), SSIM 0.395 vs 0.430 |
  | lower_finger_tops.png | S3 | 41.6 vs 37.4 |
  | lower_oe_tops.png | S3 | 45.2 vs 42.3 |
  | lower_floor.png | S3 | 59.8 vs 59.3 |
  | lower_org_l.png | S3 | 41.1 vs 41.0 (SSIM 0.403 vs 0.417) |
  | lower_org_r.png | S1 | 48.8 vs 56.2 |
  | lower_strip_l.png | S1 | 59.6 vs 60.6 |
  | lower_strip_r.png | S3 | 64.1 vs 56.1 |
  | lower_manifold.png | S3 | SSIM 0.223 vs 0.276 |
  | lower_plate_l.png | S3 | 45.0 vs 43.4 |

  Darker percentile (p30, S3) on board/fingers/OE was slightly worse (52.9 -> 53.3, 37.4 -> 38.4, 42.3 -> 42.8):
  kept the median. Holes (never-owned texels, filled with the median): strips 0.53-0.66 (fibers on top), fingers
  0.38-0.41 (gaps and straps in the rect), OE 0.72-0.78 (under legs), floor 0.41-0.54.
- Results (lower_100 -> lower_106; board top now = tex_board): board top edge 4.12 -> 1.39 px, color residual
  64.0 -> 45.5, SSIM 0.291 -> 0.464; organizers L 48.9 -> 39.1 (SSIM 0.233 -> 0.428), R 56.9 -> 48.7;
  finger tops 38-56 -> 36.4 (SSIM 0.337); OE tops 41.3; left plate 52.5 -> 42.6. Strips stay about 60 (thin,
  mostly covered by fibers).
- Negative: flat color refit (`scripts/model/lower/color_fit.py`: photo/render median ratio over ID-owned pixels
  of 15 train views) made most flat parts worse on probe (white clips 35.9 -> 43.3, copper sides +1..3);
  reverted except base and braid_b (oe_base 58.9 -> 55.0, braid_b 52.2 -> 50.3).
- Tool problem: `scripts/tools/texture_from_photos.py` sample_view reads `<stem>.jpeg`, but the sources here are
  `.JPG`, so texture_from_photos.py and bake_id_owned.py fail on this capture. bake_lower.py has a local copy that
  finds the file by stem. (Coordinator: shared tool fix needed.)


## 5c. Phase 2 (interrupted 2026-10-06 00:03), reconstructed from files by the Wave 2 agent (2026-10-06)

The previous agent's session ended before it updated this devlog. Reconstructed from runs lower_200..203,
`outputs/fits/lower_braid_b2/`, `outputs/scratch/lower/*.log`, file times and the TOML:
- 23:58-00:00 `scripts/model/lower/tubegeom.py` (new): shared centripetal Catmull-Rom centerline, parallel-transport
  frames and tube surface grid used by both build.py (`_tube_mesh`: UV tube mesh, u along, v around) and
  bake_lower.py (`--tubes`: cylindrical bake on the same grid, ID-owned, cos > 0.3, per texel the median of its
  best 5 views by cos x px/mm, 4 px/mm). braid_b's fitted offsets moved from build.py into tubegeom.centerline.
- lower_200 (probe, full): braid_a/b as UV tube meshes, flat color: same numbers as lower_106 (braid_a edge 2.33,
  color res 60.4; braid_b 3.29 / 50.3).
- Braid bakes by session (ID run `outputs/runs/lower_idtrain`, 38 train views) and A/B on probe (full):

  | run | texture session | braid_a edge / pts / color res / SSIM | braid_b edge / pts / color res / SSIM |
  |---|---|---|---|
  | lower_200 | none (flat) | 2.33 / 2.11 / 60.4 / 0.199 | 3.29 / 2.27 / 50.3 / 0.157 |
  | lower_201_S1 | S1 | 1.89 / 2.10 / 52.0 / 0.218 | 3.17 / 2.16 / 50.2 / 0.197 |
  | lower_201_S3 | S3 | 2.11 / 2.11 / 57.1 / 0.191 | 3.00 / 2.27 / 49.5 / 0.175 |
  | lower_201_all | all train | 1.80 / 2.11 / 47.3 / 0.224 | 3.17 / 2.27 / 48.8 / 0.199 |

  Kept "all" for both (`assets/textures/lower_braid_a.png` = `lower_braid_a_all.png`, same for b; the per-session
  files are the A/B variants). Note: this mixes sessions (the cables cross the whole bay and no single session
  covers them well).
- 00:00-00:02 fit `lower_braid_b2` (fit_params, knot offsets kx/kz at Y 125/200/275, 10 train views; edge loss
  1.791 -> 1.720; best kx200 -4.5, kz200 +6.75), braid_b re-baked on the moved centerline, lower_202/203: braid_b
  color res 49.3 / 49.2, SSIM 0.173 / 0.184, worse than lower_201_all (48.8 / 0.199). Reverted at 00:03: knots
  zeroed in the TOML, `lower_braid_b.png` restored from `_all` (identical md5).
- State at the resume (= v1 snapshot, `outputs/runs/v1_probe`): braid_a edge 1.80 px, pts 2.11 mm, color res
  47.3, SSIM 0.224; braid_b 3.17 / 2.27 / 48.8 / 0.199. Not done: better depth for braid B, fiber detail,
  connector latch, clip shapes, OE port boots, board/finger re-bake.
- v1 holdout error budget (share of squared error, 5 holdout views; group lower 7.1 percent): tex_floor 1.5,
  tex_board 1.3, tex_manifold 0.5, braid_b 0.5, tex_plate_l 0.4, tex_org_l 0.3, braid_a 0.3, connector 0.2,
  tex_fingers 0.2, rod_l 0.2, smd_board_l 0.2, everything else <= 0.1.

## 5d. Wave 2 (2026-10-06)

TODO
- [x] Tray-frame switch: rigid front-bay parts built level, `lib.apply_frame(objs, "tray")`; residual dzdy kept
      (board plane fit in the tray frame); deep-part X values converted to tray-local X; board/fingers re-checked
- [x] Braid A/B textures: A/B of "all" vs one session with fill (braid B now S1 + S3 fill; braid A stays "all")
- [x] Braid B depth: Viterbi over a braid-texture mask in 14 train views (corridor +-8 mm)
- [x] OE port boots (new); crossbar clip fitted; connector latch: no evidence in the points (not modeled)
- [x] Fiber exits: tested a later dive (points), worse in edge/color: reverted
- [x] Holdout budget items: re-bakes of all flat textures on the new ID pass; "all"-session variants lost
- [x] Final per-part table: `DevLog/parts/DevLog-002-lower_parts_wave2_20261006.txt` (v1_probe | lower_215)
- [ ] Next: braid A mask (teal braid not isolated by the speckle mask), fiber exits with a crossbar-aware path,
      connector latch from tri2 picks, floor texture (holdout share 1.5 percent; shadows across sessions)

Progress

- 00:10 Plane fits in the tray frame (train points, `outputs/scratch/lower/planefit_tray.py`): board z = -0.0016 x
  + 0.0061 y (1382 pts, MAD 0.17), finger tops/organizers/floor dzdx +0.003..0.008 and dzdy -0.009..-0.015 (stacked
  sub-features, unreliable), joint board/fingers/floor dzdy +0.0008..0.0037. Roll is captured by the tray frame;
  kept the residual pitch dzdy 0.006 about Y 200 (board evidence), applied as a shear before apply_frame.
- lower_210 (switch only): fingers/OE edge and SSIM worse (finger_0 1.04 -> 1.25 px, tex_oe SSIM 0.186 -> 0.134).
  Cause: the rotation moves parts at depth in X (x_world = x_local - 0.0148 z: +0.40 mm at Z -27, +0.58 at -39),
  and the TOML X values were world X from orthophotos. Converted all rigid parts below Z -10 (fingers x_first, base,
  long tubes, fittings, manifold, organizers, strips, rod, plate, SMD board, tex rects) by x += 0.014835 z
  (backup `outputs/scratch/lower/lower_pre_xconv.toml`). lower_211: back to v1 levels, fingers slightly better
  (finger_0 pts 0.58 -> 0.53, finger_1 1.01 -> 0.83), board pts 0.20 -> 0.15, board edge 2.66 -> 2.75.
- Braid B depth: new `scripts/model/lower/braidmask.py` (speckle density of contrast-normalized high-pass, dark
  pixels; isolates the braid from the smooth black chassis and gray plates) and `ray_depth_b.py` (Y stations every
  4 mm, state (dx, dz) on a +-8 mm grid = corridor, cost = V-shaped mask distance summed over 14 train views, L1
  smoothness 15 per mm; refuses holdout). Mean cost per view -3.0 -> -5.8, no station at the corridor edge. Result
  moves the cable up 2-6.5 mm and -X 1-4 mm over Y 130-300, agreeing with the interrupted knot fit (kz200 +6.75,
  kx200 -4.5). Debug sheet `outputs/scratch/lower/braid_b_viterbi.jpg`. Old offsets (dx, dz, knots) zeroed.
  Braid A: the mask does not isolate the teal braid (cost +14.8 per view, 43 percent of stations at the corridor
  edge): braid A geometry not changed.
- Train ID pass refreshed twice (`outputs/runs/lower_idtrain`, after the frame switch and after braid B moved);
  every flat texture re-baked with its 1B session, braids re-baked. Backup of the v1 textures:
  `outputs/scratch/lower/tex_bak_v1/`. bake_lower.py: `world()` now applies dzdy + the tray frame; new
  `--fill-session` (texels the primary session never saw come from a second session).
- lower_212 (braid B moved + re-bakes): braid_b pts 2.29 -> 2.06 mm, edge 3.20 -> 3.18, color res 49.1 -> 49.5.
- lower_213 (A/B): flat textures from all train sessions vs their 1B session: board color res 46.5 -> 52.6, fingers
  35.9 -> 41.3, manifold SSIM 0.276 -> 0.214, plate_l 0.231 -> 0.180, others tie: kept the 1B sessions. Braid A
  S1 + S3 fill 47.0 -> 51.7 (kept all); braid B S1 + S3 fill 49.5 = 49.5, edge 3.18 -> 3.03: kept S1 + fill
  (`lower_braid_b_s1f3.png`, one session where seen). Texture sessions now: board, fingers, OE, floor, org_l,
  strip_r, manifold, plate_l S3; org_r, strip_l S1; braid_a all train views; braid_b S1 (fill S3).
- lower_214: OE port boots (white ribbed strain reliefs at each fiber's port, IMG_5749 crop: 4-5 mm long, twice the
  fiber width; `oe_boots_l/r`, r 1.8, 4.5 mm): edge 0.96 / 1.16 px, pts 0.87 / 0.69 mm: kept. Fiber exits with a
  later dive (blue train points stay at Z -28..-33 to Y 315): pts r 3.36 -> 2.27 but edge +0.6..0.8 px and color
  res +3..10 on both sides: reverted.
- Connector latch: train points above the connector (196) give a top of 12-17 mm over X 22-62 and nothing that
  rises above it; no latch geometry modeled (no evidence at this resolution).
- Crossbar clip: named bounds, `fit_params` (`outputs/fits/lower_clip_bar`, 7 train views, edge loss 1.98 -> 1.51):
  X 37.5..69.5, Y 340.75..363.5, top 20.5. lower_215: clip_bar edge 3.60 -> 2.80 px, pts 1.62 -> 1.24 mm.
- Final lower_215 vs v1_probe (`DevLog-002-lower_parts_wave2_20261006.txt`): mean over lower parts edge 1.807 ->
  1.769 px, color res 53.05 -> 52.98, median of part pts medians 1.052 -> 1.037 mm. braid_b 3.17/2.27/48.8 ->
  3.05/2.06/48.7 (SSIM 0.199 -> 0.186), braid_a 1.80/2.11/47.3 -> 1.77/2.02/47.1, clip_bar 3.63/1.63 ->
  2.80/1.24, board pts 0.20 -> 0.15, tex_board pts 0.50 -> 0.27.
- Removable stale files (previous Phase 2 variants on the old braid B path): `assets/textures/lower_braid_{a,b}_
  {S1,S3,all}.png`, `lower_braid_b.png` (all, new path, unused); scratch backups in `outputs/scratch/lower/`.


## 5e. Wave 3 (2026-10-06): details

Coordinator brief: floor texture, braid A depth with another cue, fiber detail, small missing details.
TODO
- [x] Floor: see-through ID-owned bake (fibers/boots/strips count as owned), S3 vs all sessions
- [x] Braid A depth: teal cue, corridor Viterbi; front bay kept, middle bay reverted
- [x] Fibers: per-fiber routes traced in orthophotos (port, strip crossing, organizer entry)
- [x] Board screws tested (disabled: board-area error up); connector/clip details: no new evidence
- [ ] Next: fiber exits under the crossbar (two tries lost), floor shadows, braid A middle bay with a better cue

Progress
- `bake_lower.py --see-through a,b`: object names/prefixes whose ID pixels count as owned. Floor holes 0.54 ->
  0.34 (S3) / 0.28 (all). lower_301 (S3): tex_floor color res 59.3 -> 59.1, SSIM 0.191 -> 0.200; lower_302 (all
  train sessions): 59.9 / 0.188. Kept `lower_floor_st.png` (session S3, see-through).
- Braid A: `braidmask.teal_mask` (G and B above R, G close to B; blue fibers have B >> G) AND relaxed speckle
  density; `ray_depth_b.py --cue teal`. Front bay Y 136-332 (15 train views, +-8 mm): cost per view 13.7 -> 6.1,
  22 percent of stations at the corridor edge; with +-12 mm the interior is identical (only the station before the
  crossbar keeps moving). Sparse points within 6 mm of the tube: 681 -> 732, radial residual medians closer to 0
  in 6 of 9 bins. Re-baked (all sessions, new ID pass). lower_301: braid_a edge 1.77 -> 1.55 px, pts 2.02 ->
  1.91 mm, color res 47.1 -> 46.3: kept. Middle bay Y 390-560 (lam 25): lower_305 edge 1.55 -> 1.48 but pts
  1.91 -> 1.99 and color 46.3 -> 50.6: reverted (TOML from `outputs/scratch/lower/lower_304.toml`, texture from
  `tex_bak_w3a/`).
- Fibers: orthophotos on z = -36.5 of IMG_5725 (S1, 9 px/mm, `outputs/scratch/lower/ortho_fib_l/r.jpg`); each
  fiber's x at its port, the strip crossing (Y 240) and the organizer entry, world -> tray-local x (-0.5 mm), as
  `fibers.trace_l/r`. Organizer entries form two clusters of 4 (left -55..-43 and -33..-19), not an even spread.
  Two right-group ports are partly hidden by braid A (model values kept). lower_301: fibers_l edge 1.62 -> 1.44,
  fibers_r 1.93 -> 1.68, pts r 2.49 -> 2.20 (l 1.55 -> 1.60), boots pts 0.87/0.69 -> 0.69/0.49. Strips and
  organizers re-baked on the new ID pass (lower_302): tex_strip_r color 57.1 -> 55.5, SSIM 0.116 -> 0.295;
  tex_strip pts rose (1.73/1.58 -> 2.08/2.75; points of real fibers over the strips now nearest to the strip quads).
- Fiber exits: IMG_5732 overlay shows the exits leaving the organizers about level and passing under the crossbar;
  the lower_214 dive variant already lost; not changed.
- Board screws: train points at all six holes reach 1.4-4.8 mm above the top (p90), bright. Cylinders r 3.5, h 2.5:
  pts 0.76 mm, but metallic color res 75.7 (lower_303), matte gray 58.6 (lower_304); board area squared error
  (tex_board + screws, probe) 1101 -> 1109 and tex_board edge 1.34 -> 1.41: disabled in the TOML (code kept).
- Board, fingers and OE re-baked (S3) after braid A moved over them: tex_board 46.7 -> 46.6, fingers 35.3 -> 35.1.
- Final lower_306 vs lower_215 (`DevLog-002-lower_parts_wave3_20261006.txt`): braid_a 1.77/2.02/47.1 ->
  1.55/1.91/46.3; fibers_l edge 1.62 -> 1.44; fibers_r 1.93/2.49 -> 1.68/2.20; tex_floor SSIM 0.191 -> 0.201;
  tex_strip_r SSIM 0.116 -> 0.295. Mean over lower parts flat (edge 1.769 -> 1.771, color res 52.98 -> 53.18:
  +-2..4 shifts on small unchanged parts such as oe_7 and ribbon_a_front).
- Backups: `outputs/scratch/lower/tex_bak_v1/`, `tex_bak_w3a/`, TOML copies `lower_2xx/3xx.toml`.

## 5f. Wave 4 restart (2026-10-06)

Rollback (audit DevLog-005, section 3.2): the previous Wave 4 lower agent (01:04-01:17) diagnosed parts on holdout
views (per-part holdout error, holdout crop sheets) and acted on them (rod_l fit, plate_l see-through re-bake).
Its work is discarded: `config/model/lower.toml` = end-of-Wave-3 `outputs/scratch/lower/lower_306.toml` (checked
identical 01:20); every texture the TOML references predates 01:04. Its holdout-derived files and the tainted TOML
are not opened by this agent. Code left from that session (build.py 01:10, bake_lower.py 01:14, braidmask.py,
fit_rod.py 01:07: generic options such as a strap clip, rod dx/dz, a "stem" key) is inert unless a TOML key
enables it; no Wave 4 conclusion is reused. Unreferenced files from it (`lower_plate_l_st.png`,
`lower_floor_st_fill.png`, runs lower_401-403) are ignored. Targets below come from the v2 probe table/budget and
own runs on train lists only.

TODO
- [x] Rollback recorded; fresh train ID pass (others' occluders: only braid A middle-bay pixels differed from the old pass)
- [x] Fiber exits under the crossbar: level ribbons (kept, mixed photometric score, see below)
- [x] Floor shadows: 6 regions, session per region, out-of-sample check
- [x] Braid A middle bay with another cue (sparse-point circle fits + silhouette fit): no change
- [x] Worst probe parts: board (3 regions), SMD board (new texture), connector/plug geometry, org_r session; braid B,
      manifold, plate_l, org_l, fingers tested, no change
- [x] Every keep checked on probe and list B; texture keeps also out of sample (bakes that exclude the scored list)
- [ ] Next: crossbar lower lip (section 7, Wave 4 request), rod_l (points inconclusive), manifold/fingers (view-dependent metal)

Method. List B = IMG_5714, 5723, 5730, 5732 (S1), 5744, 5745 (S2), 5752, 5753 (S3): train views, not probe, chosen by
lower-part pixel counts in the ID pass. Score = absolute squared error per part after the per-view color fit, blur 4
(`outputs/scratch/lower/w4/sse.py`, refuses holdout runs; 1e6 units), plus parts.json edge / points. Probe and B are
also bake views, so texture keeps were re-checked out of sample: `bake_lower.py --exclude <list>` (new; excluded views
are treated like holdout) bakes every variant without the scored list's views. All tables:
`outputs/scratch/lower/w4/evidence_wave4_20261006.txt`; final per-part table
`DevLog/parts/DevLog-002-lower_parts_wave4_20261006.txt` (lower_540base | lower_540 | same on B). bake_lower.py now writes
`outputs/textures/<stem>_views.json` per bake (session, fill, see-through, exclude, ID run, primary-session views).

Progress
- 01:25 Baseline lower_500 / 500b (= lower_306 state). Lower SSE probe / B: floor 1697 / 2399, board 1083 / 674,
  manifold 360 / 451, fingers 315 / 642, braid_b 292 / 409, smd_board_l 235 / 652, connector 213 / 74.
- Fiber exits: IMG_5733 crop (s2) shows about 8 parallel fibers per side leaving the organizers' rear faces level and
  passing straight under the crossbar. rayplane z = -30 on IMG_5733 (near top-down, X insensitive to Z): bundle L X
  -48.4..-31.7, R 57.2..72.7; blue train points Y 295-310 at Z -31.7..-27.4, X medians L -36, R 67-74. New
  `[fibers.exit_ribbon]` (n 8, r 0.8, span 14, xc -38.9 / 66.5 local, z -29.5, Y 299-334); build.py keeps the old
  dive when the table is absent. Overlays in 5732/5733/5730/5734/5729 match the ribbon position. A/B with the new
  floor fixed (512 vs 512x): pts exit_r 3.36 -> 2.07, exit_l 2.05 -> 2.08 mm; SSE exits+floor probe 1795 -> 1851
  (worse), B 2392 -> 2344 (better); edge exit_l 3.98 -> 4.51 px. y1 = 320 lost on both (floor shows through).
  Kept on geometry (photo crops, points); the probe loss is the fibers drawn where the photo shows a crossbar
  surface at Y about 315-331 that the model lacks (section 7, Wave 4 request). Rename the table to revert.
- Floor: split into 3 x 2 regions (X -85.58/-20/45/111.42, Y 205/262/330), see-through bakes (fibers, boots,
  strips owned) per session with fill on ID pass lower_idtrain5. In-sample choice (probe + B) per region; the probe-only
  and B-only choices agree except r10. Out of sample (bakes without the scored list): mix vs all-S3, B 2101 -> 1951,
  probe 1633 -> 1642; r10 back to S3 (out-of-sample probe 407 vs S1 429). Kept r00 S1, r01 S1, r02 S2, r10-r12 S3
  (fill S3 / S1+S3 / S1): final floor SSE probe 1720 -> 1590, B 2436 -> 1946.
- Braid A middle bay: train-point circle fits (fixed r 5, 20 mm bins, Y 380-600) give shifts under 3 mm with
  residual medians 0.8-1.4 mm; free-radius fits give r 4.3-5.9 (r 5 fine). New knot offsets mx/mz at Y 400/470/540
  (tubegeom, taper 25 mm); fit_params on the 4 train views that see it outside probe/B (5727, 5733, 5741, 5743): edge
  loss 1.58 -> 1.40, mx400 +5, mx470 -4, mz +2; points disagree (residual at Y 400 0.90 -> 2.61). A/B: edge 1.55 ->
  1.76 (probe), 1.49 -> 1.69 (B), pts 1.85 -> 2.02, SSE 93 -> 129 / 48 -> 91: rejected, knots stay 0. The points say
  the middle bay is already within about 1 mm.
- Sessions of the other surfaces (re-bakes S1/S2/S3 on lower_idtrain5, in-sample on both lists, then out of sample):
  manifold S1 won in-sample (366 -> 354 / 471 -> 413) but lost out of sample (460 -> 547 B, 374 -> 484 probe):
  rejected; org_r S2 kept (out of sample 228 -> 137 B, 161 -> 120 probe; final 135 -> 120 / 160 -> 128); plate_l S1
  and org_l S1 rejected out of sample; fingers, oe: S3 stays (L/R finger split tested, S3 best in both, no gain:
  reverted to the single texture). Braid B: re-bakes S1+S3 fill / S2 / S3 / all vs current: 224 / 213 / 220 / 217 /
  214 probe, 324 / 325 / 355 / 375 / 332 B: no change.
- Board top: 3 X bands (-100/-33/34/101). x0, x1 S3 on both lists; x2 S2 (probe 276 -> 220, B 164 -> 122); out of
  sample x2 S2 vs S3: B 154 -> 125, probe 273 -> 234. Kept S3 / S3 / S2 (S2 has 2 primary views, fill S1+S3).
- SMD board: IMG_5723/5730 show a gray PCB with pads where the model is flat black. New `[[tex]] smd` (top quad):
  S3 beat S1 on both; out of sample S3: B 672 -> 372, probe 238 -> 121 (flat -> textured).
- Connector: train points (6 x 8 mm bins) put the housing top at 14-15 over X 22-64, Y 92-100; X 70-86 is the white plug
  (bright, top 8.4) and tape. Connector X 22-88 -> 22-66 (Y 89-107 kept; Y1 102 tested: edge 3.05 vs 1.92 px), plug
  X 70-82 / Y 82-97 / top 6 -> X 69-85 / Y 83-95 / top 8.4. Connector edge 2.52 -> 1.98 (probe), 3.19 -> 2.60 (B), pts
  0.82 -> 0.75; plug pts 1.66 -> 0.73; connector + plug + clip_board SSE 268 -> 148 (probe), 105 -> 79 (B). Board
  re-baked on a fresh ID pass (lower_idtrain8) after the connector shrank.
- rod_l: train-point circle fits per 25 mm bin, n 11-23, shifts -1.3..+3.4 mm without a trend: no change.
- Final back to back (lower_540base = lower_306.toml vs lower_540, full mode; lower total SSE): probe 6512 -> 6173
  (-5.2 percent), B 8899 -> 8286 (-6.9 percent). Report-level probe: color residual 37.7 -> 37.6, points median 1.14 ->
  1.13 mm. Worse parts: fibers_exit_l/r (137 -> 265 probe, 65 -> 391 B; offset by the floor), plug (23 -> 31 / 2 ->
  15), clip_board probe (35 -> 51). Per-part edge means are not comparable in total (8 more parts).
- Texture sessions now (assets/textures/): board x0, x1 S3 (fill S1), x2 S2 (fill S1+S3); floor r00, r01 S1 (fill S3),
  r02 S2 (fill S1+S3), r10-r12 S3 (fill S1); org_r S2 (fill S1+S3); smd S3 (fill S1); unchanged: fingers, oe, org_l,
  manifold, plate_l S3, strips S1/S3, braid A all, braid B S1 + S3 fill. Unused variants and intermediate runs deleted;
  ID passes kept: lower_idtrain5 (floor, org_r), 7 (smd), 8 (board; matches the final model).

## 5g. Wave 5 (2026-10-06): visible details toward v4

Coordinator rule: accept a detail if neutral on probe AND list B, reject if it regresses both. Back-to-back A/B
script `outputs/scratch/lower/w5/ab2.sh` (old TOML then new TOML, probe and list B, flags edits by other groups during
the A/B). Run names must differ by more than case: on macOS `lower_601B` and `lower_601b` are the same folder (the first
A/B was lost that way and repeated). A first A/B was also confounded by concurrent chassis edits (crossbar lip toggled
between the halves): repeated until the unaffected parts matched.

TODO
- [x] Individual fibers: already 2 per OE (8 per organizer, traced in Wave 3); fibers thinner (r 1.1 -> 0.8)
- [x] OE port boots ribbed (core + 5 rings)
- [x] Board screws enabled at the gold pads measured in the baked board texture
- [x] Finger rear-block screws (8)
- [x] Braid textures at 8 px/mm (sharper weave)
- [x] Ribbon fan at the connector: 16 wires
- [ ] White clips' true shape, plug opening, connector latch, OE port latches: no measurable evidence (sparse points mixed
      with the bracket and fingers; shapes not resolved in the crops); not modeled

Progress (SSE 1e6 on lower parts, probe / list B; old -> new back to back)
- Fibers r 0.8 (IMG_5726 s2 crop: about 10 px wide at about 6.5 px/mm = 1.5 mm) + ribbed boots (`boot_ribs` 5, core r 1.3,
  rings r 1.8): total 4931 -> 4915 / 6489 -> 6464; fibers_l 133 -> 97, fibers_r 95 -> 70; floor r00-r02 +8..15 (baked
  see-through fibers now show beside the thinner tubes). fibers_l pts 1.60 -> 1.89 mm. Accepted.
- Board screws (r 2.8 head, h 2.5): first at the 1A positions (+12 / +5, neutral), then moved to the gold pad centers
  found in the board texture (X -94.8 / 0.3 / 94.8, Y 65.9 / 103.1; 1A had -88 / 0 / 96, 63 / 101): screw edge 2.49 ->
  1.31 px (probe), 2.35 -> 1.18 (B), SSE neutral. Accepted.
- Braid textures: single-session or n_best 1-2 at 8 px/mm regress both lists (braid_b 162 -> 194..208 probe, 222 ->
  327..371 B): rejected. Same statistic as before (median of best 5) at 8 px/mm: braid_b 162 -> 152 / 222 -> 222,
  braid_a 90 -> 90 / 49 -> 49: accepted (files `lower_braid_a_w5z.png`, `lower_braid_b_s1f3_w5z.png`).
- Finger screws (dark copper head r 1.7 at block center Y 161; texture dark spots at Y 160.8-161.3, X within 2 mm of the
  finger centers): 4922 -> 4922 / 6468 -> 6467. Accepted.
- Ribbon fan 9 x r 1.3 -> 16 x r 0.85 (about 16 wires of 1.7 mm in the crop): neutral, pts 0.47 -> 0.24. Accepted.
- Final back to back (lower_610_old = Wave 4 end TOML, lower_610_new): lower SSE probe 4927 -> 4910, B 6487 -> 6467
  (other groups unchanged during the run). Per-part table `DevLog/parts/DevLog-002-lower_parts_wave5_20261006.txt`.
  Note: totals are lower than in Wave 4 since a concurrent chassis edit (02:33, most likely the requested crossbar lip)
  now hides most of the fiber exits and part of the floor (exits 135 -> 14 on probe between two otherwise equal runs).

## 6. Notes for other groups

- Upper: no lower copper crosses Y 330-350 (see section 4). Braided cable A is lower's over its full length,
  including the rear wire fan to Y 616 at X 52-76, Z about 4 (rear-bay connector itself not modeled by lower).
  Braid B continues under the crossbar into the middle bay to Y 490 (X -99, Z -34) where it is lost.
- Chassis: lower's front-bay parts sit inside X -116..112; the ducts (X 140-217) are untouched.

## 7. Requests to coordinator

- (Wave 2, done) World frame tilt: board and crossbar-top plane fits give dz/dx about 0.011-0.012 and dz/dy about 0.005-0.007
  (0.65 deg / 0.35 deg). If the world frame is re-leveled, set `[tilt]` in lower.toml to zero (or to the
  residual) and re-check; lower's local Z values assume the current frame.
- (Fixed in the shared tool by 2026-10-06; bake_lower.py keeps its local copy) Shared tool: texture_from_photos.sample_view opens `source/images/<stem>.jpeg`; the files are `.JPG`
  (2026-10-05). Fix in the shared tool (glob by stem); bake_lower.py carries a local workaround.
- (Wave 4, 2026-10-06) Chassis: in IMG_5733 (rayplane z -30) the fiber exits vanish under a gray surface at Y
  about 315-318, and in IMG_5729/5730/5732/5734 the photo shows crossbar metal about 40-60 px (s4) beyond the model's
  crossbar front edge over X -50..75. The crossbar front flange (flange_depth 30) probably has a forward lower lip or
  is slanted; please check. Until then lower's level fiber ribbons (to Y 334) are visible there and cost probe SSE.
- Re-bake order: lower textures are ID-owned against the current model; re-run bake_lower.py (after an ID render
  of the train views) when braid A/B, the cable loop or other groups' occluders over the front bay move.
