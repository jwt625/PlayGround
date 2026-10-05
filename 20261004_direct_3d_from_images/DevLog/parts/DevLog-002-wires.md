# DevLog-002-wires: cables, external board, white cable, tray loops

| field | value |
|---|---|
| group | wires (agent: wires modeling agent) |
| owns | `scripts/model/wires/` (build.py, fit_wire.py, ray_depth.py, traces.json), `config/model/wires.toml` |
| phase | 1A done 2026-10-05; 1B pass 1 done 2026-10-05 |
| last run | `outputs/runs/wires_202` (probe, all parts, full) |
| objects | 30: wires.bundle_{red,yellow,orange,green,brown}, wires.ext_board, wires.label_ext_board, wires.ext_pot_{1,2}, wires.label_ext_pot_2, wires.ext_strip, wires.ext_conn4, wires.ext_conn4_pins, wires.white_single_{a,b}, wires.white_pair, wires.loop_{yellow,blue,red,brown}, wires.loop_tie, wires.hv_cable, wires.grey_cable, wires.grey_inner, wires.red_{a,b,c,d,e} |

## TODO
- [x] External tan board (plane fit + corner picks) with two trimmers
- [x] Five-wire colored bundle: pcb.conn_b header -> over -Y rim -> arc (apex z ~45-52) -> board solder holes
- [x] White cable: upright 4-pin housing, two single wires back over the +Y rim, flat pair with lifted free end
- [x] Four colored loops inside the tray (yoke leads -> header near pcb.conn_c), rough
- [x] Photo-sampled base colors; PVC Specular IOR Level 0.3 (coordinator rig 2026-10-05)
- [x] 1B: white singles routed down behind the heat-sink plate (rim crossing (40-41, 47.6, 29) -> (41, 46, 10)); end point assumed
- [x] 1B: loop starts at the yoke terminal block (stub exit ~(52.3, 20.3, 22.5)) via the cream cable tie (wires.loop_tie)
- [x] 1B: ext board texture (texture_from_photos, 12 train views, ID occlusion); pot_2 top texture; ext_strip under the board
- [x] 1B: 4-pin housing slots + pin contacts; white_pair as a flat two-conductor ribbon; red dashed / x stripe shader
- [x] 1B: cables over the main board: hv_cable, grey_cable + grey_inner, red_a..red_d (coordinator request)
- [x] 1B: holdout leak fixed (IMG_1595, IMG_1609 removed; all affected solves/fits re-run on train views)
- [ ] 1C: pot_1 top texture (bake was all wire occluders; removed), pot hex nut / shaft shape
- [ ] 1C: white_single_a still fp 0.21 (path near the rim/plate); single wire ends inside the tray unmeasured
- [ ] 1C: housing slots invisible at 1/4 scale (uniform color); housing pose edge 3.2 px
- [ ] 1C: more red loom wires over the board (only 4 traced); red_d pts median 2.3 mm (mask cost high, path rough)
- [ ] 1C: grey_cable edge 4.5 px (bezel end at x 3.5 is a guess; may pass under the bezel)

## Method (provenance)
1. Sparse points (data/points_world.npy) filtered by HSV classes outside the case footprint: board brown cluster
   (x 100-130, y -40..10), white cable cluster (x 20-125, y 55-100, z 1-28), bundle reds/oranges (x 47-112,
   y -53..-70, z 28-38).
2. External board: RANSAC plane on brown points (275/398 inliers, 1 mm): centroid (119.7, -16.6, 5.3), normal
   (-0.175, -0.119, 0.977), tilt 12.2 deg. Corners by ray-plane from IMG_1629 and IMG_1606 picks (agree within
   ~2 mm). Check: tri2 of solder holes IMG_1629/IMG_1603 -> (119.83, -7.22, 6.54) err 1.8/2.4 px,
   (122.88, -12.71, 6.67) err 0.2/0.3 px, (128.44, -22.54, 6.15) err 1.1/1.4 px; plane predicts z 6.4/6.3/6.1.
3. Solder holes (wire ends), ray-plane from IMG_1580: brown (126.2, -18.0), orange (123.0, -12.3), yellow
   (119.4, -7.4), green (116.6, -1.7), red (114.3, 0.6) -> TOML ends use the IMG_1629 values (within 1 mm).
4. Bundle start: pcb.conn_b top (x 10.5-24.9, y -47.9..-40.5, z 15; pcb agent), 5 pins at 2.5 mm pitch.
   Epipolar check of the header in IMG_1560/IMG_1550: wire entry at z ~12 at (17.8, -47.5).
5. Bundle and loop paths: 2D traces read from labeled crops of the near top-down IMG_1580 (traces.json); depth
   along each pixel ray by `ray_depth.py` (Viterbi over z, color-mask distance in 6 side views: 1587, 1590,
   1513, 1511, 1609, 1614). Cross-check: brown apex z ~52 by epipolar crop IMG_1580 -> IMG_1587; solver gave 52.5.
   Then `fit_wire.py` (free 3D control points, ends fixed, color-mask distance in 6-8 views, ROI 15-20 px,
   regularized) for red/yellow/green/orange/brown: moves 0.3-6.7 mm. loop_blue fit rejected (snapped to clutter,
   fp 0.10 -> 0.24; reverted); loop_yellow fit not applied (10.5 mm jump onto the yoke tape).
6. White wires: IMG_1580 traces; depth from white masks in mat-background views (1587, 1590, 1595, 1625, 1626,
   1588); rim crossing fixed by hand at (40-41, 50, 29) (rim top 28 + r); then fit_wire (8 views).
7. 4-pin housing: pick.py tri (after coordinator fix) (125.63, 62.66, 3.2) 4 inliers, (123.71, 60.82, 11.75)
   4 inliers -> upright box; fit_params (cx, cy, cz, yaw; 8 views, 60 evals) loss 7.47 -> 2.00:
   (121.0, 64.0, 8.5, yaw 35); `outputs/fits/wires_conn4/best.json`.
8. Pots: DLT of top centers IMG_1580 + IMG_1629 (err 18-47 px: rough), pot 2 supported by an orphan cluster
   (116.9, -21.1, 10.1) that disappeared after the move.
9. Colors: hue-masked median sRGB along IMG_1580 traces, converted to linear (TOML comments carry sRGB).

## Progress
- 2026-10-04: read guide; sparse-point survey; board plane; scratch overlay/epipolar tools (deleted).
- 2026-10-05: first TOML (centerline bundle) -> overlays showed bundle far too low/close; measured apex.
- 2026-10-05: traces + ray_depth for bundle, white wires, loops; wires_001: bundle edge 2.2-2.6 px,
  pts 0.4-1.5 mm; ext_conn4 fp 0.49 (misplaced).
- 2026-10-05 02:39 (last): wires_002: conn4 fit (fp 0.49 -> 0.02); wires_003: white fit (single_a fp 0.25 -> 0.15,
  single_b 0.21 -> 0.05); wires_004/005: bundle fit (edge 1.8-2.5 px, fp <= 0.08); wires_006: pots moved
  (pot_2 fp 0.19 -> 0.02); wires_007 (full): final 1A numbers below.

## Measurements: wires_007 (probe, 12 views)
| part | fp | edge px | pts med mm |
|---|---|---|---|
| bundle_red / yellow / orange / green / brown | 0.005 / 0.029 / 0.003 / 0.029 / 0.080 | 1.87 / 2.19 / 1.75 / 1.82 / 2.50 | 0.79 / 0.35 / 0.38 / 0.56 / 0.81 |
| ext_board / pot_1 / pot_2 / conn4 | 0.005 / 0.000 / 0.024 / 0.020 | 2.49 / 3.63 / 2.93 / 2.72 | 0.34 / 0.81 / 2.04 / 0.84 |
| white_single_a / single_b / pair | 0.145 / 0.054 / 0.102 | 3.14 / 2.79 / 2.36 | 0.87 / 1.51 / 0.69 |
| loop_yellow / blue / red / brown | 0.003 / 0.098 / 0.018 / 0.021 | 1.87 / 2.06 / 2.27 / 2.25 | 1.06 / 1.17 / 1.20 / 1.49 |

## How to route a wire (pipeline used here)
1. Trace in 2D: in a near top-down train view (IMG_1580 for most wires, IMG_1560 for the board cables), read
   10-20 points along the wire from `pick.py crop` grids (1/2-scale px) into `scripts/model/wires/traces.json`
   (`view`, `uv`; optional `prepend_mm` / `append_mm` fixed 3D anchors: connector pins, solder holes, rim crossing).
2. Depth along the rays: `uv run python scripts/model/wires/ray_depth.py WIRE --views V1,..,V6 [--sparse]
   --zmin --zmax --smooth 1.0 --write`. Each trace pixel fixes a ray; z per point is picked by Viterbi over a
   0.5 mm grid with cost = color-mask distance (fit_wire.HSV) in 5-6 side views (+ sparse-point distance with
   --sparse) and an L1 smoothness term. Use views with a plain background behind the wire (mat).
3. 3D refine: `uv run python scripts/model/wires/fit_wire.py WIRE --views ... --fix-ends --roi 8 --max-move 4
   [--debug sheet.jpg] --write`. Least squares on all control points (ends fixed), color mask restricted to a
   corridor of --roi px (1/4 scale) around the current projected path, each point bounded to --max-move mm. The
   corridor + bound keep it from jumping onto same-colored clutter (yoke tape, blue caps).
4. Check the debug sheet and the next iteration's parts.json (fp, edge, pts med, pts >4 mm); revert a wire whose fp
   rises (done for loop_blue twice). Smooth zigzags by hand when a fit adds kinks (hv/grey cables).
5. Both scripts refuse holdout views (data/split.json).

## Phase 1B (2026-10-05)
- Holdout leak (coordinator audit): 1A used IMG_1595 (white wires, fit_wire bundle) and IMG_1609 (ray_depth
  bundle). Re-run on train views only: bundle ray_depth (1587, 1590, 1513, 1511, 1606, 1614) + guarded fit
  (1580, 1587, 1590, 1513, 1511, 1614, 1560, 1594); white wires ray_depth (1587, 1590, 1594, 1625, 1626, 1588)
  + fit; ext_conn4 fit_params re-run without 1595 -> (120.5, 64.25, 8.25, yaw 23.75), loss 2.36. Guard
  `check_views` added to both helper scripts.
- Orphan (69, 118, 10): its sparse points are mat blue (hue 0.57, sat 0.6-0.9); IMG_1587 (2061, 1415) is the
  mat's molded rib, not the cable. White cable not moved for it (env / mat geometry).
- Yoke leads: epipolar IMG_1560 -> IMG_1580/1590: lead exit at the yoke terminal block z ~22-23, cable tie at
  (47.0, 23.4, 21). Header ends at pcb.conn_c top (z 13.5).
- ext_strip: epipolar IMG_1603 -> IMG_1600/1601 puts the strip center at z ~11-12, x ~97.6-98.6 (case +X wall
  face); y -11..11. Placed at x 98.8.
- Pots: centers moved to the texture blob centers (106.4, -8.3) and (116.2, -23.4); pot_2 top baked; pot_1 bake
  showed only wires crossing over it (deleted).
- Board cables (IMG_1560 traces, ray_depth --sparse in 1590/1626/1588/1614/1513, fit with corridor): hv_cable
  r 1.5 cream from pcb.hv_boot -X end (50, -45.5, 22.6) to under the yoke; grey_cable r 2.2 from the bezel side
  (3.5, -37.7) to (22.5, -28); grey_inner (cream) to (27.3, -19.8); red_a/b (socket board leads), red_c, red_d.
- Iterations wires_101-108 (8, plus 2 helper renders for texture occlusion and the stripe test).
- Final wires_108 (probe): global IoU 0.942, edge 3.03 px, pts median 0.38 mm. Bundle edge 1.8-2.6 px, pts
  med 0.37-0.99 mm; white_pair edge 2.06 px, pts med 0.48 mm (>4 mm 0.11, was 0.45 at 1A in the new metric);
  hv_cable 2.27 px / 0.99 mm; grey_cable 4.51 px / 0.63 mm.

## Phase 2 (2026-10-05): missing geometry
- Baseline wires_201 (probe, no --parts): error_budget "no model" 6.1 percent of total (absolute 1806e6);
  after wires_202: 7.6 percent (absolute 1774e6, -1.8 percent). The share rose because other groups cut the total
  (29436e6 -> 23354e6). Unmodeled object px 23500 -> 23488.
- Where the no-model error is: 794e6 of 1774e6 lies within 6 px of wire pixels (IMG_1624 187, 1604 167, 1560 133,
  1587 76, 1527 75). Zoom sheets (photo | render | cyan missing + red wire outline) show it is almost all cast
  shadows of the white cable on the mat and blur halos, not missing wires; the rest is case/mat shadows and lips.
  No unmodeled wire runs were found in the 12 probe views beyond the items below.
- Red loom: IMG_1560 s2 crops show two parallel red leads under the HV cable; red_d re-traced (upper) and red_e
  added (lower), z 9-13 by ray_depth --sparse + corridor fit. The other reds over the board are red_a/b (socket
  board bottom leads) and red_c (top lead); the loop_red is the yoke lead.
- white_single tails: IMG_1627 shows the wires going over the rim and down outside the plate (the "down" part
  seen there is partly their reflection in the glossy wall); inside the tray nothing is visible (IMG_1588).
  Tails now drop between the wall and the plate at y 48.6 to z 16 (hidden).
- pot_1 top texture: best train view IMG_1622 shows the green bundle wire across the top and the pot only
  partly; no clean view exists, so no texture (the 1B bake was removed).

## Notes / issues
- The external board's -X edge reaches x 96 (z 1.4-2.8), inside the case outer wall (+X end at ~98.9 from
  case.toml). Measured from two views; either the case +X end or the board edge is ~3 mm off. Not resolved.
- Board regions: remaining cyan is dark shadow under the board and a strip near the case end (possibly a second
  board piece seen in IMG_1603/1604), not modeled.
- zsh: `for a in "V u v"; pick.py tri $a` does not word-split (use `${=a}`); this explained early empty tri runs.

## Requests to coordinator
- New objects over the main board for the pcb re-bake (ID ownership): wires.hv_cable, wires.grey_cable,
  wires.grey_inner, wires.red_a, wires.red_b, wires.red_c, wires.red_d, wires.loop_*, wires.loop_tie.
- The ext_strip sits at the case +X wall face (x 98.0-99.6): overlaps the case outer wall if the case is right.
