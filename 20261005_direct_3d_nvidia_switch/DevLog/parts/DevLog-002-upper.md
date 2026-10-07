# DevLog-002-upper: middle bay, rear bay, rear UQD fittings, copper tubes

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | Phase 1A done; Phase 1B (top-face photo textures) done; Wave 2 in progress (2026-10-06) |
| Owns | `scripts/model/upper/build.py`, `config/model/upper.toml`, this devlog |
| Runs kept | `upper_001`, `upper_009` (1A), `upper_013`/`upper_014` (1B), `upper_015`..`upper_022` (Wave 2), `upper_idtrain` (ID pass, train views) |

## 1. TODO

- [x] Data-driven builder: TOML entries `[[box]]`, `[[prism]]`, `[[cyl]]`, `[[cyls]]`, `[[tube]]` (polyline with
      filleted corners, POLY curve) -> one object `upper.<name>` each; colors stored as photo sRGB, converted to linear
- [x] Rear bay: board, serpentine cold plate + tube, board2 (two nickel plates, coin cell, connector), -X board,
      ribbon cable, power connectors, cable bundle, L-blocks (+ tabs), hoses, rear corner bracket
- [x] Middle bay: main board, ASIC cold plate, copper block + QR block, tubes, elbow, clamp bar/arms/pads, rear
      arms, fin block, two connector spreaders + blue wire bundles, connector row, two capacitor groups,
      two black baffles (prisms), +X wall manifold block + 2 copper tubes, -X wall block, rear bracket, components
- [x] Rear UQD: block + cylinder per side (rough, see open issues)
- [x] UQDs: two bodies per side, stacked in Z (Wave 2, section 9)
- [x] Baffle_px top: plane z = -32.82 + 0.0176 x + 0.0225 y (Wave 2; near-flat, -21 -> -17.5 along Y)
- [x] Teal cable at X -100, Y 378-446, Z -33..-35 (Wave 2)
- [x] Phase 1B: top-face photo textures for 17 entries (`bake_upper.py`, ID-owned per-texel median/p25 of the best
      5 train views) -- reconstructed in section 8
- [x] Wave 2: textures on 25 more entries (clamps, pads, connector row, components, wires, blocks, L-blocks,
      brackets, rear_pcb_left, pwr_conns, UQD blocks); session per texture by full-mode A/B (section 9)
- [x] Wave 2: baffle_px plane top; UQD bodies re-measured (stacked in Z); teal cable added
- [x] Wave 2: side-face textures on 23 boxes (`tex_sides = true`); serpentine plate thinned to 4 mm
- [x] Wave 2: baffle_px plane refit without capacitor tops (first fit was wrong); +X wall tubes trimmed
- [x] Second curved duct edge in baffle_nx: measured (low step near z -32), rib tried, neutral, skipped
- [x] Wave 3: session/stat variants per texture; cylinder side textures (UQDs, caps_px); baffle error analysis
- [ ] Wave 3 open: mid_pcb +X wall strip (geometry at the wall; request filed); component geometry details
- [ ] rear_pcb_left (pts median 4.7 mm): train points at (-195, 625) give a top near -11 (now -12); the rest of
      its points are cables/connectors above it; not changed (weak evidence)

## 2. Method (what worked)

- Sparse-point height modes per XY box (2 mm histogram bins) for every surface height; region stats by color class
  (copper, dark) for tubes and baffles.
- Orthophotos of train views on planes z = const (`outputs/scratch/upper/ortho.py VIEW Z x0,x1,y0,y1 ppm`, 25 mm
  grid) for XY outlines; use the plane at the part's own height (parallax otherwise: rear bay is about 60 mm
  above the middle bay board). IMG_5749 is the best top view; 5713/5748/5753 agree to 1-3 mm at the right height.
- Tube line fit: grid search of a straight segment's (Y, Z) maximizing copper-ness ((R-B)/sum) along its
  projection in 5-6 train views (`linefit.py`). Found the divider run at Y 582, Z 12 (I had assumed Y 600, Z 15);
  confirmed by overlay on IMG_5734.
- Per-run far-point grid (`orph.py RUN`: points > 4 mm from the model, 20 mm cells, Z percentiles, nearest part)
  from `points.npz`; this drove most edits.
- Did not work: multi-view plane sweep over the whole region (top views have small baselines; noisy), tri2 on
  blurry UQD ends (no consistent triangulation; ray gaps 10-50 mm).

## 3. Measurements (mm, tray frame)

| Item | Value | Source |
|---|---|---|
| Rear-bay board top | -3 | coin cell mode 0.5 |
| Serpentine plate | X -115..22, Y 600..767, top 14 | ortho z16 IMG_5749; plate-only patch modes 13-15 |
| Serpentine tube | OD 6, rows Z 16, 9 rows Y 635..730, left leg X -98 Z 14 | ortho trace; line fit row Y 656 Z 15-16, leg X -98 Z 11-12 |
| Divider tube run | Y 582, Z 12, X -186..186 | copper line fit (6 views), overlay IMG_5734 |
| L-blocks | X 184..216 / -212..-182, Y 578..600, tabs to Y 627 / 612 | ortho z-6 / z2 IMG_5749, red O-ring points at X +-180, Y 582 |
| Hoses | X 206 / -205, Z 10-22, Y 612/627 -> 790 | ortho strips, side-wall point Z |
| Board2 plates | A X 25..101 Y 616..652 top 9; B X 23..81 Y 680..768 top 8 | ortho z12, plane NCC 8-9 |
| Middle-bay main board top | -48 | modes -47/-49 (X 60-115) |
| ASIC cold plate | X -41..40, Y 474..560, top -22 | QR label mode -23; ortho z-24 |
| Copper block | X -13.6..11.6, Y 503..530, top -19; tubes OD 4 at X -5.6 / 4.5 | ortho z-24 at 5 px/mm |
| Clamp | bar Y 421..440 top -34; arms Y 452..478 top -26; pads abs(X) 60..89 top -41 | sub-cell modes |
| Capacitors | 9 at X 123..152, Y 416..447, tops -34; 5 more at X 147..170, Y 522..568, tops -32 | red-top sparse points |
| Baffles | nx top -32, px top -21 (slanted) | modes; outlines from ortho z-32 / z-26 |
| +X wall tubes | X 202 / 211, Z -20, Y 400..507 (visible part) | far points Z -17..-19; copper line fit X 208 Z -18 (weak) |

Scale/frame note: plate-only patch modes on the serpentine plate are 13 (X -110) vs 15 (X +15): about 0.9 deg roll
(Z rising toward +X), same sign and size as the 0.65 deg board/crossbar tilt the lower agent reported. Not
compensated per part; waiting for the frame audit.

## 4. Connections (what connects where; replaces the Y = 350 hand-off)

Per the lower agent (relayed 2026-10-05): no copper crosses Y 320-410, the front-bay tubes end in a center
manifold at Y 298-326. So there is no Y = 350 hand-off. Removed my assumed -X gray tubes (they were a misreading
of front-bay tubes at Y 311-314) and the +X tube drop to Y 350.
- Rear loop: +X UQD -> black hose along +X wall (X 206) -> +X L-block (over the divider) -> copper run along the
  divider (Y 582, Z 12) -> inlet at X 6 -> 9-row serpentine -> left leg X -98 -> divider run -> -X L-block ->
  hose along -X wall (X -205) -> -X UQD.
- Middle bay: copper block on the ASIC plate -> two 4 mm tubes (+Y) -> copper elbow/manifold at Y 548-562 (where
  it goes from there is hidden under the rear bracket).
- +X wall: aluminum manifold block (X 183..213, Y 507..550) with two copper tubes running -Y along the wall to
  about Y 400, then hidden under the black duct/baffle; destination not visible.

## 5. Progress log

- 2026-10-05 23:05: Read guide, playbook 5-8, Phase 0 devlog. Height histograms: middle bay content Z -50..-20,
  rear bay content Z 0..+20 (rear bay is a raised level).
- 2026-10-05 23:12: upper_001: 44 parts from orthos and height modes. Sparse median (all parts) 6.6 mm.
- 2026-10-05 23:18: upper_002/003: clamp split into bar/arms/pads, baffle outlines from orthos, wire bundle and
  baffle heights, hoses, L-blocks. upper_004: divider tube run Y 582 Z 12 from the copper line fit.
- 2026-10-05 23:28: upper_005: ASIC copper block was 15 mm off in Y (ortho z-24 at 5 px/mm); elbow edge 6.6 -> 2.3 px.
  Added second capacitor group. upper_006: rear-bay side strips (L-block tabs, ribbon width, cables).
- 2026-10-05 23:35: upper_007/008: rear clamp arms, rear bracket, board components; plate top 14, tube rows 16,
  +X wall tubes at X 202/211 Z -20.
- 2026-10-05 23:40: upper_009: removed Y = 350 tube extensions per lower agent. Global sparse median 1.33 mm
  (all groups), silhouette IoU 0.947.
- 2026-10-06 00:10-00:30 (Wave 2 agent): reconstructed Phase 1B (section 8); A/B tools; fresh ID pass; 25 more
  textured entries; three-session bake A/B (upper_015-017).
- 2026-10-06 00:30-01:00: session choice per texture; teal cable; UQD re-measure (upper_020-022); erode test.
- 2026-10-06 01:00-01:30: thin serpentine plate, baffle_px refit, +X tubes trimmed (upper_023/024); rib
  (upper_025, skipped); side-face textures (upper_026 vs 027). Devlog sections 9-10.

## 6. Before / after (probe, upper_001 -> upper_009): edge mean px, sparse median mm

Selected: serp_tube 2.63->1.80 px, 2.13->1.16 mm; serp_plate 3.09->1.39 mm; asic_elbow 6.21->2.29 px;
cu_block2 3.38->0.77 mm; wires_px 5.08->1.58 mm; baffle_px 5.31->2.51 mm; hose_nx 6.63->2.28 mm;
lblock_nx 6.93->3.23 mm; ribbon 2.78->1.44 mm; hs_px 1.53->0.84 mm; mid_pcb 1.26->0.95 mm; wtube_px_a
7.51->5.70 px. Worse or unresolved: uqd_block_px 3.2->13.2 mm (few points), uqd_px 1.9->8.4 mm, rear_pcb_left
4.0->4.7 mm, cables_nx 4.4 mm, cu_block 4.18 mm in 009 with only 13 points (1.13 mm with 65 points in 005: points
now attributed to a neighbor, likely another group's object). Full table: `outputs/scratch/upper/parts_001.json`,
`parts_009.json`.

## 7. Requests to coordinator

- None blocking. Rear UQD brackets (black outboard housing at the rear corners, IMG_5741/5742) may be chassis:
  please assign.

## 8. Phase 1B reconstruction (previous agent's session was killed at 00:03; reconstructed 2026-10-06 from files)

The devlog stopped at Phase 1A; files show what Phase 1B did (TOML vs this devlog, run reports, texture md5s):
- `build.py` (23:47): `tex = "<t>"` on box/prism/cyls entries puts `assets/textures/upper_<t>.png` on top faces
  (normal z > 0.9), UV = entry XY bounding box; [[cyls]] map every cap to one instanced texture (capacitors).
- `bake_upper.py` (23:48): ID-owned bake per texel (first n_best = 5 ranked train views where the part owns the
  pixel in `upper_idtrain` ID renders), median or a percentile, optional one session; holdout never used.
- TOML (00:02): tex on rear_pcb, serp_plate, board2_a/b, board2_conn, ribbon, mid_pcb, asic_plate, cu_block,
  cu_block2, clamp_bar, fin_block, hs_nx, hs_px, caps (cap 4 sampled, instanced), baffle_nx, baffle_px;
  [texbake] options: p25 for rear_pcb, mid_pcb, cap, hs_px, baffle_px, ribbon; S1 for asic_plate, cu_block(2),
  clamp_bar, fin_block. Geometry unchanged since upper_009 except the 2nd UQD bodies (uqd_*_a/_b, brackets).
- Runs: 010 (8 more UQD/bracket objects; flat), 011 (first textures, median, all sessions: color residual
  54.0 -> 45.5, SSIM 0.348 -> 0.398), 012 (p25 variants: 43.4), 013 (session-specific "_s" bakes), 014
  (serp_plate, board2_*, baffle_nx, hs_nx restored to the all-session median: md5 match with
  `outputs/scratch/upper/tex_med`). Its note "S3-only lost on the rear bay" disagrees with my per-part SSE
  (013 was better on serp_plate 7.9 vs 10.4, board2_a/b, baffle_nx); 014 was the v1 state.
- v1 textures backed up to `outputs/scratch/upper/tex_v1/` (17 files).

## 9. Wave 2 (2026-10-06)

Tools (scratch, `outputs/scratch/upper/`): `eb.py` (absolute per-part SSE after color fit + blur 4, the
error_budget method, A/B across runs), `ebview.py` (per view), `partview.py` (photo | fitted render | error per
part), `cmpbox.py` (photo | render crop of a world box), `plane.py` (robust plane fit to train points),
`settex.sh` (switch texture variants). Probe views unless noted; numbers are SSE/1e8 per part (lower is better).

### 9.1 Textures (upper_015..019)
- 25 more entries textured (clamp_side/clamp2_side/clamp_pad x2, conn_row_top, comp_nx, asic_rear_bracket,
  asic_elbow (98 percent hidden: no file kept), wires x2, wblock x2, lblock x4, rear_bracket, rear_pcb_left,
  pwr_conns, UQD brackets/blocks/holders). Fresh ID-train pass first (`upper_idtrain`).
- Three complete bakes (all / S1 / S3) A/B in full mode: probe total PSNR(blur 4) 14.39 (upper_014) ->
  15.19 (all) / 15.12 (S1) / 14.95 (S3). Upper SSE 17.2e9 -> 12.8e9 (all).
- Per texture: one session unless "all" is better (choices in the TOML [texbake] block). Big movers vs v1:
  mid_pcb 33.7 -> 23.7, clamp_side_nx 11.2 -> 4.2 (S3), conn_row_top 6.3 -> 1.6, clamp2_side_nx 5.4 -> 0.7,
  clamp2_side_px 3.8 -> 0.9, serp_plate 10.4 -> 7.1 (S1), clamp_side_px 5.3 -> 2.5, board2_b 2.9 -> 1.9,
  lblock_nx 2.0 -> 0.9. Worse: serp_tube 3.2 -> 3.8 (untextured; neighbors changed the color fit).
- asic_plate: S3 won the probe total (3.80 vs 4.07 S1) only via S2/S3 views and lost on both S1 views; kept S1
  (holdout is 3/5 S1). baffle_nx S3 (wins 5 of 6 views). Probe views are also bake views: single-session bakes
  score their own session's probe views optimistically.
- erode 0 vs 1 (mid_pcb, baffles; upper_019 vs 018): no change (SSE within 0.1 percent); kept 1.
- Remaining mid_pcb error: a strip along the +X wall where the photo shows the bright wall flange (render: dark
  board), and sun-lit board near the capacitors in S3 views (the all-session median is dark).

### 9.2 Geometry
- (superseded by the refit below) baffle_px slanted: robust plane fit to 71 dark train points in its outline: z = -58.07 + 0.0198 x + 0.0682 y
  (-28 at Y 392, -17 at Y 552; MAD 4.7 mm). baffle_nx plane fit gives -29.7..-34.5 (flat -32 kept).
- Teal braided cable: teal train points at X -103..-96, Y 393-438, Z -38..-33; tube X -100, Y 378-446, r 4;
  material = median of its pixels in 6 probe views (29,33,34); the first try (45,75,95) rendered light blue
  (SSE 2.07, PSNR 11.2).
- UQDs (rear views IMG_5741/5742/5754/5729/5755): per side two bodies along +Y, stacked in Z, not side by
  side in X. Image offset A -> B in IMG_5741 (250, -212 px at 1/2 scale) solved with the per-axis image
  Jacobian: (dX, dY, dZ) = (-0.5, +2, -43) mm. Body A collar at (200, 853, 14): rays of IMG_5741 and IMG_5754
  at X = 185/205 cross at X 200 with a 2.4 mm Z gap (tri/tri2 failed: 1 observation, "all picks disagree").
  Bodies r 9.5, Y 851-873; silver block (188..214, 822..851, 1..27); black holder for body B
  (186..216, 803..851, -45..1); -X mirrored and 6 mm lower (roll). Hoses extended into the blocks.
  Result (upper_022 vs old geometry upper_021, 5 rear views): body edge px px_a 5.44 -> 3.25, px_b 6.56 -> 4.27,
  nx_a 4.55 -> 2.66, nx_b 5.08 -> 2.90; fp stays high (0.26-0.84) because the SAM mask marks the UQD bodies as
  background in IMG_5741 (checked on the mask overlay); in IMG_5742 the bodies are inside the mask.

- serp_plate 17 mm box -> 4 mm plate (Z 10..14): its side rendered as a bright band where IMG_5729 shows the thin
  edge and dark below. upper_023 vs 018: serp_plate SSE 7.33 -> 5.43, pts median 1.39 -> 1.16 mm; rear_pcb
  2.99 -> 4.12 before its rebake (more of it visible), 3.10 after (upper_026).
- baffle_px plane refit: the first fit (section 9.2 top) was contaminated by the capacitor tops (-34) in the
  window: false 3.9 deg slope. Refit on 45 dark points excluding X 115-172 / Y 405-460: z = -32.82 + 0.0176 x
  + 0.0225 y (MAD 1.3 mm), -21.4 at Y 392 -> -17.5 at Y 552; the +X wall strip points (X 170-217, Y 411-490)
  at -18..-20 agree. baffle_px pts median 1.93 (v1) / 2.44 (wrong slant) -> 1.55 mm, edge 3.37 -> 2.99 px.
- +X wall copper tubes ran on top of baffle_px for Y 400-507; photos (IMG_5749/5734) show them only between
  the wall block and the baffle edge: trimmed to Y 478-507 (X 202) and 460-507 (X 211). wtube_px_a SSE
  1.68 -> 0.42.
- Curved second duct edge inside baffle_nx: traced on IMG_5749/5753 orthos at z -32 (agree within 2-3 mm);
  IMG_5742 (very oblique, dxy/dz 2.15) puts the visible edge within about 3 mm of -32; IMG_5729 shows a 7 mm dark
  band (gap/shadow, not a 13 mm wall: that height would shift 5749 vs 5753 by about 8 mm). A 3 mm rib along
  the trace (upper_025): baffle_nx + rib SSE 15.90 -> 15.87, rib edge 5.7 px, baffle_nx edge 3.33 -> 3.82;
  kept in the TOML with skip = true.

### 9.3 Side-face textures (upper_026 vs 027, same time, sides removed for 027)
- `build.py` `tex_sides = true`: the four side faces of a box get `upper_<t>_<xp|xn|yp|yn>.png` (up = +Z,
  right = up x outward normal); `bake_upper.py` bakes them with the parent's options (session, stat, ppm).
  Sides never seen in a session-limited ranking get no file (flat material stays).
- 23 boxes. Every one improved: asic_plate 4.13 -> 3.84, lblock_nx 0.94 -> 0.57, conn_row_top 1.51 -> 1.27,
  board2_conn 0.93 -> 0.76, lblock_px 1.51 -> 1.32, UQD brackets 0.87/0.83 -> 0.60/0.54, UQD blocks
  0.55/0.51 -> 0.02/0.06, wires_px 1.22 -> 1.12. Upper SSE 11.92e9 -> 11.58e9.

### 9.4 Before / after (full mode, probe 8 views; v1 = upper_014, now = upper_026)
Per-part SSE/1e8 after color fit + blur 4 (PSNR dB). Upper total 17.20e9 -> 11.58e9 (-33 percent).
Report: color residual 43.4 -> 38.7, SSIM 0.400 -> 0.416, edges 2.92 -> 2.89 px, sparse median 1.19 -> 1.14 mm
(all groups; other agents edit concurrently).

| part | v1 | now |
|---|---|---|
| mid_pcb | 33.7 (12.8) | 23.0 (14.0) |
| baffle_nx | 17.1 (15.1) | 15.3 (15.6) |
| baffle_px | 16.3 (15.4) | 15.2 (15.9) |
| clamp_side_nx | 11.2 (9.7) | 4.3 (13.9) |
| serp_plate | 10.4 (16.4) | 5.3 (18.6) |
| conn_row_top | 6.3 (9.6) | 1.3 (16.5) |
| clamp2_side_nx | 5.4 (7.9) | 0.7 (16.2) |
| clamp_side_px | 5.3 (13.2) | 2.5 (16.5) |
| asic_plate | 4.3 (17.3) | 3.8 (17.7) |
| rear_pcb | 3.8 (16.0) | 3.1 (18.6) |
| clamp2_side_px | 3.8 (10.2) | 1.0 (14.8) |
| serp_tube | 3.2 (18.7) | 3.7 (18.4) |
| board2_plate_b | 2.9 (16.1) | 1.7 (18.5) |
| hs_nx | 2.5 (16.7) | 2.7 (16.5) |

Remaining error: mid_pcb strip along the +X wall (photo: bright wall flange) and sun-lit board near the
capacitors in S3 views; baffles: view-dependent sheen of the black plastic (5749 photo 24 vs 5729 76 on
baffle_nx) that a diffuse texture cannot follow; hs_px/hs_nx sun-lit in S1.

### 9.5 Files
- Textures: `assets/textures/upper_<t>.png` (42 tops, cap instanced) and `upper_<t>_<side>.png` (side faces);
  bake views and coverage `outputs/textures/upper_*_views.jpg`. v1 textures: `outputs/scratch/upper/tex_v1/`.
- A/B tables: `outputs/scratch/upper/eb_014_017.txt` (sessions), `eb_026_027.txt` (sides), `eb_014_026.txt`.
- Runs kept: upper_014 (v1), upper_015 (all-session bake), upper_022 (UQD, rear views), upper_024, upper_026
  (current); upper_idtrain (ID pass, train views, current geometry).

### 9.6 Requests to coordinator
- (Wave 5) Unmodeled gray rail on the +X side of the rear bay: X 190-199, Z 13-21, Y 641-761 (19 train points,
  7.5 mm from any surface); likely chassis sheet metal.
- (Wave 3) +X wall strip over the main board edge (Y 380-590, X about 200-217, board Z -48): in IMG_5734 the
  photo shows a bright surface there (195) where the model shows the board; in IMG_5729 the photo is dark (12).
  Looks like the chassis wall/rim geometry near the board edge; please check with the chassis agent.
- SAM tray mask of IMG_5741 excludes the rear UQD bodies (black bodies, chrome collars at the +X and -X rear
  corners): correct geometry scores as fp there.

## 10. Wave 3 (2026-10-06)

### 10.0 Holdout disclosure (rule update DevLog-005)
- I opened `outputs/snapshots/v1/error_budget_holdout.json` (Wave 2 start) and
  `outputs/snapshots/v2/error_budget_holdout.json` (Wave 3 start; printed its upper-part shares). No holdout
  photos, crops or holdout runs were opened. Wave 2 kept asic_plate on S1 citing the holdout session mix; that
  choice is re-decided below on probe evidence only (now S3 p25). From here on: probe/train budgets only.

### 10.1 Session/statistic variants (upper_028 baseline, upper_029-034)
- Fresh ID pass (`upper_idtrain`), then six complete bakes of 15 large textures (all/S1/S3 x median/p25, with
  side faces) and six full probe runs back to back. New tool option `eb.py --fitx`: the per-view color fit uses
  only non-upper pixels, so upper changes do not move the fit.
- Caveat found: the blur-4 error spills across part borders, so in a run where many textures change at once a
  part's SSE also moves with its neighbors (mid_pcb read 27.1 -> 29.7 in the all-p25 run although its own
  texture rendered identically, see 10.2). Choices below are from `eb_var3x.txt`; small margins are noise-level.
- Chosen (SSE/1e8, fitx, probe): baffle_px S1 p25 (19.30 -> 18.98), clamp_side_nx all p25 (4.28 -> 3.95),
  asic_plate S3 p25 (3.92 -> 3.60), hs_px S3 median (3.20 -> 3.04), clamp_side_px all p25 (2.88 -> 2.78),
  clamp_bar S1 p25 (2.85 -> 2.49), hs_nx S1 p25 (2.82 -> 2.64). Kept: mid_pcb, baffle_nx (S3 median: 16.71;
  S3 p25 16.85, all p25 19.73), serp_plate, rear_pcb, ribbon, comp_nx, fin_block, caps.
- Side files that the chosen session did not produce were moved out (asic_plate_xp, hs_px_xp/yn, hs_nx_yn:
  `outputs/scratch/upper/tex_w2/stale_*`) so the files match a re-bake from the TOML.

### 10.2 mid_pcb strip along the +X wall
- Large-error mid_pcb clusters (upper_037, per-pixel |fit - photo| > 60): IMG_5734 (S1) an 83 x 606 px strip
  along the +X wall: photo 195 vs render 36; IMG_5729 (S1) a 102 x 265 px strip: photo 12 vs render 86;
  IMG_5749 (S3) 124 x 302 px right of the capacitors: photo 180 (sunlit board) vs render 20.
- The same strip is bright in one view and dark in another: an occlusion/geometry mismatch at the wall (a bright
  surface covers the board edge in IMG_5734), not a texture statistic. A bake with |X| > 196 mm texels from a
  dark percentile (p10, `dark_x` option in bake_upper.py) changed 17.7k texels but no probe pixel (upper_036 vs
  035: identical SSE): the strip pixels are not those texels. Re-bake with the current ID pass = Wave 2 file in
  render (upper_035 vs 037 identical). Left as is; request to coordinator below.
- The sunlit area right of the capacitors (S3 views) cannot be matched by one static texture; S3 bakes lose on
  S1/S2 views (S3 median mid_pcb 38.1 vs 27.1).

### 10.3 Baffles: what drives their error (upper_037, color fit on non-upper pixels)
- baffle_px: 60 percent of its SSE is on photo-bright pixels (blurred luminance > 90): the S3 views are sunlit
  (IMG_5749 photo median 103, IMG_5751 157 vs render 59/76); in S1/S2 views the render is brighter than the
  photo (fit 56-86 vs photo 40-56). baffle_nx: 46 percent bright share (IMG_5729 sun patches); elsewhere the
  render is brighter than the photo (fit 67-96 vs photo 24-56).
- A darker texture cannot follow: the per-view affine fit has a positive offset, so near-black renders map to
  that floor. One-session or p25 bakes: baffle_px S1 p25 kept (small gain, 10.1); S3 bakes are far worse on the
  other sessions (baffle_px S3 median 46.3 vs 19.3). Cast sun patches are not reproducible with this rig.

### 10.4 Cylinder side textures (UQD bodies, capacitors; upper_038 vs 039, probe + IMG_5741/5754)
- `build.py` `_cyl_side_tex`: side faces of [[cyl]] with tex, and of [[cyls]] with tex_sides (one instanced
  texture), get `upper_<t>_side.png` (u = angle, v = along the axis). `bake_upper.py` bakes cylinder sides with
  all train views nearest first and a per-texel facing test (the back side projects onto the same object).
- UQD bodies: uqd_px_a SSE 0.43 -> 0.27, colres 36.6 -> 25.0, SSIM 0.35 -> 0.69; uqd_nx_a 0.115 -> 0.069
  (colres 48.4 -> 35.1); px_b 0.171 -> 0.134; nx_b 0.070 -> 0.046.
- Capacitors: caps_px 3.13 -> 2.94; caps2_px 0.233 -> 0.289 (worse): side texture on caps_px only.

### 10.5 Before / after Wave 3 (probe, full mode, eb.py --fitx, current masks; upper_028 -> upper_040)
Upper SSE 13.03e9 -> 12.77e9 (-2 percent). Per part (SSE/1e8): mid_pcb 26.19 -> 25.96, baffle_px 18.16 ->
17.81, baffle_nx 15.69 -> 15.67, serp_plate 6.02 -> 5.90, clamp_side_nx 4.31 -> 4.01, asic_plate 4.15 ->
3.90, hs_px 3.18 -> 3.07, hs_nx 2.86 -> 2.73, caps_px 2.79 -> 2.52, clamp_bar 2.77 -> 2.39; worse:
clamp_side_px 2.92 -> 3.01, fin_block 1.85 -> 1.88. Table: `outputs/scratch/upper/eb_028_040.txt`.
Not done in Wave 3: component geometry (connectors, fins); worst edges now ribbon 5.05 px (IMG_5742: a light
untextured slab of mine next to it), uqd_bracket_px 4.3, hoses 4.2-4.3, rear_bracket 4.1.

## 11. Wave 4 (2026-10-06): geometry details, validated on two view lists

Lists: probe (8) and list2 = train views not in probe: IMG_5713, 5724, 5731, 5733, 5741, 5744, 5753, 5755
(`outputs/scratch/upper/list2.txt`; all are train views, so also possible bake views). Runs back to back per
change: upper_041/042 baseline, 043/044 hose_px, 045/046 ribbon + bracket + fittings, 047/048 fittings off +
hose_nx, 049/050 cables_nx route, 051/052 final. Probe/train evidence only. Per-part SSE: `eb.py --fitx`.
Group totals moved by up to 2.3e9 between back-to-back pairs from other agents' concurrent edits (mid_pcb pixel
count 293k -> 257k on probe between 041 and 051 without an upper change there): only per-part numbers of
the changed parts are used.

| change | evidence | probe | list2 | kept |
|---|---|---|---|---|
| hose_px at Z about 8 (was 20) | overlays of Z 20/6/-6 lines on train 5713/5753: hose between the Z 6 and 20 lines, Z 20 ran on the gray rail | edge 3.00 -> 2.48, SSE 1.42 -> 0.40 | edge 4.47 -> 3.16, SSE 2.83 -> 0.50 | yes (pts 2.99 -> 4.39 mm: the rail points at Z 16-26 now lie off it) |
| ribbon 1.5 mm thick (was 4) | light ribbon side strip under the serpentine tube (5744, 5742) | SSE 2.94 -> 2.15, edge 4.95 -> 5.04 | SSE 3.02 -> 2.90, edge 4.24 -> 4.32 | yes (pts 1.44 -> 1.17 mm) |
| uqd_bracket_px: 3 mm plate (was 22 mm block) | 5713/5741/5755 show an open fitting where the block rendered solid | SSE 0.64 -> 0.30, edge 3.26 -> 3.01 | SSE 4.07 -> 0.64, edge 4.58 -> 4.64 | yes |
| uqd_px_fit chrome barb (r 6, Y 790-822, body A axis) | 5741 chrome tube + nut below the silver block | SSE 0.03 (881 px) | SSE 0.43 (4791 px, PSNR 13.4); without it bracket edge 4.64 -> 4.89 | yes (edges) |
| uqd_nx_fit (mirror) | none direct | PSNR 12.4, edge 4.7 | PSNR 8.0, edge 6.2 | no (skip = true) |
| hose_nx Z 10 -> 7 | 5713: hose between the Z 0 and 10 lines; 5741: on Z 10 | SSE 0.43 -> 0.30, pts 2.75 -> 1.78 mm, edge 4.96 -> 4.94 | SSE 0.59 -> 0.35, edge 4.01 -> 4.05 | yes |
| cables_nx lower/thinner | no dark points on the old route Y 675-790 | edge 3.38 -> 4.80, pts 4.2 -> 11.8 | edge 3.26 -> 5.72 | no (reverted) |

Looked at, not changed: conn_row_top and comp_nx errors are photometric (black parts lifted by the color-fit
offset) or the chassis crossbar covering comp_nx in 5713/5753 (silver over 40 percent of its outline: chassis
geometry); rear_bracket error is sunlit S3 sheet metal; fin_block fins are below the 1/4-scale resolution; the
light slab next to the ribbon in IMG_5742 and the small light box at the -X rear corner are chassis objects
(tex_wall_mx_in, clip_mx1). The uqd_bracket_px texture files (old block) were moved to `tex_w2/`.

## 12. Wave 5 (2026-10-06)

Baselines after the chassis ledge (chassis.crossbar_ext): fresh ID pass, upper_053 (probe) / upper_054 (list2);
final upper_059 / upper_060. Per-part SSE/1e8 with `eb.py --fitx`; probe/train evidence only.

| change | probe | list2 | kept |
|---|---|---|---|
| re-bake middle-bay textures against the ledge (mid_pcb, baffles, comp_nx, conn_row_top, clamps, wires) | mid_pcb 12.42 -> 11.91, conn_row_top 0.88 -> 0.83, baffle_px 15.10 -> 15.19 | mid_pcb 19.28 -> 18.99, conn_row_top 3.13 -> 2.91, baffle_px 25.78 -> 25.71 | yes |
| capacitor re-bake (cap, cap_side) | 2.150 -> 2.161 | 1.979 -> 2.005 | no (Wave 4 files restored) |
| serp_tube S-jog before the -X L-block: Y 582 -> 577 between X -158 and -168 (orthophotos of train 5749 and 5753 at z 12 agree) | SSE 5.08 -> 4.91, pts 1.21 -> 1.16 mm | 5.38 -> 5.37 | yes |
| hose_nx to the dark-mask line fit (X -200, Z 6; same in two 6-view train sets) | edge 4.94 -> 3.92, SSE (with rear_pcb_left) 1.01 -> 1.12 | edge 4.05 -> 4.41, SSE 0.65 -> 0.76 | no |
| cables_nx dark-mask fit | best line X -162..-165, Z 19 (on the edge of the search range, over the ribbon): not a cable trace | - | not tried |

hose_px rail points (Wave 4 pts 2.99 -> 4.39 mm): 66 train points at X 195-222, Y 610-800, Z 14-30 are mostly on
chassis.wall_px_flange / rim (bright). A separate cluster is unmodeled: 19 gray (median RGB 93/97/98) train
points at X 190-199 (median 195), Z 13-21 (median 17), Y 641-761, median 7.5 mm from any model surface; it is
the gray rail the old Z 20 hose sat on in IMG_5713/5753. Sheet metal along the +X wall inboard of the hose:
likely chassis structure (reported to the coordinator, not modeled here).

## 13. Wave 6 (2026-10-06): visible detail toward v4

Rule (coordinator): accept if neutral on probe AND list2, reject if it regresses both. Other agents edited
chassis/lower during the wave (a whole-scene color-fit shift made upper_061/062 vs 059/060 unusable: serp_tube
4.9 -> 26.8 with identical raw render colors), so every A/B is now a back-to-back pair rendered by
`outputs/scratch/upper/pair.sh` (baseline TOML + textures swapped in temporarily, then the current state; runs
<prefix>a_p/a_l/b_p/b_l) and compared with `ab.sh` (per-part edge/pts and SSE/1e8 with `eb.py --fitx`).
New build features: [[cyls]] `top_round` (rounded top edge) and `verts`; [[tube]] `fillet_n`, `bevel_res`;
[[boxes]] (repeated boxes as one object).

| detail | evidence | probe (SSE/1e8) | list2 | kept |
|---|---|---|---|---|
| serpentine bends 12 segments, 4 bevel steps | - | 4.91 -> 4.95, edge 2.01 -> 1.99 | 5.43 -> 5.42 (pair 064), edge 2.06 -> 2.09 | yes (neutral) |
| capacitor tops rounded 0.6 mm, 24 sides | - | caps_px 2.14 -> 2.20, caps2 0.163 -> 0.152; edges better | 2.03 -> 2.05, caps2 0.035 -> 0.033 | yes (neutral) |
| power connectors: 3 housings + latch tabs (was one block) | ortho of train 5749 at z -2: housings X -180..-168, Y 657-675 / 702-722 / 750-768 | pwr + rear_pcb_left 0.95 -> 0.91 | 0.35 -> 0.39 | yes (mixed) |
| QR blocks single best view (n_best 1, 24 px/mm) | - | cu_block 0.47 -> 0.71 | 0.53 -> 0.65 | no (both worse) |
| UQD collars (r 10, 6 mm), red O-ring steps, barb ridges; first try chrome (too bright) then darker metal | IMG_5741 crops | UQD region +0.03 (neutral) | +0.03 without the nut (collars +0.18, barb 0.43 -> 0.27) | yes |
| UQD hex nut | - | +0.002 | +0.146 (PSNR 14.1) | no (skip) |
| clamp bar screw row: 8 dark discs, pitch 7.95 | ortho 5749 at z -31 (8 px/mm), 5753 agrees | clamp_bar + screws 2.54 -> 2.55 | 2.27 -> 2.26 | yes |
| +X heat spreader screw row (8, pitch 8.1) + connector latch clips | ortho 5749 at z -29 | region 4.61 -> 4.67; clips edge 1.30 px, pts 0.47 mm | 5.68 -> 5.72; screws pts 0.35 mm | yes (neutral) |
| re-bake under the new details (rear_pcb_left, hs_px, clamp_bar, wires_px, ribbon; fresh ID pass) | - | rear_pcb_left 0.81 -> 0.78, rest within 0.01 | 0.32 -> 0.29 | yes |

Not done: heatsink fin arrays (the "fin_block" region is a stainless plate with a screw row, not fins; the
comb-like teeth are the connector latch clips, now modeled on +X); -X heat spreader screws/clips (not measured);
serpentine plate screws (in the texture).

## 14. Wave 7 (2026-10-06): last measurable details

Back-to-back pairs (`pair.sh`, `ab.sh`), SSE/1e8 probe | list2; final runs upper_069b_p / upper_069b_l.

| detail | evidence | probe | list2 | kept |
|---|---|---|---|---|
| -X heat-spreader screw row (8, X -82.8, Y 486.7-542.5, pitch 8.0) + larger hole (-66.7, 514.7) + 8 latch clips (X -96..-87); measured, not mirrored | orthophotos of train 5749 and 5753 at z -33 agree | region 3.09 -> 3.12; hs_nx pts 0.93 -> 0.83, wires_nx 0.91 -> 0.78 mm | 7.17 -> 7.22 | yes (neutral) |
| coin cell moved (85, 662) -> (89, 672) + holder tabs | orthos of 5749 and 5753 at z 0 agree | coin edge 2.82 -> 1.51 px, pts 1.47 -> 0.53 mm, SSE 0.29 -> 0.20 | edge 3.17 -> 1.61, SSE 0.38 -> 0.18 | yes |
| re-bake rear_pcb, hs_nx, wires_nx (fresh ID) | - | rear_pcb 2.74 -> 2.71 | 2.88 -> 2.84 | yes |

Stopped here: the remaining board details (small QFN chips with labels, the holder clip springs, board2
connector latch) are 2-5 mm features near the 1/2-scale resolution limit and are already in the textures.

## 15. Next steps
- mid_pcb: the +X wall strip (chassis flange visible over the board edge?) needs a look with the chassis agent;
  per-session or darker-percentile bakes for sun-lit board areas.
- Baffles: try p25/p75 bakes per session; consider a glossy black material A/B (CRT: flat).
- rear_pcb_left height; UQD bodies as textured cylinders (chrome collar, black body); hose ends.
- Re-bake after other groups change occluders (ID pass first: `upper_idtrain`).
- (Wave 3 update) Variant A/B with many simultaneous texture changes is confounded by blur-4 spill across part
  borders: test large neighbors (mid_pcb, baffles) one at a time. Component geometry (connector rows, fin
  block, cable ends) and the light slab beside the ribbon in IMG_5742 are the next edge items.
- (Wave 4) Hose and cable bundle routes on the -X side are weakly constrained (few dark points); a
  dark-mask line fit (like the copper line fit) over 5-6 train views would pin them. Serpentine left leg near the
  -X L-block has an S-jog in IMG_5753 that the model lacks.

