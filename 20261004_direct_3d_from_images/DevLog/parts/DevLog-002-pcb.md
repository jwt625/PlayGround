# DevLog-002-pcb: main board, flyback, heat-sink plate, components

| Field | Value |
|---|---|
| Group | pcb |
| Owner | pcb modeling agent |
| Files | `scripts/model/pcb/build.py`, `config/model/pcb.toml` |
| Started | 2026-10-04 |
| Phase | 1B (textures, materials, details, fits) |

## TODO
- [x] Read AGENT_GUIDE.md, DevLog-001 sections 2 and 4
- [x] Board plane height and outline
- [x] Flyback box (with plain label quad), heat-sink plate
- [x] Large components (caps, chokes, trimmers, connectors, regulators) as TOML lists
- [x] Iterate with `scripts/iterate.sh pcb_NNN probe pcb. geom` (4 runs: pcb_001-003 + one 4-view check run)
- [x] Phase 1B: textures (flyback top, flyback +Y sticker, board top), materials, details, fits (pcb_004-009)
- [x] Phase 1C: flyback sides, small parts, board re-bake excluding wire-owned texels, bake tool promoted (pcb_010-013)
- [ ] Next: re-bake board when internal wires (red loom, white HV cable, grey cable) are modeled; TO-220 detail

## Method
- Sparse COLMAP points (world frame) in the tray region cached to `outputs/scratch/pcb/pts.npy`; top-down and side
  maps colored by RGB and z; per-component clusters give top z (90th percentile) and top-centroid xy.
- IMG_1560 (camera at (38, 2, 180) mm, looking almost straight down) resampled as orthophotos at z = 6, 17, 27
  (pixel rays intersected with the plane); features at that height are in true xy.
- pick.py tri for board plane and plate top edge.

## Measurements (mm, world)
| Item | Value | Provenance |
|---|---|---|
| Board top | z 6.0 | tri IMG_1560 (1170,790),(1060,785),(1250,800): z 5.86/6.37/5.62; brown sparse pts z 3.5-8 median 5.9 |
| Board outline | x 1.5..96.3, y -48..47 | brown sparse-point percentiles (x 1-98 pct 2.6..98.7; y -46..46.8), ortho z6 |
| Flyback top | x 56.9..87.2, y -49.2..-28.3, z 26.9 | sparse pts z 24-29 (x 1/99 pct 57.1/87.4, y -50.7/-28.2, z median 26.8); coordinator label tri (76.4,-44.4,26.7),(65.3,-32.3,27.1) |
| Flyback label quad | corners from coordinator (warn_spec.json) | pick.py ray IMG_1560 at z 26.9 |
| Heat-sink plate | vertical, y 46.8..48.3, x 12..95, top 28.4 | tri IMG_1626 top edge: (31.0,49.0,27.9),(44.3,48.2,28.4),(15.9,48.6,28.7); IMG_1613 shows the -Y face |
| Black block "8XSZ" | x 64..91.6, y 38.2..46.8, top 21 | sparse pts z median 21.0 |
| Small black caps | d 5, top 17 | sparse cluster z90 17.0-17.5 |
| Trimmers | 7 x 7, top 13.5 | sparse cluster z90 13.1-14.3 |
| capK_l (large black) | (91.2,-23.3) d 10 top 21.2 | sparse z90 21.2 |
| Regulator heat tab | inclined slab, center (28.5,36.3,20.0), 14 x 8 x 1.5, tilted 50 deg about X | tri IMG_1626 (1420,915) -> (23.5,36.5,21.8) 5 inliers; (33.1,34.7,19.2); sparse y-z profile z 17 -> 24 over y 33.5 -> 38 |
| White part, ceramic discs | white (42..48, -34.5..-27.5) top 15; blue discs top 12-13 | orphan cluster (45.3,-32.5,13.8) n 43 in pcb_001; sparse z 90th pct |
| Chokes | a (74.5,30.5) top 25.5; b (85.7,20) top 21.7; c (78.2,-23) top 19.4 | sparse z90 |

## Progress
- 2026-10-04: measurements above; first block-out written (board, flyback + base + label quad, plate, 10 blocks,
  7 trimmers, 15 cylinders).

- 2026-10-04: pcb_001: all parts in place within about 1-2 mm on IMG_1560 overlay. Found an ID-pass decode bug
  (some Workbench ID colors 1 LSB off the id_map, parts read as "no model", fn blamed on neighbors); reported to
  coordinator, fixed in evaluate.decode_id (nearest color, L1 <= 6).
- 2026-10-05: pcb_002: added white_part, cer_a/b/c (orphan cluster gone); regulator tab changed from flat box at
  z 12-13.5 to an inclined slab (edge 4.95 -> 4.06 px, pts 0.73 -> 0.67 mm). New `[[slab]]` list (center, size,
  rot_x_deg, rot_z_deg).
- 2026-10-05: pcb_003 (fixed evaluator, last run): flyback pts median 0.21 mm / edge 2.36 px; label 0.12 / 1.88;
  plate 0.29 / 1.99; black_block 0.29 / 4.47; board 0.70 / 3.78; caps and chokes 0.4-1.2 mm / 2.0-3.4 px;
  worst small parts: cer_c 2.50 mm, conn_c 2.42, film_b 2.14, film_a 1.88, white_part 1.58, capB_i 1.48.
  Remaining fn on the plate (8401 px, IMG_1624) is the case outer wall region beyond the plate (case group).
  Visual check views IMG_1613, IMG_1615, IMG_1626, IMG_1550: positions consistent; the case tray wall in the
  model is taller than in the photos and hides the flyback face and the plate top (case group, rim measured 24.9).
- Phase 1A block-out done.

## Phase 1B progress (2026-10-05)
- Textures (specs in outputs/scratch/pcb/tex_*.json):
  - assets/textures/pcb_flyback_top.png: coordinator corners, n_best 5 median (IMG_1613, 1622, 1632, 1562, 1561);
    clean, no glare.
  - assets/textures/pcb_flyback_side_py.png: "SY-F100C NO. 7-9" sticker on the +Y face, z 17..26.9 (lower part
    hidden by chokes/caps), n_best 3 (IMG_1627, 1625, 1628).
  - assets/textures/pcb_board_top.png: own script outputs/scratch/pcb/board_tex.py (uses texture_from_photos
    helpers): z 6 plane, 8 px/mm; texel visible in a view only where pcb.board owns the ID pixel (model occlusion,
    eroded 2 px); per texel median of the first 3 visible views in rank order (IMG_1560, 1561, 1559, 1588, 1589,
    1622, 1629, 1558, 1562, 1631); never-seen texels (57 percent, under parts) filled with the median board color
    instead of inpaint (the tool's Telea inpaint smeared large holes). Unmodeled wires/small parts are painted on.
  - Applied as quads: pcb.label_board_top, pcb.label_flyback_warning, pcb.label_flyback_side.
- Materials: base colors from lit photo samples (outputs/scratch/pcb/sample.py), Specular IOR Level 0.25 for
  non-metals (coordinator rig); metal/cap_top/leg metallic.
- Details: silver vent discs on electrolytics, white trimmer rotors (base top 12.3, rotor 13.6), TO-220 leads
  (pcb.legs_0), plate top flange y 46..48.3 (sparse pts at z 26-30 concentrate at y 46-48.5) and chamfered -X
  corner, black flyback end faces (IMG_1613 shows the -X face dark; flyback color residual 92 -> 66).
- fit_params (dx, dy offsets via a [fit] table, 14 views, 28-40 evals each, outputs/fits/pcb_*): accepted
  film_b (+1.4, +1.25), capB_i (-0.56, -1.0), white_part, film_a, trim_6 (< 0.5 mm). Rejected cer_c dx -3.75:
  blue sparse points at (65.6-67.5, -19.4, z 12.7-15) put it at x 62-68.5, top 15 (pts 2.50 -> 0.56 mm).
  Film caps raised to top 16.5-17 (green points z 15.7-17.7; film_a pts 2.39 -> 1.15). Added dark_small
  (sparse cluster at (67-71, -15..-13)).
- Incident: a TOML edit cut at the first "[mats]" string (inside a comment) and dropped the block/cap lists
  for a few minutes; the build failed for other agents' runs in that window. Restored; build.py edits now go
  through a tested copy.
- Last run pcb_009 (probe + IMG_1613, IMG_1626, full mode): median over pcb parts pts 0.54 mm, edge 2.61 px,
  color residual 50.8. Sheet: outputs/runs/pcb_009/pcb_textures_sheet.jpg.

## Phase 1C progress (2026-10-05)
- Flyback -Y face: texture tried (outputs: discarded) - the face is behind the case near wall except z > 24;
  IMG_1613 shows a dark grey core/bobbin band (sRGB 72,71,67) below the top edge, and the -X face is dark too.
  Flyback body is now dark grey (fly_dark); beige only via the textured top and +Y sticker quads. Black base
  height not measurable (hidden by the wall); kept base_h 3.0 (same dark look).
- Small parts (IMG_1560 ortho z 8 at 14 px/mm, sparse-point boxes, tri): ic_a DIP-8 (tri IMG_1626
  (39.6,21.1,8.9)), q_a small black standing part (62,-22.3), capK_n (orphan cluster (75.6,20.9,19.7) removed),
  hv_boot (HV cable boot at the flyback -X face, horizontal, z 22.6), diode_a and res_green as [[axial]] items
  (body + leads). New TOML list [[axial]] (center, axis, r, len, lead_len, mat); caps accept vent = false.
- Board top re-bake: scripts/tools/bake_id_owned.py (promoted from scratch; occlusion_object may be a list -
  the board box and its texture quad both own the top), ID run outputs/runs/pcb_bake_id with 16 wires.* objects
  present. The internal red loom, white HV cable and grey cable are not modeled yet, so they still paint onto
  the board. Re-run when wires change:
  `V=$(python3 -c "import json;print(','.join(json.load(open('outputs/scratch/pcb/tex_board.json'))['views']))"); scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- --out outputs/runs/pcb_bake_id --build scripts/model/build_all.py --views $V --passes id && uv run python scripts/tools/bake_id_owned.py outputs/scratch/pcb/tex_board.json`
- AGENT_GUIDE.md: one line under Textures for bake_id_owned.py (coordinator request).
- Last run pcb_013 (probe + IMG_1613, IMG_1626, full): 50 pcb parts, medians pts 0.58 mm, edge 2.72 px, color
  residual 49.7; flyback 0.21 mm / 2.21 px; labels warning 0.13 mm SSIM 0.54, side 0.08 mm SSIM 0.61; model PSNR
  12.46 (pcb_009: 11.76). Sheet: outputs/runs/pcb_013/pcb_textures_sheet.jpg.

## Phase 1D (2026-10-05): re-bake with wires, holdout check, fits
- Holdout check (data/split.json): board, flyback top and +Y sticker textures used no holdout view. The discarded
  -Y face test used IMG_1522 (holdout); its files were deleted and nothing uses it.
- Board re-bake with bake_id_owned.py; ID run had 28 wires.* objects incl. hv_cable, grey_cable, grey_inner,
  red_a..d, loop_*. hole_frac 0.60 (was 0.595); the white HV cable mostly drops out; some red wire remains where
  the modeled wires and photo wires differ.
- Fits (14 probe+side views, train only): hv_boot dx +1.7 accepted (loss 3.27 -> 2.85); its dy -2.25 rejected
  (sparse y median -44.7; wires.hv_cable starts at the boot). ic_a dx +3.4 rejected (loss 7.86 -> 7.54 only,
  contradicts sparse x 37-43.6 and tri (39.6,21.1,8.9)). Axial items now take [fit] dx/dy/dz offsets.
- TO-220 detail skipped (geometry of reg_a/reg_b not measured well enough to be cheap).
- pcb_014: PSNR inside pcb ID pixels 10.33 -> 10.38 dB (pcb_013 -> pcb_014; script outputs/scratch/pcb/psnr_pcb.py,
  raw photo vs render, no color fit); label_board_top color residual 32.5 -> 30.1.

## Phase 2 (2026-10-05): photo textures on procedural parts, missing-geometry check
- bake_id_owned.py: added --rank / "views": "auto" (TRAIN views from data/split.json only, ranked by cos x px/mm
  at the patch center) and a hard stop if a spec lists a holdout view. Specs: outputs/scratch/pcb/tex/*.json;
  ID runs outputs/runs/pcb_bake_id2 (plate, black block) and pcb_bake_id3 (connectors, trimmers).
- New TOML list [[tex_quad]] (name, texture, corners); trimmers 0-4 switch to a box up to rotor_top with a
  textured top quad when assets/textures/pcb_trim_<i>_top.png exists.
- Kept: plate_top, plate_face, black_block_top ("8XSZ" sharp), conn_a_top, trim_0..4 tops. Discarded (occluded
  or wire-dominated): black_block front, flyback -X face, conn_b top, trim_5, trim_6.
- Missing geometry: probe run without --parts (pcb_015) has no "unmodeled" region tiles; cyan fn pixels lie on
  the case silhouette, mat lines and external wires, none over the board (the board covers the tray, so unmodeled
  board parts show up as board-texture error instead, already painted by the bake). Nothing new to model there.
- error_budget (probe, color-fit dB per part) pcb_015 -> pcb_017: heatsink_plate 14.1 -> 16.7 (plus label_plate_face
  17.7, label_plate_top 15.3); conn_a 12.1 -> 13.5 (top 17.7); black_block top 19.9; trim tops 15.0-18.6 vs
  procedural 15.6-17.8 (mixed). pcb group share 13.1 -> 12.6 percent (pcb_016); pcb_017 shares shifted because the
  case group changed concurrently (case 62.9 -> 75.1 percent). Raw PSNR inside pcb ID pixels 12.65 -> 13.09 dB.
- Worst remaining pcb parts (dB): trim_6 10.1, capB_i 10.2, capK_n 10.4, conn_b 11.3, reg_tab 13.2, choke_a 13.7.

## Phase 3 (2026-10-05): photo textures on every component
- scripts/model/pcb/uvparts.py: analytic patches (box faces, cylinder side + polar top) + shelf-packed atlas per
  part at 6 texels/mm; parts_from_params(P) covers blocks, slab, trimmers, caps/chokes, axial parts, flyback,
  flyback_base, heat-sink plate (one box, chamfer dropped), board. build.py makes UV meshes from the same patches
  and replaces the procedural object (and its old label quads) when assets/textures/pcb_parts/<part>.png exists.
- scripts/model/pcb/bake_parts.py: per texel, best K = 5 of all TRAIN views (data/split.json) by cos x px/mm,
  texel must face the view (cos > 0.15) and the ID pixel (1 px interior) must belong to the part or its labels;
  median of the K samples from the 1/2-scale photos; unseen texels = material color. Sheet:
  outputs/textures/pcb_parts_sheet.jpg. Needs a train ID run of the current model:
  `scripts/bslot.sh -b --factory-startup --python scripts/blender/render_views.py -- --out outputs/runs/pcb_idtrain --build scripts/model/build_all.py --views train --passes id && uv run python scripts/model/pcb/bake_parts.py --id-run outputs/runs/pcb_idtrain`
- Tested: min-cos 0.35 (no gain, reverted), K 3 -> 5 (pcb PSNR 13.64 -> 13.75 dB, kept).
- Geometry fits (probe, step 0.5): conn_b, capB_i, trim_6, film_a, choke_c all moved < 1.2 mm with loss gains
  < 1.5 percent: positions already at the edge-loss optimum; not applied.
- error_budget probe pcb_019 -> pcb_025: pcb share 15.6 -> 11.2 percent; raw PSNR in pcb pixels 13.12 -> 13.75 dB.
  Per part dB: conn_b 11.0 -> 17.5, trim_6 9.9 -> 19.5, capK_n 10.3 -> 17.1, capB_i 9.9 -> 19.5, reg_tab 13.1 ->
  17.2, choke_a 14.2 -> 16.1, choke_b 14.8 -> 19.6, film_a 15.7 -> 19.3. Plate (as one textured box) 2.5 percent
  at 16.6 dB vs 2.4 percent before as quads (neutral). Board (generic bake) 3.0 percent at 17.5 dB (neutral).
- Holdout check (pcb_hold, 14 holdout views, eval only): pcb share 9.3 percent, PSNR in pcb pixels 13.49 dB
  (probe 13.75): the textures generalize.
- Sheet: outputs/runs/pcb_025/pcb_components_sheet.jpg. Board re-bake pending the wires agent's final paths.

## Phase 1B plan (as of 1A)
- Texture the flyback WARNING label (coordinator spec outputs/scratch/coord/warn_spec.json, best views IMG_1613,
  IMG_1622, IMG_1632) and the flyback -X face ("SY-F100C NO."); board top texture (orthophoto at z 6 from
  IMG_1560 is clean outside the yoke footprint).
- Materials: brown board, beige paper flyback, aluminium plate, blue/black sleeves with silver vent tops.
- Details: trimmer white rotors, cap vent tops, TO-220 bodies/legs, connector pins, small ceramic/film caps and
  diodes, more small caps under/near the yoke; flyback black base and HV cable exit; the plate's angled -X end.
- Fit (fit_params.py) small parts with pts > 1.4 mm (cer_c, conn_c, film_a/b, white_part, capB_i, trim_6).

## Scratch tools (outputs/scratch/pcb/, small Python scripts kept for 1B)
- pts.py (cache tray sparse points), topmap.py / side.py (point maps), ortho.py VIEW Z [x0 x1 y0 y1 S]
  (plane-resampled orthophoto), comp.py / box.py (cluster heights and extents), cmp.py RUN VIEW (photo with model
  edges | composite), idcheck.py RUN (ID decode check).

## Requests to coordinator
- (done) ID decode tolerance.
- 2026-10-05: board top re-baked with wires.red_e and re-traced red_d (train views, hole_frac 0.604); pcb_018 probe: label_board_top color residual 33.3 -> 30.3, SSIM 0.305 -> 0.324; raw PSNR in pcb pixels 13.09 -> 13.12 dB.
- 2026-10-05: final re-bake after the wires agent's Phase 3 (29 wires objects in a fresh 123-view train ID run): all 47 part atlases incl. board. pcb_026 probe: raw PSNR in pcb pixels 13.75 -> 13.76 dB; per-part dB unchanged within 0.2 (board 17.6 -> 17.5); pcb share 11.2 -> 11.5 percent while missing-geometry share fell 6.4 -> 5.4 (shares redistribute). Sheet outputs/runs/pcb_026/pcb_components_sheet.jpg.
