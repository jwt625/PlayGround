# DevLog-002-chassis: tray shell (Phase 1A block-out, Phase 1B details and photo textures)

| Field | Value |
|---|---|
| Date | 2026-10-05 to 2026-10-06 |
| Status | Phase 1A accepted; Phase 1B done; Wave 2 done; Wave 3 done (sign-free wing, rear-bay deck height); Wave 4 done (bake A/B, no change kept); Wave 5 done (crossbar edges, duct width: rejected); open items in TODO |
| Scope | Floor pan, side walls with top folds, rear wall, front panel (port plate, top lip, grip bar, 8 MMC groups, 4 RJ45, USB, LED row), corner levers, NVIDIA crossbar, divider, black front-bay side ducts |
| Files | `scripts/model/chassis/build.py`, `chassis_surfaces.py` (textured quads), `bake.py` (texture baker), `config/model/chassis.toml`, helpers `scripts/model/chassis/tools/`, textures `assets/textures/chassis_*.png` (sheets `outputs/textures/chassis_*_views.jpg`) |
| Runs | `outputs/runs/chassis_001` (first block-out), `chassis_007` (end of 1A, geom), `chassis_008` (start of 1B, full), `chassis_017` (end of 1B, full, probe), `chassis_own` (end of 1B, full, own view list), Wave 2: `chassis_018` (tray frame), `chassis_024` (texture winners), `chassis_029` (divider flange), `chassis_034` (end of Wave 2, probe), `chassis_own2` (end of Wave 2, own list), Wave 3: `chassis_045` (rear deck), `chassis_fz48` (deck sweep, 7 rear-bay views), `chassis_048` / `chassis_own3` (end of Wave 3), `chassis_idtrain` (ID pass of 38 train views of the final model, for baking) |
| Authors | Claude (chassis agent) |

## 1. TODO

- [x] Read AGENT_GUIDE, playbook 5-8, DevLog-001
- [x] Measuring helpers (`tools/`): `ortho.py` (resample a view onto an axis plane with an mm grid),
      `prof.py` / `profa.py` (edge profiles across / along an axis plane, chunked), `sample.py` (photo color
      medians at world points), `edges.py` (render edges over a photo), `cmp.py` (per-part table across runs).
      Run from the project root with `uv run python scripts/model/chassis/tools/<x>.py ...`; outputs go to
      `outputs/scratch/chassis/`.
- [x] Block-out, one object per part, 7 probe iterations + 4 fits
- [x] Rim height, floor underside, rear end, crossbar, divider, ducts, MMC group pitch
- [x] Photo-sampled flat colors
- [x] Phase 1B geometry: lever hook heads (disc, axis X; fit), wing as an extruded Y-Z profile (replaces two
      slabs; reaches the levers), lip moved behind the plate, rim clips (4), port boxes disabled (texture)
- [x] Photo textures (20 quads): crossbar top (NVIDIA logo), front port plate (RJ45, USB, LEDs, port labels),
      8 MMC fronts, wing, lip top, both outer walls, -X inner wall, both rims, divider front, both duct tops
- [ ] Wing shape: the real part is a curved handle with an oval slot; the wing texture contains the acrylic sign
      and plinth pixels (sign not modeled). Re-bake once the sign exists in the ID pass (owners exclude it).
- [x] divider_front texture contains upper-group tube pixels above the divider; re-baked on the Wave 2 ID pass
      (the wall now ends at Z -1.5 under the flange, so the tube band is no longer part of it); re-bake again
      after upper is final
- [ ] Wall thickness (assumed 1.2 mm), top-fold profile, rear wall notches and the slanted rear end of the
      side walls, rear black block on the +X wall (IMG_5741). Divider top flange: done in Wave 2.
- [ ] Brackets: front-bay silver rails at |X| 110-122 (orthos z = -3) tried and disabled (residual 82/75,
      points 7-8 mm off: height unknown); no other bracket confirmed
- [ ] mmc_g6/g7 edge 7-10 px (occluded near the +X lever in front views)
- [x] Switch `body.roll_deg` to the shared "tray" frame in config/frames.json when the coordinator adds it

Wave 2 (2026-10-06, coordinator brief; v1 holdout error budget: chassis 44.9 percent):
- [x] W2.1 Tray-frame switch: `body.roll_deg = 0`, parts built level, `lib.apply_frame(objs, "tray")`; bake.py and
      chassis_surfaces.py use the same frame (frames.json). Re-check walls and crossbar.
- [x] W2.2a Floor textures per bay (floor_front, floor_mid, floor_rear), rear wall inner face, +X inner wall
- [x] W2.2b Re-bakes in the tray frame; sign SAM mask excluded from all bakes; A/B ducts p25, K = 1 variants
- [x] W2.2c Duct inner side faces (textured); wing and lip from sign-mask views only (lip kept, wing rejected)
- [x] W2.3a Divider: wall ends at a horizontal top flange (Z -1.5, Y 557-593, rounded ends), fit; flange textured
- [x] W2.3b Crossbar top Z +1 mm (train points); MMC depth 6 -> 1.5 tried and rejected
- [x] Final full-mode comparison (probe + own list), full re-bake on the final ID pass
- [ ] Wing: real curved handle with the oval slot (not started; the sign ghost stays in the wing texture: the
      sign-mask-view bake lost on the front list, 1658 -> 1930)
- [ ] Rear wall notches / slanted rear end: the -X side views (5755-5757) do not reach Y > 740; only IMG_5741
      sees the +X rear end (silver wall face ends at Y about 790, black block behind it): single view, not
      triangulated
- [ ] Rear-bay floor height: smooth plate, few train points (Z -18..+14 are cables/board edges), plane sweeps
      inconsistent; floor_rear (largest chassis item, SSE 3200 probe, mostly IMG_5742) stays patchy
- [ ] mmc_g4..g7 edges 7-10 px vs g0-g3 2-4 px (not depth: 1.5 mm protrusion lowered them only 1-1.7 px)
- [ ] Duct geometry: real ducts are two stacked channels with deep windows; flat top + sides only

## 2. Plan (as run)

1. Measure on orthophotos and edge profiles (top-down IMG_5749 at Z = 0; side views on X = +-219; front views
   on Y planes), sparse-point histograms; check by two- or three-view agreement.
2. Block-out, iterate on probe, then fit only where edges are trustworthy.
3. Cross-check body length (776 mm) and front panel height (87 mm).

## 3. Measurements (world = tray frame, mm)

3.1 Frame roll. The crossbar top points (471 points, Y 332-370, |Z| < 4) fit z = 0.0127 x + 0.018 y - 5.47:
the frame is rolled 0.73 deg about Y relative to the tray. Side walls agree: -X wall bottom/top edges at world
Z -65.7 / 17.8 (IMG_5755, 5756, 5757, 4 Y chunks each, spread 1 mm); +X bottom edge -60 to -62 (inner
views 5743, 5745, 5746), about 5 mm higher, as the roll predicts (5.6 mm over 438 mm). All chassis parts are
built in a local frame and rotated by `body.roll_deg = -0.73` (world z = local z + 0.0127 x).

3.2 Walls. Local Z: floor underside -63.0, rim 21.3 (-X edges give 20.6; +X sparse points about 22; fit 22.1).
Side-wall height 83.5-84 mm. Thickness 1.2 mm assumed (sheet metal; not resolvable in the photos).

3.3 Rear end. Edge profiles on planes Z = h over X -150..150 in IMG_5754, 5749, 5741 agree only at h = 20,
with edges at Y 785, 798-799 and 802-803 (rear lip with notches/tabs at Y 800-806). Fit (probe, edge loss):
805. Model: `y_rear = 803`.

3.4 Front panel. Port plate (MMC/RJ45 plane) at Y about 23 in the current frame (Y = 0 stays where it is, by
coordinator decision 2026-10-06; the plate is the physical panel face): RJ45 and USB X positions agree within 2 mm between
IMG_5711 and 5738 at Y = 23, not at Y = 0. Y = 0 in the frame is the mode of the front views' sparse points,
dominated by the acrylic sign in front of the panel. Grip bar: fit on 5735/5736/5738/5740 gives front -20.5,
top -34, bottom -51.75 (model -52). Sparse points at Y -32..-36, Z -50..-54 (X -147..139) gave fp 0.53 when
modeled (likely the sign base); rejected. Top lip at local Z about 22 (points at Y 32-38, Z 14-18 behind the
plate; edges in 5738/5740). MMC groups: edges at |X| 119-121, 143-147, 164-168, 186-189 and panel edge
214-219 on both sides (IMG_5740, 5711, 5735, 5736, 5738): 4 groups per side, pitch 22.5, 19 mm wide,
centers +-130.5, 153, 175.5, 198; Z -48..-10.

3.5 Interior. Crossbar top Y 331-371 (fit; points 328-372), flanges to Z -30 (assumed). Divider at Y 578 (points
3.4 mm median), top at rim. Front-bay black ducts: dark points X 126-212 both sides, top cluster local Z -6..-2
(model 128-217, top -3), Y 30-325. Middle-bay black baffles are modeled by the upper group (`upper.baffle_nx`,
`upper.baffle_px`); not duplicated here.

3.6 Colors (8-bit sRGB medians, 7x7 px patches at projected world points): crossbar top 164,170,161 (5749) /
154,156,148 (5751); +X wall inside 154,143,113 (5744, sunlit); -X wall outside is lawn-tinted (100,120,59,
5756), not used; bezel 170,155,126 (5738 sunlit) / 103,98,88 (5740 shade); duct tops 67,70,72; MMC 32,53,32.

3.7 Cross-checks against published values.
- Body length: port plate (Y 23) to rear lip (Y 803) = 780 mm vs 776 published (+0.5 percent, within the 1 percent
  scale budget). Y = 0 to rear = 803 (+3.5 percent): Y = 0 is not the panel face (3.4).
- Front panel height: not measurable directly (panel bottom edge hidden; the tray stands front-panel down).
  Side-wall height 83.5-84 mm; top lip to floor underside about 85 mm: 2-4 percent below 87 (more than 1 percent).
  Fact: measured as above. Assumption (coordinator, 2026-10-06): the difference is the absent top cover. The length check
  (+0.5 percent) argues against a scale error; the lever barrel spans Z -70..21 (91 mm, fit), so the front
  assembly may be taller than the walls.

## 4. Progress log

- 2026-10-05 23:10: v1 block-out from orthos (chassis_001): IoU 0.923, edge 5.25 px (stub box: 0.929, 5.59 px).
- 2026-10-05 23:25: frame roll found and applied, rim/floor measured (chassis_002): IoU 0.934, edge 4.37 px.
- 2026-10-05 23:32: fit shell (`outputs/fits/chassis_shell1`): y_rear 805, crossbar 331/371 accepted;
  z_bot -60 rejected (floor underside barely visible in probe; direct edges say -63).
- 2026-10-05 23:33: front fit with 8 parameters (`chassis_front1`) ran away (lever r 1, panel bottom -36):
  masks near the plinth; rejected. Restricted fits: grip (`chassis_grip1`), lever (`chassis_lever1`) accepted.
- 2026-10-05 23:45: colors, MMC pitch, grip split, lever arms removed (no photo evidence) (chassis_007):
  IoU 0.947, edge 3.19 px, 56 percent of model edges within 2 px.

Before/after per part (probe, fp / edge px / points median mm), chassis_001 -> chassis_007:

| part | 001 | 007 |
|---|---|---|
| crossbar_top | 0.00 / 4.7 / 1.3 | 0.00 / 3.0 / 1.2 |
| crossbar_fl_front / rear | 3.9 / 24.4 ; 5.6 / 17.8 | 3.4 / 2.5 ; 3.5 / 6.1 |
| rear_wall | 0.00 / 5.1 / 17.2 | 0.02 / 2.5 / 4.0 |
| wall_mx / wall_px | 3.4 / 2.4 ; 6.7 / 11.1 | 3.7 / 1.0 ; 4.4 / 2.0 |
| divider | 3.4 / 24.1 | 4.4 / 3.4 |
| duct_front_mx / px | 4.8 / 4.4 ; 5.9 / 9.9 | 3.6 / 1.8 ; 5.2 / 3.9 |
| front_grip | 0.22 / 6.8 / 6.9 | 0.01 / 5.8 / 3.0 |
| lever_mx / px | 0.21 / 6.7 ; 0.13 / 7.7 | 0.01 / 4.2 ; 0.01 / 5.6 |
| mmc_g0..g3 | 2.4-3.0 px | 2.2-3.0 px |

Remaining heat in probe views is mostly not chassis: the black plinth under the front panel is in the tray mask
(IMG_5751, right strip), and other exhibit hardware below the tray (IMG_5742, 5746 bottom-left).

Phase 1B (2026-10-06), probe, chassis_008 (1A geometry, flat colors, full mode) -> chassis_017:
whole model IoU 0.965 -> 0.966, edge 3.18 -> 2.92 px, color residual 54.2 -> 43.2, SSIM 0.349 -> 0.401
(other groups also changed). Per surface, flat part -> textured quad (color residual / SSIM):

| surface | flat (008) | textured (017) |
|---|---|---|
| crossbar top | 49.2 / 0.49 | 30.5 / 0.55 |
| front plate | 50.8 / 0.49 | 29.0 / 0.60 |
| MMC g0-g3 | 32.5-37.5 / 0.18-0.23 | 21.7-24.4 / 0.46-0.50 |
| MMC g4-g7 | 16.6-23.6 / 0.30-0.57 | 7.3-14.8 / 0.48-0.82 |
| wall +X out | 36.0 / 0.76 | 22.7 / 0.88 |
| wall -X out | 57.0 / 0.57 | 25.2 / 0.79 |
| rims | 59.8-60.5 / 0.29-0.37 | 39.0-42.7 / 0.24-0.41 |
| divider front | 57.4 / 0.40 | 45.2 / 0.43 |
| wing (grip) | 47.2 / 0.44 | 42.5 / 0.35 |
| lever heads | 47.6-48.8 | 31.9-32.2 (fit) |

Box parts that now carry a textured quad keep only their edge pixels in parts.json (their own residual rises).
Own view list (9 train views, chassis_own): IoU 0.922, edge 2.81 px, color residual 41.5, SSIM 0.408.

Texture recipe and A/B results (bake.py: ID-owned texels, facing cos > 0.25, best K train views per texel):
- Median of 5 vs darker percentile (p25, view whose luminance is nearest the 25th percentile): p25 lost on
  every metal surface (crossbar 40.6 -> 44.6, walls +0.2..1.8, lip +3.1, plate +1.8; chassis_010). Median kept.
- Single best view (K = 1) vs median of 5: better for the front plate (34.0 -> 29.7, SSIM 0.51 -> 0.60) and
  MMC g0-g3 (SSIM +0.08..0.11); worse for g4-g7. K = 1 kept where it won (registration blur in the median).
- Sessions: crossbar S3 beat S1 (40.6 -> 33.7); front plate, MMC, wing, lip from S1 (only S1 sees the front);
  walls, rims, ducts, divider from all sessions (all beat one session by 1-3 residual; chassis_012).
- Session per texture: crossbar_top S3; front_plate, mmc_g0..g7, wing, lip_top S1; wall_mx_in S3;
  wall_mx_out S1+S3 (texel share 0.96/2.09); wall_px_out S2 only in practice; rim_mx, rim_px, divider_front,
  duct tops mixed (shares printed by bake.py, TOML [tex.*]).
- Tried and dropped: crossbar flange faces (textured 60.9/56.1 vs flat 57.9/55.5), divider rear and rear wall
  inner face (75 percent unseen / structured), grip slab textures (sign and plinth bled in before the wing fix).
- Shared texture tools could not open this capture's originals (.JPG vs .jpeg) at the time; bake.py has its own
  sampler (finds the file by stem). The coordinator has since fixed the shared tools.

Lever fit (`outputs/fits/chassis_lever2`, 8 front/corner train views): barrel r 10.5, axis X = +-227.5,
Y 6, Z -70..21; head r 9, length 16, center 6 below the top. Clips: sparse points 0.2-1.4 mm from 3 of 4
clip boxes; height 7 (fp 0.07-0.16 at 9).

- 2026-10-05 23:50 to 2026-10-06 00:01: Phase 1B: chassis_008 (heads, clips), 009 (first textures), 010 (p25 / K = 1 A/B),
  011 (crossbar S3, plate K = 1), 012 (all-session A/B, wing to the levers, lip behind the plate), 013 (divider
  texture), 014 (rims, crossbar flanges), 015 (wing texture), 016 (lever fit, rails tried), 017 (final re-bake).

- 2026-10-06 00:22 (Wave 2, runs chassis_018-025, probe, full mode). Tools added: `tools/sse.py` (absolute
  photometric SSE per chassis surface family, error_budget method: color fit, blur 4 px; a box part and its
  textured quads compare as one), `tools/partcrop.py` (worst views of one part: photo | fitted render | ID),
  `tools/settex.py` (set a [tex.*] key for A/B). bake.py: `--sessions` override; sign SAM mask
  (`data/sam_s2/sign`, dilated 3 px at 1/4 scale) excluded from every bake (the sign is not modeled).
  - Tray frame (chassis_018 vs 017, textures not yet re-baked): parts within noise; MMC g0-g3 SSIM 0.50 -> 0.40
    until the re-bake (0.6 mm shift), restored in 019. Crossbar top roll from train points (|Z| < 3/4/6 mm,
    617-657 points, MAD 0.5 mm): 0.88 / 0.97 / 1.02 deg. 0.73 deg is not supported; 0.85 kept, no private
    correction. The same fit puts the crossbar top about +1 mm above Z = 0 (bins: x -160: -0.9 vs model -2.4;
    x 0: +1.0 vs 0; x 80: +2.4 vs +1.2). Walls in the tray frame: -X bottom -66.2 (audit -65.7..-66.4), +X
    bottom -59.7 (audit -59.8..-60.3).
  - SSE (1e6) per family, 018 -> 024 (all winners): floor 6381 -> 4838, rear 1035 -> 434, walls 2965 -> 2147,
    divider 1795 -> 1753, crossbar top 1751 -> 1660, ducts 6529 -> 6547 (025 p25: 6093); chassis 26737 -> 23519.
  - A/B (SSE): floor_front median all sessions 839 vs S1 900 / S2 934 / S3 852 / K1 911 (kept median);
    floor_mid 503 vs K1 493 / S1 504 (kept median; 97 percent of it is never seen); floor_rear K1 2818 vs median
    3062 / S1 3582 / S2 2931 / S3 3531 (K1 kept); wall_px_in all sessions 706 vs S1 747 / S3 1042 / S2 none /
    untextured 1515 (all kept); ducts p25 of 8 views 6093 vs median of 5 6547 (kept); crossbar top K1 1638 vs
    1660 (kept); lip K1 1245 vs 1064, divider K1 1850 vs 1753 (both rejected); rear_in K1 430 vs 434 (median kept).
  - rear_out (outer face of the rear wall): no camera behind the rear (all centers at Y < 803): skipped.
  - Rear bay floor: the bare +X floor (X 100-210, Y 600-800) is smooth; train points there are few and at Z -18..+14
    (cables, board edges); plane sweeps across 4 views gave no consistent height (featureless metal, periodic
    emboss). Floor kept at Z -62; the floor_rear bake stays patchy (sun/shadow differences between views).

- 2026-10-06 00:37 (runs chassis_026-034, own2). Lip and wing from the 7 train views with a sign mask
  (`sign_mask_views_only`): lip SSE front list 1768 -> 1101, probe 1064 -> 922 (kept); wing 1658 -> 1930 front,
  1233 -> 1122 probe (rejected; sum worse). Divider: orthos at Z = -5 of 5734 and 5754 show a horizontal silver
  flange with rounded ends (front edge Y 555-557, |X| <= 185-190); plane sweeps over the band peak at Z -6..+4;
  fit `outputs/fits/chassis_divflange1` (z_top, flange_y0, flange_y1; 8 train views): -1.6 / 557.25 / 593.0, edge
  3.10 -> 1.36 px. The wall now ends under the flange (was up to the rim 21.3). divider SSE 1763 -> 1074, edge
  4.2 -> 2.3 px, flange edge 1.8 px, whole model 48922 -> 47455 (chassis_029 vs 028). Crossbar top Z 0 -> +1:
  SSE 1691 -> 1536, texture-quad points median 1.7 -> 0.9 mm, rear flange 5.8 -> 4.7 mm (030). MMC depth 1.5:
  MMC SSE 364 -> 256 but wing +202, plate +54 (more of them exposed), chassis +140: reverted. Duct inner side
  faces textured: ducts 6046 -> 5657 (033). Final ID pass and full re-bake (034): within 0.2 percent of 033.

End of Wave 2, SSE per family (1e6, tools/sse.py; probe 8 views and own list 9 views; other groups changed too,
so compare chassis rows):

| family | probe 017 | probe 034 | own | own2 |
|---|---|---|---|---|
| ducts | 6535 | 5654 | 5200 | 4507 |
| floor | 6391 | 4793 | 4009 | 2746 |
| walls | 2970 | 2057 | 4178 | 4072 |
| divider | 1789 | 983 | 2121 | 991 |
| crossbar top | 1751 | 1590 | 2135 | 1634 |
| crossbar flanges | 1337 | 1351 | 1106 | 1102 |
| wing | 1239 | 1217 | 482 | 479 |
| lip | 1090 | 958 | 1013 | 813 |
| rims | 1049 | 1051 | 1280 | 1253 |
| rear wall | 1037 | 412 | 729 | 428 |
| front plate | 891 | 828 | 1047 | 946 |
| levers / MMC | 392 / 250 | 390 / 261 | 231 / 69 | 231 / 75 |
| chassis total | 26721 | 21545 | 23598 | 19275 |
| whole model | 52832 | 43614 | 48733 | 40500 |

Whole-model report lines (probe 017 -> 034): IoU 0.966 -> 0.978, edge 2.92 -> 2.91 px, color residual 43.2 -> 40.2,
SSIM 0.401 -> 0.411, points median 1.19 -> 1.15 mm, orphans 0.164 -> 0.157 (all groups).

Textures (assets/textures/chassis_<name>.png; sheets outputs/textures/chassis_<name>_views.jpg): new in Wave 2:
floor_front, floor_mid, floor_rear, rear_in, wall_px_in, divider_top, duct_front_mx_in, duct_front_px_in; all
others re-baked in the tray frame with the sign mask excluded. Sessions per surface are in the TOML [tex.*].

### Wave 3 (2026-10-06, coordinator brief: sign masks in 22 views; details)

- [x] W3.1a Re-bake wing / lip / front plate / MMC with the sign excluded in all 20 train views that have a mask
- [x] W3.1b Wing nose (points at Y -31, Z -48 along X -162..+187) tried rounded and as a textured facet: rejected
- [x] W3.2 floor_rear height sweep: rear-bay deck at Z -13.8 (dz 48); floor_front sweep: stays on the pan
- [x] W3.3a Duct tops: darker statistics tried (p10 of 8, S3-only p25): both worse; p25 of all sessions kept
- [ ] W3.3b Duct channel geometry: not built (the texture registers well; the residual is sun/shade brightness
      between sessions, which one texture cannot carry)
- [ ] W3.3c mmc_g4..g7 edges: fit of z0/z1/depth ran away (z0 -48 -> -33, loss 5.57 -> 5.38): rejected; unresolved
- [ ] W3.3d Rear notches / slanted rear end: no new evidence (+X rear end only in IMG_5741; -X side views stop at Y 740)
- [ ] W3.4 Fasteners: screw heads on walls, crossbar and divider are in the photo textures; no geometry (heads of
      about 1-2 mm protrude below the edge resolution in these views; not measured)
- [x] Final runs (probe chassis_048, own list chassis_own3)

- 2026-10-06 00:56 (runs chassis_035-045, chassis_fz*). Sign-excluded re-bakes (037/038 vs 035/036; untouched
  families moved up to 10 percent between runs from the per-view color-fit coupling, so small deltas are noise):
  wing 1564 -> 1549 front list, 1196 -> 1192 probe (kept: no sign ghost); lip from the flag's 20 views 1040 -> 1687
  (rejected; lip pinned to its 7 old views with the new `views` key); front plate and MMC unchanged within noise
  (old bakes kept). Wing nose (front.grip profile to Y -30.5, Z -49): IoU 0.961 -> 0.973, grip edge 6.1 -> 5.4 px,
  but grip fp 0.20 -> 0.29 (nose facet fp 0.54: SAM masks call it background) and wing family (grip + wing quad +
  unmodeled) 1833 -> 1886-1888 front list, 2757 -> 2754-2757 probe: rejected, profile reverted (TOML comment).
  floor_rear: textured plane height sweep (new `dz` key; ID pass + median-of-5 bake + 7 rear-bay train views per
  height): dz 0/10/25/38/42/45/48/51/55 -> floor family 7667/6588/5190/4046/3765/3581/3531/3606/4042, whole model
  44674 -> 38788 at dz 48. The visible rear-bay surface is a deck at local Z -13.8 (train points at Z -15..-11, X
  120-180, Y 740-760; 5749/5754 plane sweep peak -16), not the floor pan. K1 3433 vs median 3531 (in-sample,
  within noise; median kept); S1/S2/S3 alone 5927/3707/5242. Upper rear-bay parts keep identical ID pixel counts
  (none hidden). Probe 044 -> 045: floor 4762 -> 2655, walls 2044 -> 1594, chassis 21483 -> 18821, whole model
  43224 -> 40246. floor_front dz 0/15/30: 842/910/4043 (dz 30 hides lower-group parts): stays at 0.

- 2026-10-06 01:05 (runs chassis_046-048, own3; fit `outputs/fits/chassis_mmcz1`). Duct tops p10 of 8 / S3-only
  p25: ducts 5638 -> 5797 / 5969 (both rejected). MMC fit (z0, z1, depth; 8 front train views) ran away to z0 -33
  for a 3 percent loss gain: rejected. Final ID pass and bakes match the final geometry.

End of Wave 3, SSE per family (1e6; probe and own list; other groups changed too):

| family | probe 034 | probe 048 | own2 | own3 |
|---|---|---|---|---|
| floor | 4793 | 2647 | 2746 | 2151 |
| walls | 2057 | 1595 | 4072 | 2933 |
| rear wall | 412 | 300 | 428 | 298 |
| wing | 1217 | 1175 | 479 | 414 |
| ducts | 5654 | 5641 | 4507 | 4496 |
| divider | 983 | 1023 | 991 | 990 |
| crossbar top | 1590 | 1600 | 1634 | 1649 |
| chassis total | 21545 | 18815 | 19275 | 17352 |
| whole model | 43614 | 39935 | 40500 | 38025 |

Whole-model report lines, probe 034 -> 048: IoU 0.978 -> 0.979, edge 2.91 -> 2.89 px, color residual 40.2 -> 38.7,
SSIM 0.411 -> 0.416, points median 1.15 -> 1.14 mm, orphans 0.157 -> 0.154.

### Wave 4 (2026-10-06, coordinator brief: v2 holdout budget chassis 41.2 percent; divider top 5.3, crossbar top 4.6,
duct tops 3.5/2.5, floor_rear 3.2, rear_in 2.7)

- [x] Registration check (partcrop on probe): crossbar logo, ribs and divider flange edges line up with the photos;
      the residual is sun/shade brightness between sessions (S1 sunlit, S3 shaded) and flat-colored rounded flanges
- [x] Bake A/B (probe chassis_049 baseline vs 050A-D; own list own4 vs own4C/D): no variant kept
- [ ] Fasteners: not attempted (no edge gain expected from flat screw heads already in the textures)

- 2026-10-06 01:10. Wave 4 SSE per surface (1e6, probe; baseline 049 = Wave 3 bakes):

| surface | baseline | A: K1 / darkest view | B: p25 of 8 (ducts p25 of 5) | C: S1 only | D: S3 only |
|---|---|---|---|---|---|
| divider_top (median 5, all) | 914 | 980 | 1134 | 1237 | 1115 |
| crossbar_top (S3, K1) | 1573 | - | 1871 | 1396 (S1 K1) | 1529 (S3 median 5) |
| rear_in (median 5, all) | 299 | 294 | 427 | 576 | 398 |
| duct tops mx / px (p25 of 8) | 2069 / 2609 | 2156 / 2878 (p0) | 2190 / 2527 | - | - |

  Crossbar S1 K1 won on probe but lost on the own list (tex_crossbar_top 1636 -> 2561; S3 median 5: 1805): probe
  holds the S1 views it samples from (in-sample), so the S3 K1 bake stays. rear_in K1 294 vs 299 is within the
  run-to-run noise: median kept. Model and textures unchanged from the end of Wave 3 (chassis_048 / own3; the
  baseline reruns 049 / own4 match within 0.1 percent).

### Wave 5 (2026-10-06, coordinator brief: geometry; validate every keep on probe and own list)

- [x] Crossbar rounded long edges (`crossbar.edge_r`, `edge_n`; thin-shell Y-Z profile extruded along X): rejected
- [x] Duct outer edge x_out 217 -> 212 (dark train points end at X 212): rejected (wash)
- [x] Regions sheet (probe chassis_051): #1-3, #5 = mask leaks below/beside the tray (request 3), #4/#7 duct
      tops, #6/#8 crossbar; no other chassis geometry indicated
- [ ] Duct channel geometry (windows, groove): not built; the texture registers (regions #7) and the measured
      x_out change did not score

- 2026-10-06 01:25 (paired runs; another agent re-baked upper textures 01:15-01:18, so A/B runs were made back to
  back and only chassis rows compared). Crossbar points (train, |X| < 205): top at Z +0.6..+1.7 over Y 328-372 with
  steps under 1 mm; bright points reach Y 322-326 (rounded lips). Fit `outputs/fits/chassis_xbar_r1` (edge_r, y0,
  y1; 8 train views): r 6.9, y0/y1 330.6/370.75, edge 2.16 -> 1.72 px. Arc r 7 (flat-colored arcs, top texture
  re-baked on the narrower flat): crossbar top + flanges probe 2963 -> 2842, own list 2783 -> 2952; whole model
  +583 / +999; part edges top 3.6 -> 1.3 px, flanges 4.1 -> 2.9 / 3.6. 45 deg chamfer with photo-textured chamfer
  faces: worse than the arc on both (probe 2994, own 3029). Rejected on the own list: `edge_r = 0` (code kept).
  Duct x_out 212 (re-baked duct tops/sides, inner walls, floor_front): chassis probe 18770 -> 18725, own
  17370 -> 17374 (ducts -2 / -1 percent, walls +5 / +1.5 percent): rejected. Final state = end of Wave 4 (crossbar top
  re-baked at r 0 on a fresh ID pass): chassis_056 / own10 vs 051 / own5: chassis 18812 -> 18790, 17352 -> 17335.
- Holdout disclosure (audit 2 rule, DevLog-005): in Wave 2 I read the v1 `outputs/snapshots/v1/error_budget_holdout.json`
  (top-part list, as cited in the coordinator's brief) to pick surfaces; Wave 3-4 priorities came from the
  coordinator's v2 holdout budget text (I did not open that file). No holdout photos, crops, renders or runs were
  opened; bakes, fits, sweeps and the sign hull used train views only (the hull skipped IMG_5737).

### Wave 5b (2026-10-06, coordinator task from upper DevLog 10.2: bright strip in IMG_5734, dark in 5729)

- [x] Diagnose: strip pixels (photo 190 / render 67, 98 x 591 px at 1/4 scale) cover upper.mid_pcb and floor_mid;
      rays through them hold Y 372-390 for any Z (X 40-200): the band right behind the crossbar, not the +X wall
- [x] Measure: plane sweeps (5729/5734/5749/5751) over Y 374-384 peak at Z 0 / 2 / 4 / 4-6 for X -160 / -80 / 0 /
      80-160 (pair NCC up to 0.93); Y 384-402 no peak; train points Z about 3 over Y 372-381, 0 at 384
- [x] Model: crossbar rear ledge `chassis.crossbar_ext` (Y 371-381.5, top Z +3.5; fit `outputs/fits/chassis_xbar_ext1`,
      edge 3.54 -> 1.52 px), rear flange moved to its edge, ledge photo-textured (crossbar_ext, S3, K1)
- [x] First hypothesis, a +X-wall shelf at Z -25 (train points X 177-217 Z -25): neutral, disabled

- 2026-10-06 01:38. Paired A/B (back to back), chassis + upper SSE (1e6): shelf 29690 -> 29712 probe, 28201 ->
  28185 own (neutral, disabled). Ledge (y_ext 385 / z 3): 29694 -> 27653, 28184 -> 25542. Fitted ledge (381.5 / 3.5,
  chassis_059 / own13): 29703 -> 27174 (-8.5 percent), 28156 -> 25212 (-10.5 percent); upper.mid_pcb 2266 -> 980 /
  2115 -> 883; whole model 38013 -> 35408 / 37971 -> 35125; crossbar_fl_rear edge 4.1 -> 3.6 px; probe IoU 0.986,
  color residual 37.7. Kept. The 5729 reverse strip was not diagnosed separately (mid_pcb drops in both lists).

### Wave 5d (2026-10-06, coordinator task: unmodeled gray sheet metal along the +X wall, rear bay)

- 2026-10-06 01:50. Train points X 189-201 (5-95 percent), Z 12-15 (median 14.2), Y 590-760 (per 25 mm Y bin: X
  191-201, Z 12-16); a second group at X 205-217, Z 18-21 is the rim fold. Plane sweeps (5741/5742/5748/5749/5754)
  over X 186-197 peak at Z 6-8 (upper hose_px nearby), over X 176-186 flat: no independent height. Overlay on
  IMG_5741: the band lies between the black hose and the wall rim; one view only. Modeled `chassis.rail_rear_px`
  (X 187-201, Y 595-765, top Z 13.5, textured top): part points median 1.2 mm, edge 2.9 px, but paired A/B chassis +
  upper SSE 27103 -> 27442 (probe, chassis_060) and 25101 -> 25292 (own list, own14): rejected, `enabled = false`.

### Wave 6 (2026-10-06, detail wave toward v4; accept if neutral on probe and own list, reject if both regress)

- [x] Crossbar front ledge `chassis.crossbar_ext_front` (lower agent's request: fibers vanish at Y about 315)
- [x] RJ45 x4 and USB openings as real recesses (holes cut through the plate, dark open cavities `chassis.port_<i>`,
      plate photo texture split into cells around the holes); port positions re-measured in the plate texture
- [x] Unseen faces in the photographed wall/floor color (untextured on purpose): all downward faces of chassis
      parts, the +Y faces of front_grip and rear_wall (`[unseen]`, `colors.unseen`)
- [ ] LED row, MMC port faces per row, lever hook/pivot refinement: not done (budget)
- [ ] Fastener relief: not done. The audit's wall features are flush holes (in the wall textures); no raised screw
      heads were measured, so none were added

- 2026-10-06 02:41. Front ledge: band sweeps (6 train views) Y 318-324 lock at Z +2 (mean pair NCC 0.84 X -180..-60,
  0.80 X 60..180); Y 312-318 only fibers (Z -26); train points Y 324-331 at Z -2..+2. Fit `outputs/fits/chassis_xbar_extf1`
  (y_ext_front, z_ext_front; 8 train views): 320.5 / 1.6, edge 1.87 -> 1.48 px. Paired A/B (no ledge = y_ext_front
  331; a first baseline with 0 was invalid: it built a ledge from Y 0): whole model 34939 -> 32709 probe, 34881 ->
  32452 own; chassis + upper 27098 -> 26106 / 25163 -> 23782; lower group 6170 -> 4915 / 5771 -> 4493. Kept.
  Unseen color: median of wall_px_in (83,82,74) and floor_front (86,82,76) -> (84,82,75). With front_plate and
  front_lip back faces included: chassis +100 / +117 (their +Y faces are seen from top views: +55 / +22 on
  front_plate / front_lip); without them: +15 / +9 (neutral): kept without them. Port recesses: first with the old
  TOML positions the holes sat off the photographed openings (crop IMG_5738); dark openings located in the plate
  texture: RJ45 centers X 8.0 / 32.2 / 54.6 / 76.9, 12.5 x 9.5 mm at Z 9.4; USB X -12.8, 13 x 5.5 at Z 6.1. Then
  front_plate + ports 852 -> 861 probe, 952 -> 953 own; chassis 16932 -> 16926 / 14941 -> 14928 (neutral): kept.

End of Wave 6 vs its baseline (chassis_061a / own15a -> chassis_065 / own19, SSE 1e6):

| | probe | own list |
|---|---|---|
| whole model | 34939 -> 32728 | 34881 -> 32408 |
| chassis + upper | 27098 -> 26121 | 25163 -> 23746 |
| chassis | 17805 -> 16926 | 16144 -> 14928 |
| lower | 6170 -> 4916 | 5771 -> 4484 |
| edge px / color residual / SSIM | 2.86 -> 2.83 / 37.6 -> 36.8 / 0.422 -> 0.427 | 2.69 -> 2.63 / 35.9 -> 34.9 / 0.434 -> 0.441 |

### Wave 7 (2026-10-06, front panel; accept if neutral on probe and own list)

- [x] MMC groups re-measured and moved (were about 5 mm too far inboard on both sides, 19 mm wide)
- [x] Front-plate photo texture extended down to Z -49 (cage port "65" at X about -118..-100, panel face between groups)
- [x] MMC per-row port openings as shallow recesses: tried (dark and green backs), rejected; code kept (`recess = false`)
- [ ] LED row: left to the photo texture (not crisp enough to place lenses); lever hook/pivot: not done (budget)

- 2026-10-06 02:51. MMC: a diagnostic strip texture over the whole MMC row (`[tex.mmc_strip]`, bake only; single best
  S1 view, 8 px/mm) gives green extents -X -211.5..-194.5 / -189.1..-172.0 / -166.4..-149.5 / -144.5..-127.4 (centers
  -203.0 / -180.6 / -157.9 / -135.9, width 17.0-17.2) and +X centers 133.2 / 156.3 / 177.8 / 201.0 (blurred side; pitch
  22.4 from 201.0: 133.8 / 156.2 / 178.6 / 201.0), green Z -51.9..-9.0. New x_centers, w 17, Z -51.5..-9.0, MMC and plate
  re-baked; paired A/B (chassis_066 / own20): chassis 16927 -> 16903 / 14929 -> 14948 (neutral), MMC 262 -> 242 /
  75 -> 58, -X box edges 3.8-4.6 -> 3.2-3.5 px (+X 4.5-9.3 -> 5.1-9.4): kept. Row separators in g0-g3 at Z -15.5 /
  -28.0 / -40.5 (pitch 12.5). Row recesses (frame 1.5, depth 4): chassis +40 / +23 (dark back), +57 / +27 (green back)
  vs none (chassis_067 / own21): rejected. Plate texture to Z -49 (paired chassis_068 / own22): front_plate 836 -> 771
  / 965 -> 908, chassis 16904 -> 16832 / 14956 -> 14887: kept. Final: chassis_068b / own22b.

## 5. Requests to coordinator

1. World frame roll: crossbar top slope 0.0127 in X (0.73 deg about Y); walls confirm (lower agent: 0.65 deg).
   Decision 2026-10-06: world frame not rotated; a shared "tray" frame will carry the roll and replace
   `body.roll_deg`. Z values in this devlog are given as local (roll-corrected) and world where they differ.
2. Y origin stays (decision 2026-10-06): port plate face at Y = 23 in the current frame.
3. Masks: IMG_5751 includes the black plinth along the front panel; IMG_5742/5746 include hardware below the
   tray. Both show as missing geometry for the chassis.
4. (Wave 2) Sign masks for 5721-5726: done by the coordinator (22 views); wing re-baked without the sign (Wave 3).
6. (Wave 3) The wing's lower front edge at Y -31, Z -48 (train points along X -162..+187) is called background by
   the SAM tray masks in front views; if the masks are wrong there, the rejected nose profile (TOML comment) would
   be right. Worth a mask check at the front lip.
7. (Wave 3) Rear-bay deck at Z about -14: coordinator decision 2026-10-06: chassis owns it (structural sheet metal).
   Wing nose: coordinator checked the masks; the Y -31 / Z -48 points are most likely the plinth edge. Rejection stands.
5. (Wave 2) Crossbar top is at Z +1 in the tray frame (train points). If the frame origin is ever re-derived,
   Z = 0 at the crossbar top would move by 1 mm; no change needed now.
