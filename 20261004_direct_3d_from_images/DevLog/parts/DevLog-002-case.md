# DevLog-002-case: black plastic housing, window insert, card, junction bracket

| Field | Value |
|---|---|
| Group | case |
| Owner | case modeling agent |
| Files | `scripts/model/case/build.py`, `config/model/case.toml` |
| Started | 2026-10-05 |
| Phase | 1A done; 1B (details, materials) done 2026-10-05 |
| Last run | `outputs/runs/case_030` (probe, full; Phase 2 photo textures) |

## TODO
- [x] Read AGENT_GUIDE.md, DevLog-001 sections 2 and 4
- [x] Measure the screen/tray junction directly (resolves the 94 vs 110 mm fit ambiguity: 110)
- [x] Screen box, tray, separate +X end wall, bracket bar + screw tabs + 2 screws, clear window insert, dark window edge, white card
- [x] One fit_params run for the main dimensions (`outputs/fits/case_dims1`), one for wall heights (`outputs/fits/case_tray2`, rejected for the long walls)
- [x] Coordinator request (pcb finding): tray long-wall height re-measured, lowered 28.5 -> 26.0
- [x] 1B: tray wall heights: far + end 27.7 (tri2 train views only IMG_1587 + IMG_1594 + IMG_1586 on the +X+Y outer rim edge: (99.64, 49.84, 27.63), err 0.9-1.4 px; the earlier 28.0 used holdout IMG_1595 and is withdrawn), near 25.0 (overlays IMG_1512/1513 + ray picks 24.3-25.3; the near +X corner is hidden by wires, tri2 not possible)
- [x] 1B: junction ramps on both long walls, junction lugs (bracket mounts), wall flare (draft), larger corner radii, -X end slot, near-wall bottom notch
- [x] 1B: window flange clips (2 on -X flange, 1 on +Y flange), bracket narrowed and moved (x -5.3..6.7) with screw pads and 2 holes
- [x] 1B: materials tuned for the white eval world (black spec 0.02, rough 0.3, bluish base; dark alpha glass)
- [x] Promoted helpers: `scripts/tools/overlay.py`, `scripts/tools/rayplane.py`, lines in AGENT_GUIDE.md "Measuring"
- [ ] Next: rods inside the window (two thin rods along X, IMG_1527), window rounded corners and lighter flange appearance, window_edge band shape
- [ ] Next: card edge error 6.8 px (near/+X edge slightly off; the +X edge overlay in IMG_1571 shows card y extents fine, x0 edge ~2 mm short)
- [ ] Next: photo textures for walls (scuffs, molded text) only if needed; color residual of black parts is dominated by mat reflections

## Measurements (world mm, provenance)

| Item | Value | How |
|---|---|---|
| -X end / +X end | x -102.8 / 99.6 | sparse dark points (x -103..-100 / 97-98), then fit_params case_dims1 (length 202.4, cx -1.6) |
| Width | 99.5 (y +-49.75) | fit case_dims1 (start 99.0); sparse wall points y -50..-51 / 49..50 |
| Yaw | -0.8 deg | fit case_dims1 |
| Screen rim z | 35.75 | fit; IMG_1543 overlay; sparse 34.5-35.3 at x -105..-30 |
| Junction | x ~7.2 (screen length 110) | IMG_1513 near-wall ray picks on plane y=-49.75: top ends x 4.6 z 35.1, slope to x 10.9 z 25.3 |
| Tray long walls z | 26.0 | IMG_1513 ray picks x 71-92: 23.8-26.5 (lip band); pcb agent: plate top 28.4 and flyback top 26.9 visible above the wall |
| Tray +X end wall z | 28.0 | IMG_1587 / IMG_1604 overlays (bright rim near z 28); fit case_tray2 27.75. Unconfirmed |
| Bracket screws | (1.7, -47.9, 37.1), (2.9, 47.9, 37.6) | own least-squares tri of IMG_1518 + IMG_1571 picks, err 0.4-1.3 px |
| Bracket bar | x -12.5..4.5, top z 37.5 | IMG_1560 top-view overlay; sparse points x -11..7 median z 37.1-37.8 |
| Window insert +X edge | x -15.6 | IMG_1518 ray at z 37; overlays IMG_1518 / IMG_1638 match the flange outline |
| Window +X raised edge | x -17..-12.5, z 37.2 | sparse points x -20..-11 median z 36.7-37.3 |
| Card (tilted ~11 deg, higher at -X) | x -93..-32.25, y -37..41.5, z 26.2 (-X) .. 14.3 (+X) | tri IMG_1527 corners (-32.4, 39.8, 14.3) 4 inliers and (-91.0, 40.5, 25.75) 4 inliers; extents from rays onto that plane (IMG_1518, IMG_1638) |

## Progress
- 2026-10-05 00:10 Read guide/devlog; pilot run case_p1 (IoU 0.874, edge 3.98 px).
- 2026-10-05 00:15 Built scratch helpers in `outputs/scratch/case/`: `ov.py` (draw world-mm wireframe hypotheses on a view with grid), `rayp.py` (pixel ray to any axis plane), `tri.py` (plain LSQ triangulation with per-view error), `pts.py` (sparse points to world mm). Top/side scatter of sparse points gave walls, rim profile, bracket zone.
- 2026-10-05 00:19 case_001: screen box, tray, bracket, screws, window (open frustum tray, solidify), tilted card.
- 2026-10-05 00:25 Bracket moved to x -12.5..4.5 with screw tabs (IMG_1560 top view).
- 2026-10-05 00:30 fit_params case_dims1 (7 params, 100 evals, 59 s): width 99.5, length 202.4, cx -1.6, screen 35.75, tray 28.5, yaw -0.8. Accepted. case_002: tray pts 0.71 -> 0.55 mm.
- 2026-10-05 00:38 Card extents enlarged (case_003). Glass rendered opaque gray: EEVEE Next without scene raytracing refracts only the world probe; switched to alpha 0.15 BLENDED (case_004).
- 2026-10-05 00:50 case_005 after the coordinator's tool fixes: numbers unchanged for case parts.
- 2026-10-05 01:05 case_006: window raised +X edge part, bracket top 37.5. window pts 1.31 -> 0.55, bracket 0.74 -> 0.45.
- 2026-10-05 01:25 Coordinator/pcb request: tray walls too tall. Split +X end wall (`case.tray_end`). fit case_tray2 prefers 28.5 for the long walls (it locks onto plate/flyback edges); direct picks say 24-26.5; set 26.0. case_007: tray edge 3.17 -> 4.11 px, pts 0.55 -> 0.60 mm (conflict noted; trust direct picks + occlusion evidence).

- 2026-10-05 01:40 Coordinator: new RGB rig (uniform white world + weak key + photo-textured mat). case_008 (full): case rendered gray (white-world specular). case_009: black roughness 0.08 with Specular IOR Level 0.2, glass base 0.25 alpha 0.12, card 0.80. Case now reads near black, glass dark-clear. pcb build failed in case_008/009 (BUILD_RESULT pcb False, not case code): tray pts rose 0.60 -> 1.92 mm because pcb points fall back to the tray.

- 2026-10-05 02:00 Phase 1B start. Promoted overlay/ray tools. Measured: bracket (IMG_1560 rays z 37.5: x -5.3..6.7, holes at y -41.7/-23.2), flange clips (IMG_1543/1555 rays z 36), -X end slot (IMG_1555 rays x = -102.2), near-wall notch (IMG_1513), tray rim heights (overlays IMG_1595/1587/1512/1513).
- 2026-10-05 02:20 case_010: rebuilt walls as rounded prisms, separate near/far/end heights, ramps, clips, bracket pads/holes. IoU 0.936, edge 3.06.
- 2026-10-05 02:35 fit case_dims2 (w-fn 1 after mask fix): width 100.0, length 201.4, cx -1.475. Material tuning vs photo wall pixels (`outputs/scratch/case/bright.py`, median gray on eroded part masks): spec 0.12 -> 0.06 -> 0.02 (case_011-013); renders now 22-75 vs photo 1-58 (residual from mat reflections the white world cannot give).
- 2026-10-05 03:10 case_014: slot + notch. case_015/016: wall flare 1.2 mm at the rim (sparse dark points z 30-37 at |y| 51.2 vs base 50.0-50.7; fit case_flare gave 0.68 but also shifted cy, not used). screen_box pts 1.14 -> 0.82.
- 2026-10-05 03:40 case_017: junction lugs (sparse points at |y| 51.3-53.2, x 4.5-13, z 31-36 were attributed to the ramps): ramp pts 2.4 -> 0.8-1.1 mm, screen_box 0.74.
- 2026-10-05 04:00 case_018: corner radius 6 (+flare), IMG_1571 overlay. Holdout check (14 views, deleted after): IoU 0.927, edge 2.93 px, sparse median 0.39 mm; case parts: screen_box edge 4.32 / pts 0.73, tray 3.45 / 0.50, window 3.95 / 0.66, card 6.59 / 0.77, bracket 4.46 / 0.64.

- 2026-10-05 04:40 Audit fix (holdout leakage: IMG_1595 is holdout). Re-measured the +X+Y rim corner with train views only -> z 27.6; far/end heights 28.0 -> 27.7. Card -X edge extended 2 mm (case_020): worse (edge 6.98 px, pts 0.90), reverted. Added `case.window_rod` (pick.py tri on IMG_1527 rod: (-15.1, 5.3, 14.1) 6 inliers, (-16.8, -14.6, 14.1) 3 inliers); it renders in the ID pass but gets no row in parts.json (too few visible pixels?), and may be the window's +X bottom edge rather than a separate rod. case_021 (new evaluator, untruncated point medians): IoU 0.940, edge 3.08 px; screen_box 4.71 px / 0.76 mm, tray 3.64 / 0.48, card 6.76 / 1.05, window 4.45 / 0.60, ramps 7.2-7.3 / 0.76, bracket 4.57 / 0.64.

## Phase 2 (appearance, 2026-10-05)
- [x] Photo-texture panels: `scripts/model/case/case_panels.py` (panel quads in case-local mm, shared by build and bake), `scripts/model/case/bake_case.py` (ranks train views per panel by facing/distance, holdout excluded via data/split.json, calls scripts/tools/bake_id_owned.py with the ID-owned mask, n_best 5, 4 px/mm = 0.25 mm texels). build.py assigns panel materials + planar-projection UVs per face (normal within ~35 deg, centroid within 2.5 mm of the panel plane). 35 panels: screen/tray outer walls, rim tops, tray interior walls, bracket + pads, window_edge, card, window shell (inner walls split into 6 segments per long side, ends, flange, bottom).
- [x] ID renders of all 123 train views: `outputs/runs/case_idtrain` (re-render after geometry changes, then rebake).
- [ ] Next: ramps, lugs, clips, window_edge sides still flat black; lamp reflection on the window is view-dependent (largest remaining window error).
- 05:00 baseline case_022: psnr_fit 13.89, blur4 15.58; case 62.9 percent of error (window 25.6, tray 10.5, screen_box 8.4, bracket 6.7, card 5.2).
- 05:15 case_023 opaque window_top quad (coordinator's suggestion): worse, 12.65 (window_top 10.0 dB: parallax of the card 10-20 mm below, lamp reflections). Disabled (`window_tex`/`window_top` flags).
- 05:25 case_024 card texture: 14.06. case_025 glass spec 0.05: 13.96; case_026 light glass (0.3, alpha 0.35): 13.71. Reverted to dark glass.
- 05:40 case_027 window shell textured (opaque): 14.21. case_028 window bottom made parallel to and 1.5 mm below the card (old bottom cut the card's +X half): 14.29. case_029 tex_spec 0.2 -> 0.0 (walls were 10-25 levels too bright): 14.53 / blur4 16.53. case_030 n_best 3 -> 5: 14.58 / 16.59.
- Holdout check once (not tuned on): case_031h psnr_fit 14.09, blur4 16.19 (3DGS 22.1); case 53.7 percent (window 19.2, tray 10.2, card 8.5). Run deleted.

## Final metrics Phase 1B (case_018, probe, full, all groups built)
- Global: IoU mean 0.940 (min 0.869), edge mean 3.06 px, color residual 37.3, sparse median 0.44 mm.
- Per part (edge px / pts median mm): screen_box 4.74 / 0.73; tray 3.61 / 0.50; ramps 7.1-7.4 / 0.76-1.08; lugs 2.6-4.1 / 0.36-0.53; bracket 4.58 / 0.64; pads 2.7-3.5 / 0.30-0.37; screws 1.06-1.12 / 0.21-0.26; window 4.51 / 0.66; window_edge 3.60 / 1.02; card 6.76 / 0.77; clips 4.9-6.3.

## Final metrics Phase 1A (case_009, probe, full; pcb group missing from the build)
- Global: IoU mean 0.905 (min 0.776), edge mean 3.27 px, color residual 40.7, sparse median 2.16 mm (pcb missing).
- Per part (edge px / color res / pts median mm): screen_box 4.65 / 29.2 / 1.42; tray 3.73 / 49.3 / 1.92; tray_end 3.40 / 46.9 / 0.91; bracket 4.43 / 38.6 / 0.44; tabs 3.08-3.64 / 0.24-0.60; screws 1.48-1.49 / 0.13-0.34; window 4.64 / 47.1 / 0.55; window_edge_px 3.32 / 0.56; card 6.77 / 28.8 / 0.91.

## Metrics before the rig change (case_007, probe, geom, all groups built)
- Global: IoU mean 0.873 (min 0.729), edge mean 3.58 px, sparse median 0.62 mm.
- Per part (edge px / pts n / pts median mm): screen_box 4.65 / 582 / 1.09; tray 4.11 / 482 / 0.60; tray_end 3.40 / 386 / 0.61; bracket 4.93 / 686 / 0.45; tabs 2.83-3.27 / 111-114 / 0.24-0.45; screws 1.45-1.53 / 119-128 / 0.13-0.34; window 4.93 / 399 / 0.55; window_edge_px 2.78 / 317 / 0.56; card 6.68 / 36 / 0.91.

## Tool notes
- Ramps' edge error (7 px) is mostly occlusion by the bracket pads/lugs and wires in the probe views; their 3D distance is fine.
- `fit_params` with w-fn 1 trades width against cy when the flare is free; prefer sparse-point profiles for wall geometry.
- Mask: the glossy black case mirrors the blue mat, so case side faces are labeled mat -> false magenta (screen_box fp 0.084; regions in IMG_1543, IMG_1513 always rank first). fit_params fp term is biased toward shrinking for the same reason (used w-fp 3).
- Orphans/cyan near the tray are the external brown PCB and wires (x > 100 or |y| > 55), attributed to case.tray / case.tray_end as "nearest"; mat points just below z 0 next to the near wall also count toward case parts.
- `pick.py` positional args: negative coordinates need `--` (argparse); zsh does not word-split `$p` (use `${=p}`).
- `pick.py tri2` crashes (LinAlgError) when all observations are outliers (inconsistent picks) instead of reporting the residuals; `tri` failed (n_obs 1) on screws and dark corners before the coordinator's affine-warp fix.
- fit_params on wall heights snaps to interior edges (plate, flyback) at similar height.

## Requests to coordinator
- Consider promoting `outputs/scratch/case/ov.py` (wireframe hypothesis overlay) and `rayp.py` (ray to x/y/z plane) into `scripts/tools/`; they were the fastest way to measure.
- Mask: treat dark, low-saturation pixels with mat-hue reflection on known case faces as object, or ignore pixels within N px of the model case silhouette, so the case fp stops dominating region ranking.
