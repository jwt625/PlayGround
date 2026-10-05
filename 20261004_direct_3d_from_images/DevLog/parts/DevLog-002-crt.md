# DevLog-002-crt: CRT tube assembly (yoke, neck, clamp, socket)

| Field | Value |
|---|---|
| Group | crt |
| Owner | crt modeling agent |
| Files | `scripts/model/crt/build.py`, `config/model/crt.toml`, `scripts/model/crt/texture_yoke.py`, `scripts/model/crt/part_psnr.py`, `scripts/model/crt/part_psnr_by_part.py`, `scripts/model/crt/label_spec.json` (1B only), `assets/textures/crt_*.png` |
| Started | 2026-10-04 |
| Phase | 1A, 1B, 1C, 1D done (2026-10-05) |
| Last run | `outputs/runs/crt_020` (probe, full); ID pass for all train views: `outputs/runs/crt_idtrain` (input of texture_yoke.py) |

## TODO

- [x] Read AGENT_GUIDE and DevLog-001 sections 2, 4
- [x] Measure axis, radii, x stations (sparse points + silhouette rays)
- [x] Block-out: yoke loft, copper end, label patch, holder, dark ring, clamp, cream ring, glass neck, cap, board
- [x] Probe iterations (4) + two local fits (neck stations, yoke/label)
- [x] 1B: yoke label texture (cylinder mode, single view IMG_1560; UV-mapped curved patch)
- [x] 1B: yoke shape knobs (scale_y/z, dy, per-ring s2..s4) fitted; clip tabs; clamp ear + screw (tri2)
- [x] 1B: notched dark ring (40-tooth gear prism), small white ring, electron gun + 8 pins inside an alpha glass neck
- [x] 1B: materials: tape, procedural copper (ring wave), metal, cream, light blue, board; colors from photo patches
- [x] 1B: axis height checked in low views IMG_1587/1590-1594 (overlay + screw tri2 + cap fit with low views)
- [x] 1C: photo textures on the yoke tape (folds), copper side, holder, clamp, white ring, cream ring, cap, neck, dark ring, socket board +X face, label resampled on the exact patch
- [x] 1C: cap flange (dark-blue rim) and board-end step; no flutes are visible in the photos at s2, so none were modeled
- [x] 1C: base colors re-tuned under the world-only rig (part medians in IMG_1560), Specular IOR level 0.25
- [x] 1D: junction gap checked (not CRT geometry, see 1D log); holder extended to x 49.5, yoke shortened; yoke placement refit (crt_yoke3); copper silhouette checked (matches)
- [ ] next: copper ring at the lower sides (tape covers it in IMG_1624; ring visible all around in the model), holder shape (not round: frame blocks), metal parts (clamp 12.3 dB) are view-dependent

## Measurements (world mm)

| Quantity | Value | Method / provenance |
|---|---|---|
| Axis y | 0.0 | yoke silhouette rays in IMG_1560 symmetric (y -24.5/+24.6 big end, -21.0/+19.8 at x 44) |
| Axis z | 23.0 | profile of sparse points: tops yoke 43, clamp 36, cream ring 34.3, cap 32.5, neck 31.3 vs half-widths from rays (13.1, 10.2, 9.0, 9.5); fit crt_neck1 moved 22.5 -> 23.0 |
| Yoke top | z 42.3-43.6 | sparse points |y| < 4 mm, x 11-50 (95th pct per 3 mm bin) |
| Yoke half-width | 24.6 at x 8, 20.4 at x 44 | `pick.py ray IMG_1560 ... --z 21.5` on silhouette |
| Yoke cross-section | elliptical: ry 24.6 > rz 20.1 at the big end | top z 43 with width 49 implies a flattened bell; circle would go below the floor |
| Yoke yellow span | x 9-51 | ray at z 40 on the yellow ends (8.8, 51.2) + point colors |
| Copper end | x 6.5-9.2 | point colors (copper at x 5-8), fit crt_yoke1 (x0 7.5), compromise 6.5 |
| Label | x 22-37, projected y -12.5..9.75 | rays at z 38-42 on label edges in IMG_1560, then fit crt_yoke1 |
| Holder (white) | x 51-56, r 18.95 | rays (half-width 18.2) + fit |
| Dark notched ring | x 56-59.5, r 17.75 | crop reading + fit |
| Metal clamp | x 60-65.5, r 12.8 | rays + point color (gray) + fit crt_neck1 |
| Cream ring | x 65.5-72, r 11 | rays (x 73.1 at z 33) + fit |
| Glass neck | x 48-85, r 9 | rays (half-width 9.5 at x 80) |
| Light blue cap | x 81-91, r 9.5 | point colors (blue at x 86-89), rays; fit wanted x0 80 |
| Socket board | x 91-93, y +-12.6, z 9-34 | rays (y +-12.6), points (brown x 91-92, z max 34-35); bottom z 9 assumed |

## Progress

- 2026-10-04 23:50 Read guide. `pick.py tri` fails on IMG_1560 (top-down; neighbors are rolled, NCC 0.47-0.73 < 0.85). Used sparse-point profiles along X (color-classified) and `pick.py ray` on silhouettes instead.
- 2026-10-05 00:01 crt_001 (probe): all parts present; pts median 0.18-1.15 mm per part; edge mean 2.7-3.8 px. Overlay on IMG_1560 matched within about 1-2 mm. Found the ID decode bug (below).
- 2026-10-05 00:08 crt_002 (6 CRT close-up views 1624/1626/1628/1550/1604/1518): side views consistent within about 2-3 mm.
- 2026-10-05 00:15 fit crt_neck1 (9 params, 100 evals, loss 2.19 -> 1.89): accepted axis z 23, clamp 60-65.5 r 12.8, cream 65.5-72, cap x0 81 (fit 80), cap r 9.5. Lowered yoke rz by 0.5 to keep its top at z 43.
- 2026-10-05 00:22 fit crt_yoke1 (8 params, loss 1.94 -> 1.52): label x 22-37, y -12.5..9.75, holder r 18.95, ring r 17.75, copper x0 6.5 (fit 7.5), holder/ring split at x 56 (fit wanted holder x1 55, leaving a gap).
- 2026-10-05 00:30 crt_004 (probe): final 1A. Board top lowered 35 -> 34 (excess at the board top in IMG_1638).

## 1A metrics (crt_004, probe, geom; run deleted)

| part | edge mean px | pts med mm | note |
|---|---|---|---|
| crt.yoke | 3.35 | 0.57 | fn 1903 px: bell bulge on +Y side (about 2 mm) |
| crt.yoke_copper | 2.22 | 0.46 | |
| crt.label_yoke | 2.95 | 0.15 | plain white |
| crt.yoke_holder | 3.06 | 0.47 | |
| crt.yoke_ring | 2.88 | 0.43 | |
| crt.neck_clamp | 2.36 | 0.43 | |
| crt.neck_ring | 2.78 | 0.37 | |
| crt.neck | 1.83 | 0.88 | |
| crt.socket_cap | 2.26 | 0.77 | fp 0.34 is a mask artifact (light blue cap keyed as mat) |
| crt.socket_board | 2.42 | 0.66 | |

## Colors (linear base colors from 9x9 medians of well-lit patches, IMG_1560 s4)

| surface | photo sRGB | linear base used |
|---|---|---|
| tape | 191 140 15 | 0.52 0.26 0.005 |
| copper (shadowed) | 87 43 22 | ramp 0.08..0.40 (metallic) |
| label | 218 218 202 | photo texture |
| cream ring | 223 220 161 | 0.74 0.72 0.36 |
| light blue cap | 100 208 238 | 0.13 0.63 0.86 |
| board | 166 127 84 | 0.38 0.21 0.09 |
| clamp metal | 136 138 134 | 0.30 (metallic) |

## Phase 1B log

- 2026-10-05 01:00 Label texture: `texture_from_photos.py` cylinder mode works (no tool edit needed). Spec in `scripts/model/crt/label_spec.json`: base (0,0,23), axis -X, ref +Z, R 20.5, angle -36.0..28.1 deg, length -37.3..-22.1 (x 22.1..37.3). A wide pass (angle -45..40, x 19..40, 4 views) located the label borders. The 3-view median ghosts the text (about 1 mm parallax between views because the label surface is not exactly the R 20.5 cylinder), so the final texture uses IMG_1560 alone (cos 0.99). Text orientation in the renders matches the photos.
- 2026-10-05 01:10 crt_005 (full): all 1B parts present; label edge mean 1.89 px.
- 2026-10-05 01:20 crt_006: glass as transmission renders white in EEVEE (no raytracing), so it is now dark glass with alpha 0.3 (BLENDED) and the gun/pins show through. Copper ring enlarged (flares past the tape). Colors from photo patches.
- 2026-10-05 01:35 fit crt_yoke2 (yoke scale_y/z, dy, copper x; 12 views, loss 1.85 -> 1.65): accepted scale_y 0.99, scale_z 0.97 (fit 0.96), dy 0.25. Rejected the copper collapse (fit shrank it to 0.5 mm; the winding texture gives many photo edges). crt_007: yoke pts median 0.57 -> 0.24 mm.
- 2026-10-05 01:50 Axis-height check: screw head tri2 IMG_1560 (1332,1205) + IMG_1590 (1357,722) -> (62.5, 14.4, 25.2), err 1.4/1.4 px (IMG_1591 pick rejected, 10.6 px). Run crt_low (IMG_1587, 1590-1594, geom): the clamp, screw, cream ring, dark ring and holder tab outlines line up with the photo edges within about 1 mm, top and bottom, so axis z 23 holds. The board corner tri2 attempts failed (picks on blurry low views were inconsistent).
- 2026-10-05 02:00 fit crt_cap1 (14 views incl. low ones; cap r/dz/dy/x0, neck r, ring radii; loss stays about 1.34): cap x0 80, cream r 10.9; cap dz -0.19 (not applied).
- 2026-10-05 02:10 fit crt_waist1 (s2..s4, loss 1.702 -> 1.687): s3 0.99, s4 1.04.
- 2026-10-05 02:15 crt_009 (final 1B pass, probe, full): see table below. Photo-vs-render sheet: `outputs/runs/crt_009/eval/crt_photo_vs_render.jpg`.

## 1B metrics (crt_009, probe, full)

| part | edge mean px | pts med mm | color res |
|---|---|---|---|
| crt.yoke | 3.11 | 0.23 | 37.3 |
| crt.label_yoke | 1.55 | 0.13 | 45.5 |
| crt.yoke_copper | 2.74 | 0.49 | 53.9 |
| crt.yoke_holder | 3.07 | 0.47 | 62.1 |
| crt.yoke_ring | 3.48 | 0.50 | 44.1 |
| crt.yoke_tab_0 / _1 | 3.18 / 4.03 | 0.41 / 0.35 | |
| crt.neck_clamp | 2.15 | 0.35 | 49.7 |
| crt.clamp_ear / screw | 2.00 / 1.29 | 0.47 / 0.41 | |
| crt.neck_white_ring | 2.56 | 0.33 | |
| crt.neck_ring | 2.98 | 0.40 | 47.2 |
| crt.neck | 1.70 | 0.74 | 52.6 |
| crt.socket_cap | 1.96 | 0.67 | 59.7 (fp 0.28, mask) |
| crt.socket_board | 2.39 | 0.64 | 40.2 |

Raw PSNR inside crt.* pixels (probe views, eroded 2 px): 10.6 dB, limited by render brightness (next section).

## Shared tool edits

None. The cylinder mode of `texture_from_photos.py` was correct as is (1B). In 1C I wrote `scripts/model/crt/texture_yoke.py` in the crt folder instead of extending the shared tool (model-exact surfaces and ID visibility).

## Phase 1C log

- 2026-10-05 02:40 crt_010 (baseline under the world-only rig). PSNR inside crt.* pixels (probe views, ID-owned, eroded 2 px): raw 11.59, fit 12.01 (compare_photometric convention: affine fit over object + mat), fit_local 13.10 (affine fit on the crt pixels themselves). 3DGS on the same pixels: 22.48 / 22.62 / 22.90.
- 2026-10-05 02:50 `scripts/model/crt/texture_yoke.py` (own tool; the shared one is untouched): texel grid built from crt.toml exactly like the mesh (elliptical, tapered yoke loft; cylinders; label patch; board +X plane), so there is no cylinder-approximation parallax. Visibility per texel from an ID pass of all 123 train views (`outputs/runs/crt_idtrain`, Workbench, 6 s): a view contributes only where its ID pixel belongs to the target object (eroded). Score = cos x px/mm, cos >= min_cos. Per-view exposure gain toward a 5-view per-texel median reference (removes most seams). Color gates (HSV) on the yoke (tape yellow, S >= 120) and the copper: reject unmodeled occluders and highlights. Holes (never seen, mostly the bottom) get the median color with a short inpaint at the border. Train views only.
- 2026-10-05 03:00 crt_011: yoke + copper textured, colors re-tuned: raw 13.38.
- 2026-10-05 03:10 crt_012: round parts textured: raw 14.08, fit_local 14.76.
- 2026-10-05 03:15 crt_013: k = 3 (median of the 3 best views per texel) beats k = 1: raw 14.28. Kept.
- 2026-10-05 03:25 crt_014: dark gear ring UV-mapped and textured; socket board +X face (`crt.socket_board_face`) textured: raw 14.41.
- 2026-10-05 03:30 Diagnostic: yoke textured from IMG_1624 alone and rendered at IMG_1624 gives only 16.2 dB on the yoke. The body matches; the error sits in texels that view excludes (grazing angle, eroded borders, gate), so the remaining yoke error is mostly at borders and cross-view misregistration, not the render pipeline.
- 2026-10-05 03:35 crt_015: label resampled on the exact label patch from IMG_1560 (`crt_label_1560.png`): label 18.80 -> 19.62 dB. A per-texel best-view label was blurrier (out-of-focus views win on px/mm).
- 2026-10-05 03:40 fit crt_copper1 (copper ry/rz) shrank the ring inside the yoke (edge loss rewards hiding it); rejected, since the photos show the copper flaring past the tape.
- 2026-10-05 03:50 crt_016 (final): textures with erode 1, min_cos 0.15: raw 14.47, fit 13.92, fit_local 15.11. Sheet: `outputs/runs/crt_016/eval/crt_photo_vs_render.jpg`. The wires group fell back to its last-known-good copy in this run.

## 1C metrics (crt_016, probe, full; PSNR dB inside crt.* pixels)

| metric | crt_010 (before) | crt_016 (after) | 3DGS same pixels |
|---|---|---|---|
| raw | 11.59 | 14.47 | 22.45 |
| fit (affine over object + mat) | 12.01 | 13.92 | 22.60 |
| fit_local (affine on crt pixels) | 13.10 | 15.11 | 22.87 |

Per part (raw): yoke 15.9 (52 percent of pixels, 41 percent of the error), label 19.6, cream ring 15.1, board face 15.4, holder 14.5, dark ring 13.4, neck 12.8, copper 12.5, clamp 12.3, cap 12.2. Geometry: edge mean 3.01 px overall, pts median 0.46 mm.

## Phase 1D log

- 2026-10-05 04:10 Junction orphans (about (-14, +-3, 14)): the sparse points there form a flat strip at z 12.4-13.9, x -13 .. -33, y -3.3 .. +4.9 (tri on IMG_1550, train neighbors: (-23.2, 0.9, 12.6) 7 inliers, (-32.7, -0.6, 12.9) 9, (-32.0, 2.2, 12.9) 10, (-33.0, -3.3, 12.7) 9; near end 2-view only: (-13.5, 0.0, 12.4), (-14.1, 4.9, 12.3)). In IMG_1550 / IMG_1560 / IMG_1640 it is a dark, translucent strip (about 7 mm wide) that lies on the window floor. It runs from under the bracket to the +X edge of the white card (card +X edge x -32.25, z 14.3). It is not tube geometry, and the CRT funnel/face is not visible in any view. Not modeled in crt; passed to the coordinator for the case group (window/card).
- 2026-10-05 04:15 Holdout audit: all fits (crt_neck1, yoke1-3, waist1, cap1, copper1) and the texture runs used train views only. In 1A, two `pick.py tri` measurements anchored in IMG_1560 matched IMG_1535 (holdout): the cream ring and clamp tops used for the first axis-z estimate. The axis was later set by fits on train views. The holder tab centers were re-measured without holdout (pick.py now refuses it): unchanged, (48.0, -19.0, 32.7) and (49.1, 17.5, 29.9).
- 2026-10-05 04:20 crt_017 baseline (new evaluator): raw 14.69, fit 13.93, fit_local 15.32.
- 2026-10-05 04:30 Holder block: in IMG_1560 the white holder is visible from x 49 on the +Y side (tri (49.1, 16.2, 29.6), (51.3, 7.1, 39.5), (55.1, 16.5, 27.4); r 17-18.3). Changes: last yoke ring x 51 -> 49.5, holder x0 51 -> 49.5. The ID pass was re-rendered and the textures rebuilt. crt_018: raw 14.84, yoke 16.5 dB, holder 15.0 dB. Fixed texture_yoke.py: the board face is now owned by `crt.socket_board_face`, and surfaces that no view sees are skipped.
- 2026-10-05 04:40 Added a yoke dz knob; fit crt_yoke3 (5 params, 15 train views, loss 1.678 -> 1.656). Result: scale_y 0.96, scale_z 0.965, dy -0.25, dz 0.06, s4 1.055. Re-rendered the ID pass and retextured. crt_019: raw 14.96, yoke 16.96 dB, yoke edge mean 3.36 -> 3.23 px.
- 2026-10-05 04:45 Copper silhouette overlays (IMG_1560, IMG_1624) match the photo edge. The copper texture gate now also accepts tape colors (the tape covers the ring at the sides): crt_020 raw 14.97 (copper unchanged at 12.3 dB).

## 1D metrics (crt_020, probe, full; PSNR dB inside crt.* pixels)

| metric | crt_017 (1D start) | crt_020 | 3DGS same pixels |
|---|---|---|---|
| raw | 14.69 | 14.97 | 22.42 |
| fit (object + mat) | 13.93 | 14.21 | 22.59 |
| fit_local | 15.32 | 15.59 | 22.87 |

Geometry (whole model, new evaluator): edge mean 3.05 px, pts median 0.39 mm. CRT parts: yoke pts median 0.22 mm and edge mean 3.23 px; label 0.11 mm and 1.65 px.

## Requests to coordinator

1. ID pass decode is exact-match on 8-bit RGB; one palette color came back off by one (crt.yoke_copper key (208,127,201) rendered (208,127,200) in crt_001), so the whole part counted as "missing geometry" (large cyan blobs, fn). Palette index depends on object order, so any group can hit it when objects are added. Suggest nearest-key decode with a tolerance of 2 in `evaluate.py` (`decode_id`); `fit_params.py` does not use that decoder.
2. Mask: the light blue socket cap is keyed as mat, so `crt.socket_cap` always shows fp about 0.3-0.4 and fits push it smaller (use `--w-fp` low for it). A hue + saturation rule for the mat (the cap is lighter and less saturated) or a manual mask patch would fix it.
3. `pick.py tri` fails on top-down views (rolled neighbors); a rotation-compensated patch (align by relative camera roll) would make it usable there.
4. Shell note for other agents: the shell is zsh, so `for p in "u v"; do pick.py ... $p` does not split; use `${=p}`.
5. (2026-10-05, 1B) RGB rig brightness: under the white world (strength 1) plus the 3 W key light, top-facing surfaces render about 1.4-1.9x their base color in linear, not 1x. Measured on crt_006/crt_009: photo-textured mat renders (78,158,221) vs photo (1,112,194) in IMG_1560 and (79,177,255) vs (30,116,177) in IMG_1527. The cap, base 0.127 linear red, renders 134 sRGB (0.24 linear, 1.87x). The photo label texture clips at 255 (photo 220). The key light alone adds about 0.8x albedo to up-facing surfaces. Suggest dropping the key light (or setting world strength about 0.55) so that photo textures reproduce photo values. The crt base colors are left at the photo-patch values, so they will be right once the rig is fixed.
6. (2026-10-05, 1B) `pick.py tri2` crashes with a LinAlgError (empty matrix) when no observation is an inlier, instead of reporting the inconsistency.
7. (2026-10-05, 1D) Junction orphans near (-14, +-3, 14) are a dark translucent strip lying on the window floor (x -13 .. -33, z 12.4-13.9, about 7 mm wide, centered on y 0; tri points in the 1D log). This belongs to the case group (window/card), not to the CRT.
