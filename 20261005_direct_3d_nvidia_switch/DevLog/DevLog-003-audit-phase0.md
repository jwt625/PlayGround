# DevLog-003: Fresh-context audit of the Phase 0 measurement chain

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | Done (read-only audit; nothing changed except this DevLog and `outputs/scratch/audit/`) |
| Scope | Metric scale, world-frame axes and origin, COLMAP model, gravity, holdout hygiene, masks and evaluator |
| Author | Claude (auditor subagent), at Wentao Jiang's request |
| Blender runs used | 0 |

Method: own scripts in `outputs/scratch/audit/a01..a19_*.py` (kept for reproduction; crops deleted). They use only
`tools/geom.py` camera loading, the COLMAP binaries, the photos, masks and configs. Measurements use train views
only (my crop helper asserts this). Pixel coordinates are 1/2 scale. World = current tray frame, mm.
Labels: FACT = measured or read from files here; EST = estimate with reasoning; SPEC = speculation.

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | Scale (438 mm width over the wall faces) | Holds. 13 independent hole triangulations give a 437.7-437.8 mm width at the current scale (-0.05 %). The independent length check passes once the front face is located correctly (780 vs 776 mm, +0.5 %). Estimate: correct to about +-0.5 %. |
| 2 | Y origin | Wrong reference. Y = 0 is the printed text on the acrylic sign. The front panel (port plate) is at Y = +23..24, the rear lip at Y = 802-803. The DevLog-001 length cross-check passed by accident. |
| 3 | Z axis | Rolled about 0.75-1.0 deg about Y relative to the chassis. +X features sit 6-8 mm higher than their -X mirrors. Yaw (X axis) is good (< 0.06 deg). Pitch is good (<= 0.3 deg). |
| 4 | COLMAP model | Good. 43 views at 1.0-1.6 px mean. IMG_5741's pose is consistent at mm level near the rear. Excluding IMG_5739 is right. f = 4430 px is plausible. |
| 5 | Gravity | Reproduced (0.25 deg). The bootstrap uncertainty is 2-5 deg. The code exists only as scratch output. |
| 6 | Holdout hygiene | No direct misuse found. 9.8 % of tray points exist only because of holdout views. Two tools have no guard. |
| 7 | Masks / evaluator | The hull-unknown rule hides model excess in 3-15 % of the hull area. The hull box inherits the Y and Z errors. Sign points count as tray points. |

## 1. Metric scale (most important)

### 1.1 Width: re-derived independently (FACT)
- The coordinator's 4 holes reproduce exactly (`a02_holes.py`): X -219.12, -218.88 / +218.55, +219.45.
  - The -X `tri` picks used views 5756, 5755, 5730 (and 5757). Holdout IMG_5728 was in the candidate pool but was not an inlier.
- +X manual correspondences are correct. Crops of IMG_5745 and 5744 show the same pattern in both (a large hole with a small slot beside it, plus a second hole). Sensitivity is 0.12-0.17 mm in X per 1 px pick error, so even the 3.8 px residuals cost less than 1 mm.
- My own holes are 7 new features at different Y (train views only, 2-3 views each, residual 0.5-2.4 px):

| Wall | Views | X (mm) | Y | Z |
|---|---|---|---|---|
| +X | 5745, 5746 | 218.94 | 236.0 | -42.0 |
| +X | 5745, 5746, 5744 | 218.46 | 283.9 | -28.9 |
| +X | 5745, 5744, 5743 | 218.53 | 388.7 | -28.4 |
| +X | 5745, 5746 | 218.41 | 186.2 | -30.6 |
| -X | 5756, 5755, 5757 | -218.91 | 389.1 | -36.0 |
| -X | 5756, 5757, 5758 | -219.21 | 284.3 | -36.6 |
| -X | 5756, 5755, 5729 | -218.92 | 488.9 | -36.2 |
| -X | 5756, 5755 | -219.34 | 588.3 | -54.7 |

- All 12 holes: +X face at 218.72 (sd 0.40, n 6), -X face at -219.06 (sd 0.18, n 6). Width is 437.78 mm, standard error about 0.2 mm.
- My holes only: 437.7 mm.
- Mirrored hole pairs at Y 389 and 284 give 437.44 and 437.67 mm.
- Conclusion: the current scale reproduces the 438 mm width to -0.05 %. The +X mean of 219.0 in `world.yaml` is pulled up by one hole (219.45).

### 1.2 Length: the cross-check in DevLog-001 is invalid; redone (FACT)
- Front: Y = 0 is not the panel. The Y = 0 mode bin (Y -3..3) holds 111 points. 80 of them sit at X -60..-20, Z -32, on the printed "Spectrum-X CPO / Switch Tray" text of the acrylic sign.
  - Projected into IMG_5740 they land on the letters (`a11_frontproj.py`, `a12_panel.py`).
  - Points on the RJ45 openings, LEDs and MMC adapter faces are at Y 22-26 (peak 23-24).
- Rear: on the +X wall in IMG_5741, the seam where the gray wall meets the black rear block is at Y 800.6-802.8, spanning Z -57.0 to +29.6. Ray-plane intersections at X = 218.6.
  - An independent S3 view (IMG_5755) overlays the rear-wall top edge on the line Y = 802 at Z 20. The line Y = 776 falls inside the tray.
  - The chassis agent independently found 800-803 (three views) and fitted 805.
- Body length from the port plate to the rear lip is 802.5 - 23.5 = 779 +- 2.5 mm, against 776 published (+0.4 %, range +0.1..+0.8 %).
- Measured length/width ratio 1.779 vs published 1.772 (+0.4 %).
- DevLog-001's checks fail on this evidence:
  - "A 438 x 776 box matches the rear wall and front panel in 4 views": the box is off by about 24 mm at both ends.
  - "Points end at Y = 763 (99.5th percentile)": a percentile of a cloud that thins out at the rear is not an end position.

### 1.3 Other dimensions
- Front panel height 87 mm: not measurable as such (the panel's bottom edge is on the plinth). This agrees with the chassis agent.
  - -X side-wall outer face, from the top fold to the bottom edge: 83.3 mm (IMG_5756) and 80-83 mm (IMG_5757), by ray-plane at X = -219. The chassis agent gives 83.5-84.
  - +X: 87.9 mm (IMG_5745). This is unreliable: the bright strip I took as the top is likely the top surface of the inward fold seen from above, which biases the ray-plane Z upward.
  - EST: 83.5-84 mm of wall plus a missing cover is compatible with an 87 mm 2RU envelope. This is not a scale check.
- Package: I overlaid a 108.25 x 107.75 mm outline (the package agent's value) and a 110 x 110 mm square at ring-top height on IMG_5740 (train). The 108 mm outline follows the outer substrate/ring edges; the 110 mm square sits about 5-8 px (s2) outside them on each side. So 108 at the current scale is confirmed.
  - Scaling to 110 would need +1.7 %. The width would then be 445 mm and the length 793 mm. Both contradict the A-rank sources.
  - EST: the SemiAnalysis 110 mm is rounded or pre-production. It is weak evidence and cannot override width and length.
  - SPEC: a single shared focal across focus distances of 0.25-0.6 m (focus breathing 1.0-2.4 %) could bias close-up-dominated regions by up to about 1 %. Not quantified.

### 1.4 Scale verdict
- FACT: the published body width (438) and length (776) both reproduce from the current scale: width -0.05 %, length +0.4 %. The length uses the port plate as the front face.
- EST: the scale factor is 1.000, range 0.995-1.002 (about +-0.5 %). The range covers the width/length disagreement and the unknown spec tolerance.
- The 438 mm identification holds. The 110 mm package figure is not supported.

## 2. Frame axes

| Axis | Evidence (FACT) | Result |
|---|---|---|
| X (yaw about Z) | Mirrored hole pairs: Y differs by 0.35 and 0.42 mm across 437.6 mm. -X holes from Y 284 to 588: X within -218.91..-219.34, no trend. | < 0.06 deg; drift < 1 mm at Y = 0 / 800. The 1.8 deg between the two crossbar views averaged out well. |
| Z roll (about Y) | Mirrored holes: +X is higher by 7.6, 7.7 and 6.7 mm. Wall bottom edges: -X -66.4/-65.8/-65.7, +X -60.3/-59.8 (difference 5.7-6.3). Crossbar top dz/dx: 1.0 deg (mine, \|Z\| < 6), 0.73 deg (chassis agent, \|Z\| < 4). Front-bay layer at Z of about 0: +0.92 deg (MAD 0.44 mm, 1718 points). | Rolled 0.75-1.0 deg. That is +-2.9..3.8 mm in Z at the walls. |
| Z pitch (about X) | -X bottom edge: -66.4 / -65.8 / -65.7 at Y 283 / 395 / 501. +X bottom: -60.3 (Y 237) to -57.0 (Y 802). | <= 0.3 deg; at most about 3 mm over the length. |

- The RANSAC plane (4,195 inliers) is the board layer at Z of about -36. That layer is flat in this frame per bay (roll +0.11 / -0.08 deg). The chassis features, by contrast, all show the roll.
- EST: either the board layer is genuinely 0.9 deg off the sheet metal (7 mm across the width; implausible for a real tray), or the SfM is slightly bent across X. Mirrored sheet-metal features are the better reference for the chassis.
- The chassis group already compensates locally (`body.roll_deg = -0.73`). Upper and lower do not. Groups can therefore disagree by up to about 4 mm in Z near the walls.

## 3. COLMAP model

- Per view (FACT, `a01_colmap_stats.py`; full resolution): the 43 used views have 533-3,694 observations, mean error 1.02-1.61 px and p90 2.1-3.0 px. Grid coverage is 0.56-1.00 except IMG_5741 (0.42).
  - Median track length is 2-3 (mean 2.82). This is thin. Many points are 2-view and unverified by a third view.
- IMG_5739 (excluded):
  - Its 275 points are all far background (Y 1.2-1.5 m, Z -0.85 m), co-observed with 5736, 5737 and 5740.
  - Its median depth is 1.77 m, against 0.4-0.6 m for the other views.
  - The pose rests on low-parallax background only, so the exclusion is right.
  - Its 68 two-view points stay in `points_world.npy`. Only 1 is inside the tray box, so they are harmless for the tray. Keep them out of any environment/backdrop fitting.
- IMG_5741 (98 observations, 1.56 px, co-observed mostly with 5742 / 5743 / 5744):
  - The rear seam measured in 5741 (Y 801-803) agrees with the S3 overlay (5755) and with the chassis agent's three-view value.
  - Foreign sparse points project onto the braided-cable edge in 5741.
  - Verdict: the pose is good near the rear end. Keep it out of holdout and probe, as now.
- Intrinsics: one SIMPLE_RADIAL camera, f = 4430.3 px, k1 = 0.0058, principal point fixed at the center. EXIF is identical for all 44 photos (5.96 mm, 26 mm equivalent, no digital zoom).
  - 26 mm equivalent by diagonal gives 4290 px; COLMAP's sensor-width prior was 4243 px. 4430 is +3.3..4.4 % above.
  - EST: plausible. "26 mm" is rounded (+-2 % for +-0.5 mm), and focusing at 0.25-0.6 m adds 1.0-2.4 % magnification.
  - Residual risk: one shared focal across those focus distances (about 1.4 % spread). See SPEC 1.3.

## 4. Gravity (`config/frames.json` "scene")

- Method: the smallest eigenvector of the display-right camera vectors, assuming a small hand roll.
  - My re-derivation handles EXIF 6 and EXIF 3 explicitly (`a15_gravity.py`) and reproduces up = (-0.067, 0.916, 0.396), 0.25 deg from the config.
  - Eigenvalues 0.69 / 10.4 / 31.9. Roll rms 7.3 deg, maximum 19.4 deg (IMG_5714).
- Uncertainty (FACT):
  - Bootstrap angle: median 2.3 deg, 95th percentile 5.5 deg.
  - Per session: S1 6.9 deg off, S3 15 deg off, S2 degenerate (eigenvalue 0.01 vs 0.12).
  - The "about 3 deg" poster/plinth check is consistent.
- Plausibility: tray long axis 23.7 deg from vertical, rear up. This matches the photos qualitatively.
  - EST: photographers tend to align the frame with a tilted object, so the zero-mean-roll assumption may be biased by a few degrees.
  - This only matters for the environment and exports.
- Reproducibility: no script computes it. Only `outputs/scratch/coord/up_est.npy` / `up_world.npy` and the JSON exist.

## 5. Holdout hygiene

- Split (`split.py`): farthest-point sampling with session and package quotas. `min_obs` 300 keeps 5739 and 5741 out. It refuses to overwrite.
  - `split_before_exclude.json` in coordinator scratch shows that holdout and probe did not change after the exclusion. No problem found.
- Guards:
  - `pick.py`, `texture_from_photos.py` ("auto" = train only), `bake_id_owned.py` and `fit_params.py` refuse holdout views.
  - The lower group's own tools assert it.
  - `rayplane.py` and `overlay.py` do not refuse holdout views.
  - `world_frame.py` matches against all views (holdout IMG_5728 was in the candidate pool; not an inlier in the current result).
  - No config or part DevLog uses a holdout view for measuring (grep over `config/`, `DevLog/parts/`, `scripts/model/`).
- Structural leakage through the sparse cloud (FACT, `a16_leak.py`): SfM and bundle adjustment ran on all 44 photos.
  - Of 19,286 tray-region points, 4,464 (23 %) include a holdout observation, and 1,888 (9.8 %) have fewer than 2 train observations: they exist only because of holdout views.
  - Package region: 56 / 664 and 42 / 664. Package top plane (used for the package frame): 12 / 549.
  - These points feed measuring (`points_world.npy`, `pick.py views`, cluster-based footprints) and the pts and orphan metrics on holdout runs.
- Masks on holdout views are SAM output with 3D-box prompts. No per-view hand polygons exist for holdout views. Fine as ground truth.

## 6. Masks and evaluator

1. The hull-unknown rule biases silhouette scores toward oversize.
   - Inside the projected tray and package hulls, SAM background becomes 128. That background is 3-15 % of the hull area (`a17_masks.py`): probe 3.9-13.9 %, holdout 2.6-15.0 % (IMG_5747 15.0 %, 5737 13.8 %).
   - Model excess there costs nothing in IoU or fp. Missing geometry is still counted. Points are one-sided (data to model), so excess inside the outline is checked only by edges and color.
   - `unscored_frac` (model px on unknown) is reported, which helps, but it does not separate excess from correctly modeled interior.
2. The hull box is stale. It is `[-219, 219, 0, 776, -75, 12]`, but the body spans Y -20.5 (grip) or 23.5 (port plate) to 803, and Z about -66 to +21 (rim).
   - Y 0-23 in front of the panel (sign, plinth gap) is unknown, so front-panel overshoot is not penalized.
   - The rear lip (Y 776-803) and the rim above Z 12 fall outside the hull. Wherever SAM drops dark parts there, correct geometry scores as fp.
3. Sign points count as tray points. About 170 sparse points on the acrylic sign's printed text (Y -10..+3) are inside the tray point region (Y > -30). They become orphans, or they are attributed to front parts (50 mm search radius) and inflate those parts' medians. The chassis agent saw this at Y -33 (sign base).
4. Per-part points are untruncated within the 50 mm nearest-surface search (the CRT problem is fixed). Points farther than 50 mm are dropped from the per-part statistics and counted at 50 mm in the global one: a mild truncation.
   - Still as in the CRT audit: the edge chamfer is one-sided and capped at 10 px, and the IoU has a 2 px band (the no-band IoU is now also reported).
5. Mask leaks seen on `outputs/masks/sam_tray.jpg`: the tray mask spills onto the plinth, bag and table below the tray in IMG_5752-5757 (train). Only 5751, 5742 and 5746 have unknown polygons. Close-up masks 5714 and 5733 are partial, as already noted.

## 7. Findings ranked by severity, with fixes

1. HIGH: the Y origin is the acrylic sign, not the front panel (port plate at Y +23..24; rear lip at Y 802-803).
   - Wrong in AGENT_GUIDE ("Y = 0 front panel face", "body ends near Y = 776", bay ranges), README/DevLog-001 (length cross-check), `mask_prompts.yaml` hull box, and possibly the lower/upper TOMLs that assume front bay "Y 0-330".
   - Fix, two options. Either keep the numbers mid-Phase-1A and correct the docs now ("port plate Y = 23.5, rear lip Y = 802.5"). Or shift `world.yaml` t by -23.5 mm in Y at a milestone, with one scripted Y shift of every TOML.
   - Either way, replace the DevLog-001 length check with port plate to rear lip = 779 +- 2.5 mm vs 776.
2. MEDIUM-HIGH: Z is rolled 0.75-1.0 deg about Y relative to the chassis, and only the chassis group compensates.
   - Fix: re-level Z from mirrored chassis features: the 3 hole pairs (dZ 7.3 +- 0.5 mm, 0.95 deg) plus the crossbar top. Rerun `world_frame.py`, or add a "chassis" roll to `frames.json` that all groups use.
   - Then set the chassis `roll_deg` to 0.
3. MEDIUM: holdout leakage through the sparse cloud (9.8 % of tray points need holdout views) and unguarded tools.
   - Fix: export a train-only point set for measuring (points with >= 2 train observations, re-triangulated from train observations) and keep the full cloud for evaluation only.
   - Add the holdout refusal to `rayplane.py` and `overlay.py`. Restrict `world_frame.py` to train views.
4. MEDIUM: evaluator and mask bias.
   - Update the hull box to the measured body, e.g. tray `[-219, 219, 23, 803, -67, 22]` in the re-leveled frame. Add a per-part diagnostic "model px over SAM background inside the hull" (not in IoU) so oversize is visible.
   - Exclude the sign box from the tray point region (the printed text is at Y -10..+3; the front face was not measured; the prompt box Y -60..-5 does not reach the text).
   - Add unknown polygons, or a plinth/below-tray box, for 5752-5757.
5. LOW: gravity has no script; uncertainty 2-5 deg. Put the eigen method in `scripts/prep/` with the bootstrap, and store the uncertainty in the `frames.json` note.
6. LOW: package 110 mm (analyst) vs 108.0-108.3 measured. Keep the measured value. Record in `dimensions_research.md` that 110 is contradicted by the width/length scale at -1.7 %.
7. LOW, housekeeping:
   - The `picks.yaml` comment says "views agree within 0.6 deg"; `world.yaml` says 1.775.
   - `world_frame.py` hard-codes 0.12 m per unit in the crossbar Z filter (actual 0.095; the 15 mm window is really 11.9 mm).
   - The `geom.py` docstring still describes the CRT frame ("Z up, origin at the case footprint center").
   - `world.yaml` checks (`wall_x_mm` +-219) are true by construction and should not be read as validation.

## 8. Progress log

- 2026-10-05 23:40: Read README, AGENT_GUIDE, DevLog-000/001, dimensions research, the CRT audit, configs and scripts.
- 2026-10-05 23:45: COLMAP per-view stats; reproduced the 4 wall holes. Wall-point histograms, a contact sheet and frame-line overlays.
- 2026-10-05 23:50: 7 independent wall holes (tri2, train views); mirrored pairs gave the width and the roll. Wall edges by ray-plane.
- 2026-10-05 23:52: Front-layer points traced to the acrylic sign; panel features at Y 23-24; rear seam at Y 801-803 (5741), confirmed in 5755.
- 2026-10-05 23:55: IMG_5739/5741 checks, EXIF, gravity re-derivation with bootstrap, holdout leakage counts, mask-hull fractions, package 108 vs 110 overlay.
- 2026-10-05 23:58: Compared with the chassis agent's independent findings (roll 0.73 deg, Y origin, rear 803): consistent. Wrote this DevLog.
