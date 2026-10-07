# DevLog-001: Phase 0 - poses, frames, scale, masks, toolkit

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | In progress |
| Scope | Everything before part agents start: own SfM, world and part frames, metric scale, split, masks, adapted CRT toolkit |
| Authors | Claude (coordinator), at Wentao Jiang's request |

Plan and answers: DevLog-000 (Sections 4 and 6.1).

## 1. TODO

- [x] Sources copied to `source/images/` (44 files, md5 identical to the download, write-protected, gitignored)
- [x] Toolkit copied from the CRT run (prep, blender, eval, tools, model lib/build_all, env); loaders read
      `config/scene.yaml` paths relative to the project (`source_dir`, `sparse_model` -> `cfg["sparse_dir"]`)
- [x] uv env (Python 3.12, CRT deps) plus dependency group `seg` (torch 2.13.0 / torchvision 0.28.0 from the uv
      cache, hydra-core, iopath); SAM 2.1 used in place from `~/Documents/GitHub/sam2` (source via sys.path,
      checkpoints read-only; nothing built or written there)
- [x] COLMAP SfM (`scripts/prep/run_colmap.sh`, Homebrew COLMAP 3.11.1, CPU, 8 of 12 threads, one shared
      SIMPLE_RADIAL camera, exhaustive + guided matching, incremental mapper); check registration per session
- [x] Scale references (research subagent -> `references/md/dimensions_research.md`); measured width, checked length
- [x] World frame = tray frame (`scripts/prep/world_frame.py`, inputs `config/picks.yaml`)
- [x] Gravity / scene frame and package frame (`config/frames.json`; package frame provisional)
- [x] Undistort (1/4, 1/2), split (`scripts/prep/split.py`: 5 holdout spanning sessions, 8 probe)
- [x] Blender cameras and 3-level masks (`scripts/prep/cameras_masks.py`)
- [x] Masks with SAM 2.1 (tray, package, sign occluder); checked on contact sheets (`outputs/masks/`)
- [x] Evaluator constants (object point regions) moved to `config/eval.yaml`
- [x] Smoke test: stub chassis box + package slab, probe iteration about 10 s
- [x] AGENT_GUIDE.md; launch part agents (chassis, upper, lower, package)
- [x] Environment group (plinth, table, ground, backdrop) - basic, `config/model/env.toml`
- [x] final_eval.sh / export names / photometric compare without 3DGS (flat-color floor), orbit about gravity
- [ ] Remaining mask issues: IMG_5714 and 5733 tray masks partial; IMG_5741 pose to verify (audit)
- [x] Roll correction as a shared "tray" frame in frames.json (0.85 deg); world frame and Y origin stay

## 2. Progress log

- 2026-10-05 22:47: COLMAP started in tmux session `nvsw_colmap`; log `outputs/logs/colmap_<timestamp>.log`.
  Feature extraction (CPU SIFT, max image size 3200 px, 11-15k features per image) took 2 min 22 s.
  EXIF focal prior 4243 px (26 mm equivalent).
- 2026-10-05 22:53: COLMAP done in 5.7 min: 44/44 registered in one model, 28,775 points, 81,227 observations,
  mean reprojection error 1.31 px (per view median 1.2-1.5 px). One SIMPLE_RADIAL camera f 4430.3 px,
  k1 0.0058. Sessions are tied: 12,183 points tracked across sessions (S1+S3 8,716; S1+S2 2,917; all three 449),
  so the scene is treated as static. Weakest views: IMG_5741 (98 observations), IMG_5739 (275).
- 2026-10-05: Split (`scripts/prep/split.py`, quotas in `config/scene.yaml`; views with < 300 observations are
  never holdout or probe): holdout IMG_5715, 5728, 5737 (package), 5747 (S2, package), 5750 (S3); probe 5729,
  5734, 5738 (pkg), 5740 (pkg), 5742, 5746, 5749, 5751. Fixed before any measuring.
- 2026-10-05: Masks: SAM 2.1 base_plus on MPS (about 2 s per view), box + point prompts projected from 3D boxes
  (`scripts/prep/masks_sam.py`, `config/mask_prompts.yaml`). Tray masks good in most views; partial in the
  close-ups IMG_5714, 5724, 5725, 5712 (per-view prompts needed).
- 2026-10-05: Product identified by the research subagent: NVIDIA SN6810-LD (2RU, single Spectrum-6, 128 MMC
  ports, MGX v1.2): body 438 x 776 x 87 mm (NVIDIA user manual v1.3, PNY and Dell datasheets; 900 mm depth
  including UQD fittings and front levers). Package substrate 110 x 110 mm is analyst-only (SemiAnalysis),
  check only. See `references/md/dimensions_research.md`.
- 2026-10-05: Frame and scale (`scripts/prep/world_frame.py`). What failed first: one-pick epipolar matching
  on smooth gray metal and on the front panel (few views see it); manual corner picks on dark silhouettes;
  edge-distance box fits (snap to internal edges); silhouette fits with a coarse box (cannot separate levers,
  fittings and cables). What worked: tray Z from 4,195 RANSAC inliers on the parallel layers; X from the
  sparse points on the NVIDIA crossbar top in two views (directions differ by 1.8 deg; averaged; refine later);
  width from screw holes on both outer side walls (-X: 2 holes by epipolar tri, 0.3-1.3 px; +X: 2 holes by
  manual matches, 2.3-3.8 px; per-side spread 0.2 and 0.9 mm). Scale 94.97 mm per COLMAP unit from the 438 mm
  body width. Cross-checks: a 438 x 776 mm box matches the rear wall, front panel and both side walls in 4
  views; sparse points end at Y = 763 mm (99.5th percentile) against the 776 mm body length. RJ45/USB openings
  were too ambiguous to use (oblique, bezel outlines; USB type unclear).
  Origin: X midway between the walls, Y at the front panel face, Z at the crossbar top face.
- 2026-10-05: Gravity (scene frame): smallest eigenvector of the cameras' display-right vectors (assumes small
  roll; roll rms 7.3 deg, eigenvalues 0.70 / 10.5 / 32.8): up = (-0.071, 0.915, 0.398) in the tray frame, so
  the tray's long axis is 23.8 deg from vertical (rear end up) and its open top faces 66.6 deg from up.
  Checked against poster and plinth verticals in 4 views (about 3 deg). A first estimate from averaged
  display-up vectors was biased by camera pitch; table and plinth tops have too few points for a plane fit.
- 2026-10-05: IMG_5739 excluded (wrong COLMAP pose: tray box overlay misses entirely; 275 observations);
  holdout and probe unchanged, train 38. Loaders skip excluded views.
- 2026-10-05: Package frame (provisional): 408 sparse points within 1 mm of a plane in the close-ups, normal
  (-0.061, -0.115, 0.992) in the tray frame (package top nearly parallel to the tray layers), center
  (107.9, -153.5, 77.9) mm, point extent 98.6 x 102.8 mm. Package masks good in 7 views (IMG_5711, 5712, 5717,
  5736, 5737, 5740, 5747).
- 2026-10-05: Masks assembled (`scripts/prep/cameras_masks.py`): object = tray or package; unknown = enclosed
  holes, background inside the projected tray-body and package hulls (SAM drops the dark braided cable and
  black parts inside the tray, which made the model look like excess geometry), the acrylic sign (SAM object
  `sign`, only in the 8 views where it is visible; per-view box for IMG_5738) and a 2 px boundary band.
  Smoke test (stub 438 x 776 x 87 mm box + 110 mm package slab, 8 probe views): IoU 0.929, package slab sparse
  point median 0.43 mm; remaining heat is real missing geometry (levers, fittings, package ring).
- 2026-10-05: Group split (also 3D regions): chassis (shell, front panel, levers, crossbar, dividers, black
  ducts), upper (middle bay Y 370-595 and rear bay Y 605+, including rear fittings and tubes down to Y 350),
  lower (front bay Y 0-330, tubes up to Y 350, braided cable full length), package (package and stand, in the
  package frame). AGENT_GUIDE.md written for this capture.
- 2026-10-05: Shared tools adapted while the agents run: compare_photometric reports a flat mean-color floor
  ("flat") instead of the 3DGS baseline (used only if renders exist); export_glb names files from
  `config/scene.yaml` model_name and rotates into the gravity frame by default (--tray-frame to skip);
  render_orbit orbits about gravity-up around the tray center (1.6 m) plus close-ups of the package, front bay,
  rear bay and front panel; pick.py default depth range 200-1500 mm. Tested on the stub model.
- 2026-10-05: Environment planned from agent measurements: table top height from the package stand base,
  plinth top from the front panel's bottom edge (lowest tray point, about 30 mm below the frame origin in
  gravity height; the package center is about 117 mm below it).
- 2026-10-05: Package agent Phase 1A done (DevLog/parts/DevLog-002-package.md): package 108.3 x 107.8 mm
  (ring and substrate edges; analyst figure 110 mm, -1.7 percent, flagged to the audit), 32 OEs (8 per side,
  pitch 7.85 mm), stand base 72 x 72 x 5 mm; point medians 0.12-0.49 mm, edges 1.05-3.65 px. Its frame
  correction (-4.0, -3.1, -0.7) mm was folded into `config/frames.json` (package origin = substrate-top center,
  world (112.031, -150.522, 77.845) mm) and zeroed in its TOML; scores identical after the fold. Agent moved to
  Phase 1B (details, lid markings and photo textures).
- 2026-10-05: Masks: the package stand is black on a black table and SAM cannot separate it, so the stand
  (SAM object `stand`, prompted from the built stand parts) is unknown in the evaluation masks, like the sign.
- 2026-10-05: Environment input: the stand base center is at world (117.2, -230.7, -6.2) mm, so the table top
  is at gravity height about -224 mm (base bottom; scene frame, origin at the tray frame origin).
- 2026-10-05: Fresh-context audit of the Phase 0 measurement chain launched (report: DevLog-003-audit-phase0.md).
- 2026-10-05: Lower agent Phase 1A done (DevLog/parts/DevLog-002-lower.md): front bay complete (control board,
  8 copper fingers at 24 mm pitch, 8 OE frames, loops, tubes, manifold, organizers, grouped fibers, braided
  cables A (Y 128-616) and B); most parts 0.4-1.7 mm point medians, edges 1.1-2.4 px; regressions on
  board_bracket and ribbon_a_front. It measures the board and crossbar planes 0.65 deg tilted in the world frame
  (consistent with the 1.8 deg crossbar-direction disagreement; audit pending) and compensates with [tilt] in
  lower.toml (zero it if the frame is re-leveled). No copper crosses Y 320-410 (relayed to upper). Moved to
  Phase 1B (fixes, textures by session).
- 2026-10-05: Chassis agent Phase 1A done (28 objects; probe IoU 0.923 -> 0.947, edges 5.25 -> 3.19 px).
  Findings: the world frame is rolled 0.73 deg about Y (crossbar top z = 0.0127 x; wall edges agree), also seen
  by lower (0.65 deg) and upper (about 0.9 deg on the serpentine plate); the front port plate face is at
  Y about 23 mm: Y = 0 came from the acrylic sign's points (the mode of the front-view points was the sign).
  Port plate to rear lip 780 mm vs 776 published (+0.5 percent): scale consistent. Wall height 83.5-84 mm vs
  87 mm published panel height: plausibly the absent top cover (assumption). Chassis and lower compensate the
  roll in their own TOMLs (body.roll_deg, [tilt]); upper does not. One coordinated re-level after the audit.
- 2026-10-05: Upper agent Phase 1A done (50 parts; all groups on probe: IoU 0.947, sparse median 1.33 mm, from
  0.925 / 6.6 mm). Open: rear UQD fittings (two bodies per side on an outboard bracket; assigned to upper).
  All four groups moved to Phase 1B (details and photo textures, per-session shadow handling).
- 2026-10-05: Mask fixes from the chassis report: unknown polygons for the black plinth in IMG_5751 and the
  hardware below the tray in IMG_5742/5746 (`config/mask_prompts.yaml` views.*.unknown_px).
- 2026-10-05: Environment (`scripts/model/env/build.py`, `config/model/env.toml`, scene frame): table top at
  -224 mm gravity height (from the package stand base), plinth box fitted by overlays in 6 views (about 15 mm),
  ground 740 mm below the table and the green backdrop at scene Y 650 are assumptions (no sparse points on
  either); flat photo-sampled colors. Render check in 4 views: placement plausible for a basic environment.
- 2026-10-05: Lower agent Phase 1B done: flange identified (gray frame under the board), fingers split into 3
  copper blocks plus straps (pts medians 0.45-1.06 mm), 10 photo textures (board top edge 4.12 -> 1.39 px,
  color res 64.0 -> 45.5, SSIM 0.29 -> 0.46; median beat darker percentiles; sessions recorded per texture);
  flat colors refit from photo medians made most parts worse (reverted). Moved to Phase 2 (cable textures,
  cable B depth, fibers).
- 2026-10-05: Shared tool bug (lower report): texture_from_photos/bake_id_owned opened <stem>.jpeg; the
  originals here are .JPG. Fixed with find_image (any extension); all agents notified.
- 2026-10-05: Decision: the world frame is not rotated and the Y origin stays (front port plate at Y about
  23 mm). Rotating would move every part already placed by measurement in the current frame (upper places
  parts directly; chassis/lower compensate in their TOMLs) and force coordinate conversions in all TOMLs. After
  the audit, the roll goes into config/frames.json as one shared "tray" frame for tray-aligned rigid parts;
  chassis and lower switch their private compensation to it.
- 2026-10-05: Package agent Phase 1B done: ring lips (1.3 mm), OEs split into gold bracket and gray chip (pitch
  7.88 mm), one photo texture over all package tops (10 px/mm, per-part height and ID ownership; lid markings
  legible). Texture statistic chosen by leave-one-view-out: 25th percentile (in use) vs median, better combined.
  On its 5 train views (texture baked from them; holdout check pending): ring color res 45.2 -> 22.2, SSIM
  0.56 -> 0.67; lid edge 3.57 -> 2.07 px, SSIM 0.35 -> 0.61, pts 0.07 mm. Worst regions now: sky reflections on
  the lid (view-dependent; not reproducible by a diffuse texture). Stand_base check: 91 percent of its pixels in
  IMG_5712 are unknown in the mask; the remaining 9 percent edge band goes back to the agent. Moved to a short
  Phase 2 (rounding, side textures).
- 2026-10-06: Audit of Phase 0 done (DevLog/DevLog-003-audit-phase0.md). Scale holds: factor 1.000
  (0.995-1.002): 11 wall holes give 437.8 mm; port plate to rear lip 779 +- 2.5 mm vs 776; the package's 108 mm
  fits its edges and 110 mm does not (scaling to 110 would give a 445 x 793 mm tray). Correction to the entry
  above on frame and scale: the "438 x 776 box matches" check was off by about 24 mm at both ends, because Y = 0
  is the acrylic sign's printed text, not the front panel (port plate face Y = 23.5, rear lip Y = 802.5).
  Other findings and what was done:
  - Roll 0.75-1.0 deg about Y (+X higher): shared "tray" frame in config/frames.json (0.85 deg, world z =
    tray z + 0.0148 x); chassis and lower to switch their private compensation to it; world frame unchanged.
  - Holdout leakage through the joint COLMAP model (9.8 percent of tray points need holdout views): measuring
    uses data/points_train_mask.npy (25,460 of 28,775 points have >= 2 train observations; export_points.py);
    overlay.py and rayplane.py now refuse holdout views (tools.geom.refuse_holdout); world_frame.py matches
    against train views only (rerun: scale -0.018 percent, origin 0.14 mm: world.yaml kept). Joint poses accepted
    as a limitation, as in the CRT run.
  - Evaluator: hull-unknown box updated to the measured body ([-222, 222, 20, 806, -71, 26] with roll pad); tray
    point region starts at Y 15 (drops about 170 sign points); new outside_unknown rule: object pixels outside
    the projected body/front/rear/cable/package boxes are unknown (SAM spill below the side walls, IMG_5752-5757).
    The hull rule still hides excess inside the outline (3-15 percent of the hull area): accepted, noted.
  - Gravity: scripts/prep/gravity.py (eigen method on 43 views, bootstrap median 2.2 deg, 95th percentile
    5.5 deg); scene up changed by 0.25 deg (5739 now excluded).
  - IMG_5741 pose good near the rear; f 4430 px plausible; excluding IMG_5739 confirmed.
- 2026-10-06: Chassis agent Phase 1B done: lever hook heads (fit on 8 views), curved grip wing, rim clips,
  ports as plate texture; 20 photo-textured quads (own ID-owned baker). A/B: the 25th percentile lost on every
  metal surface (median kept); a single best view won for the port plate and 4 MMC groups; crossbar better from
  S3 alone, walls/rims/ducts better from all sessions. Probe, whole model: color res 54.2 -> 43.2, SSIM 0.349 ->
  0.401, edges 3.18 -> 2.92 px. Front-bay rails tried and disabled (points 7-8 mm off). Next: switch to the
  shared tray frame, wing slot, rear notches, re-bake wing/divider excluding the sign.
