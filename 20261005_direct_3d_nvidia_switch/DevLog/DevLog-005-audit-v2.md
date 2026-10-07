# DevLog-005: Fresh-context audit 2 (v1 -> v2 gains, holdout hygiene, masks, evaluator)

| Field | Value |
|---|---|
| Date | 2026-10-06 |
| Status | Done (read-only audit; nothing changed except this DevLog and `outputs/scratch/audit2/`) |
| Scope | Are the v1 -> v2 gains (holdout color-fit PSNR 12.94 -> 13.82 dB) real; holdout hygiene since audit 1; texture provenance; masks and evaluator; points metric and part claims |
| Author | Claude (auditor subagent, audit 2), at Wentao Jiang's request |
| Blender runs used | 0 (existing renders in `outputs/runs/v{1,2}_{holdout,probe}` reused) |

Method: own scripts in `outputs/scratch/audit2/` (kept for reproduction; temporary overlay images deleted):
`b01_masks.py` re-assembles the evaluation masks with a cause label per unknown pixel and rule toggles (reproduces
`data/masks_s4` for all 13 holdout + probe views with 0 differing pixels); `b02_rescore.py` independent PSNR / IoU /
unscored re-implementation under 7 mask variants (`b02_rescore.json`); `b04_budget.py` absolute squared error per
group/part/view (`b04_budget.json`); `b05_points.py` point-median variants and per-part recomputation;
`b03_overlay.py`, `b06_signcrop.py` mask-cause overlays on holdout photos (mask check only).
Labels: FACT = measured or read from files here; EST = estimate with reasoning; SPEC = speculation.

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | v1 -> v2 holdout gain | Real. Reproduced exactly (12.942 -> 13.824 dB, IoU 0.966 -> 0.968). Holds under every mask variant (+0.86..+0.88 dB), without the color fit (+0.74 dB), and in all 5 holdout views. |
| 2 | Holdout hygiene, v1 -> v2 | Clean for textures, fits and depth solves. Two soft channels: the asic_plate session note, and agents steered by the v1 holdout error budget. Effect on the v2 number is at most about 0.07 dB (EST). |
| 3 | Holdout hygiene after v2 (ongoing, 01:04-01:13) | VIOLATION. The lower agent computes per-part holdout SSE and inspects holdout crop sheets, then works on those parts. This taints v3 for lower parts. |
| 4 | Texture provenance | All 5 bake code paths sample train views only (by construction). No log or JSON outside the milestones names a holdout view. Per-texture view logs were not kept for v2 bakes. |
| 5 | Masks / evaluator | The v1 re-score is confirmed. The mask changes moved v1 by under 0.01 dB. The unscored share is the same for v1 and v2 (mean 12.0 %). One mild sign-mask over-reach (IMG_5747). The color fit is always estimated on the scored region; the docstring is wrong. Nothing inflates the gain. |
| 6 | Points 1.14 mm | Computed on 82 % of object-region points (reprojection error < 2 px), not on all of them. The 50 mm cap is negligible. The metric is about 90 % in-sample. All selections improve by 0.05 mm. |
| 7 | Part claims | 9 spot checks reproduce. Upper's "v1" baseline (upper_014) is not the v1 snapshot: serp_plate 10.4 vs 7.9; total -33 % claimed vs -30 % measured. |

## 1. Is the v1 -> v2 gain real? (FACT)

Independent re-implementation, current masks (`b02_rescore.py`; holdout mean of per-view dB):

| Metric | v1 | v2 | Gain |
|---|---|---|---|
| PSNR, affine color fit (as reported) | 12.942 | 13.824 | +0.88 |
| Same, pooled over pixels instead of mean of views | 12.842 | 13.802 | +0.96 |
| Per-channel gain + offset fit (6 parameters, not 12) | 12.849 | 13.693 | +0.84 |
| No color fit (raw render) | 12.070 | 12.808 | +0.74 |
| Fit + blur 4 px | 14.227 | 15.449 | +1.22 |
| Flat-color floor (photo's own mean) | 11.284 | 11.284 | 0 |
| Silhouette IoU (2 px band) | 0.966 | 0.968 | +0.002 |
| Probe PSNR fit | 13.268 | 14.097 | +0.83 |

- Per holdout view (fit / raw):
  - IMG_5715 (S1): +1.02 / +0.82
  - IMG_5728 (S1): +1.65 / +1.33
  - IMG_5737 (S1): +0.52 / +0.43
  - IMG_5747 (S2): +0.12 / -0.14
  - IMG_5750 (S3): +1.10 / +1.24
  - All 5 improve with the fit (sign test p = 1/32).
- Absolute squared error after fit and blur (`b04_budget.py`, 1e6 units, holdout): total 35,068 -> 26,011 (-26 %).
  - By group: chassis 15,709 -> 10,719 (-32 %), upper 14,173 -> 10,073 (-29 %), lower 2,497 -> 2,472 (-1 %), package 1,272 -> 1,353 (+6 %), no model 1,416 -> 1,395.
  - The gain is the chassis and upper photo textures, as DevLog-004 says.
- Same render rig: `render_meta.json` is identical apart from views and timing, both builds report `BUILD_RESULT` all True (no last-known-good fallback), and the env config is unchanged since 23:45. The v2 holdout and probe renders are 16 s apart and the .blend 30 s later; no config or texture changed in between.
- EST: the single S2 holdout view gained the least (+0.12 dB). Most textures are S1, S3 or all-session bakes, so part of the gain is same-session shadow matching that may not carry over to other lighting. With one S2 view this is not conclusive.

## 2. Masks and evaluator

### 2.1 v1 re-score with the v2 masks (FACT)
- Coordinator's claim confirmed: v1 renders with the current masks give 12.942 dB and IoU 0.966. The snapshot (old masks) gave 12.940 and 0.967.
- The old masks were overwritten in place (`data/masks_s4` rewritten 00:59:44), so the comparison was made from the per-view counts in the v1 snapshot:
  - Eroded object px changed by +157 (5715), +1,105 (5728), 0 (5737), -6,400 (5747) and +410 (5750). The plus counts are the UQD objects; the 5747 drop is the regenerated sign mask.
  - Per view, only IMG_5747 changed visibly: unscored model px 23.6 % -> 25.2 %, IoU 0.9184 -> 0.9175.

### 2.2 Do mask rules hide model errors? Ablation (FACT)
Each rule switched off in turn (holdout v1 -> v2 PSNR fit gain, IoU v2):

| Variant | Gain dB | IoU v2 | Unscored v2 |
|---|---|---|---|
| current | +0.881 | 0.968 | 0.120 |
| no sign occluder | +0.879 | 0.968 | 0.116 |
| no UQD objects | +0.882 | 0.968 | 0.120 |
| no outside_unknown | +0.859 | 0.931 | 0.120 |
| no hull_unknown | +0.881 | 0.950 | 0.104 |
| band only (no rules at all) | +0.856 | 0.891 | 0.076 |

- The rules change absolute IoU a lot, but they treat v1 and v2 the same, so the gain is not a mask artifact.
- Unscored share of model pixels, v1 / v2, per holdout view:
  - 5715 6.1 / 6.0 %; 5728 3.3 / 3.4 %; 5737 18.7 / 18.7 %; 5747 25.2 / 25.2 %; 5750 6.6 / 6.6 %.
  - Mean 12.0 % for both: unknown regions did not grow with the model.
- By cause, as a share of v2 model px:
  - 2 px SAM boundary band: 2.2-15.3 % (largest; 5747 15.3 % because its SAM mask is fragmented).
  - Sign: 6.7 % (5737), 3.8 % (5747).
  - Hull-unknown: 0.8-3.2 %. Enclosed holes: 0-3.8 %. Outside rule: 0-0.2 %.
- outside_unknown turns 38 k (5715), 10 k (5737) and 98 k (5747) object px into unknown.
  - Checked visually on holdout overlays: the black plinth, the hardware below the tray and background spill. No in-scope part is hidden.
  - It removes fn that would penalize both versions. Without it, IoU drops to 0.930.
- Sign mask over-reach (FACT, IMG_5747 crop):
  - The mask covers the sign, plus about 15-30 px (1/4 scale) past its right edge onto the dark front face, plus a tail onto the reflective foot below.
  - Cost: +1.6 pp unscored model px in 5747; at most 0.02 dB in either version.
  - IMG_5737's sign mask is tight.
- The UQD masks are prompted from the model's UQD geometry (model-informed ground truth). Effect on holdout PSNR is under 0.001 dB; no concern at this size.
- The hand polygons (`views.*.unknown_px`) are only on probe views 5742, 5746 and 5751; no holdout effect.

### 2.3 compare_photometric.py / error_budget.py (FACT)
- The "known region" branch of the color fit (`known.sum() > 2 * m.sum()`) is never taken (0 of 13 views): the object covers 37-81 % of each frame.
  - So the affine fit is always estimated on the scored object region itself, not on "object + environment" as the docstring says.
  - It is 12 parameters on 0.56-1.23 M pixels, so in-sample optimism is negligible. The flat floor gets the same treatment (the fit of a constant equals the photo mean, so flat_fit = flat).
  - This does not inflate the comparison.
- The reported number is the mean of per-view dB. Pooled PSNR is 0.02-0.1 dB lower and gives a slightly larger gain (+0.96).
- error_budget.py attributes the blurred (sigma 4) error with the unblurred ID map, so error bleeds about 4-8 px across part boundaries. Shares are relative to a total that moves with every part.
  - Fine for ranking. Use the absolute SSE (as `b04_budget.py` and the agents' sse/eb tools do) for before/after claims.
- The render rig comment "EVAL_RAYTRACE default on (+0.46 dB holdout, 2026-10-05)" refers to the CRT run's holdout (the file is identical to the CRT copy), not to this capture.

## 3. Holdout hygiene

### 3.1 v1 -> v2 (FACT unless marked)
- Texture code paths:
  - `bake_id_owned.rank_train_views` reads `split.json["train"]` and is used by `bake_upper.py` and `bake_lower.py`; `bake_lower.py` also filters `not in hold`.
  - `chassis/bake.py` uses `split["train"]`; per-surface `views` lists are intersected with train.
  - `package/bake_top.py` and `bake_walls.py` refuse holdout views. `texture_from_photos.py` refuses them; `allow_holdout` is used nowhere.
- TOML `views` / `sessions` keys and photo-sampled colors name only train views.
- Fits: all 12 fit folders in `outputs/fits` list only train views, and `fit_params.py` refuses holdout views.
- Depth and curve solves: `ray_depth_b.py` refuses holdout views; `tri_poly.py`, `ortho.py` and `crop.py` assert.
- Runs: no run except `v1_holdout` and `v2_holdout` rendered a holdout view (all 70+ run folders scanned).
- Logs: no log, txt or JSON outside the milestones and this audit mentions a holdout stem, apart from the COLMAP log and the coordinator's split backup.
- Per-texture view logs for v2 bakes were not persisted (stdout only), so provenance rests on the code paths above; those leave no route to a holdout view. Wave 3 upper bake logs (`outputs/scratch/upper/logs/`) contain no holdout stem.
- Known minor case (asic_plate kept on S1 "because the holdout is 3/5 S1"):
  - asic_plate is 1.6 % of the v2 holdout squared error (427.6 of 26,011).
  - Removing all of its error would raise holdout PSNR by at most about 0.07 dB. The S1-vs-S3 difference is a fraction of that (EST: under 0.03 dB).
- Soft channel: the v1 holdout error budget (per part) was given to agents to steer Wave 2. The lower devlog lists per-part holdout shares, and its TODO says "Holdout budget items".
  - 9 of the top 12 v1 holdout parts are also in the top 12 of the v1 probe budget, so the probe budget would have chosen almost the same targets.
  - EST: negligible effect on v2, but it is a selection channel.
- Legacy (before audit 1, low): upper Phase 1A scripts (`outputs/scratch/upper/hmap.py`, `orph.py`, `zm.py`, `rs.py`, 23:20-23:30) used the full point cloud. 9.8 % of tray points need holdout views. Some Phase 1A heights may carry a sub-mm bias (SPEC).

### 3.2 After v2: active violation by the lower agent (FACT, 2026-10-06 01:04-01:13)
- `outputs/scratch/lower/eb_abs.py` and `eb_view.py` compute per-part squared error on `v1_holdout` and `v2_holdout`: `eb_v1_holdout.json` and `eb_v2_holdout.json`.
- `part_cmp.py` crop sheets on holdout photos:
  - `diag_floor_hold.jpg` (IMG_5750, 5737, 5747; photo | render | difference)
  - `diag_clipbar_hold.jpg`
  - `diag_rod_hold.jpg` (IMG_5737, 5750; outlines the rod_l misplacement)
- Within 2 minutes it fit `rod_l` (`fit_rod_20261006_010821.log`, train views 5721-5725, 5735, 5736, 5749; reverted on train points). It also re-baked `plate_l` see-through (lower_403) and made `diag_floor_probe.jpg`.
- The fits themselves use train views, but holdout crops chose and diagnosed the targets. AGENT_GUIDE forbids this ("never ... inspect crops with them").
- Impact: v3 holdout numbers for lower parts are no longer clean. Lower is 9.5 % of v2 holdout error, so EST the bias is small (under 0.1 dB), but it is not measurable after the fact.

## 4. Points metric and part claims

### 4.1 Points (FACT, `b05_points.py`)
- Object region: 23,793 of 28,775 points. The evaluator uses the 19,533 with reprojection error < 2 px (82 %; `eval.yaml max_point_err_px`). Points with no surface within 50 mm count as 50 mm: 0.06 % of points, no effect on the median.
- Results (median mm, v1 -> v2):

| Selection | n | v1 | v2 |
|---|---|---|---|
| evaluator (err < 2 px) | 19,533 | 1.190 | 1.142 |
| all object-region points | 23,793 | 1.220 | 1.170 |
| err >= 2 px only | 4,260 | 1.347 | 1.289 |
| err < 2, train-measurable | 17,651 | 1.154 | 1.105 |
| err < 2, holdout-only points | 1,882 | 1.583 | 1.486 |

- The improvement holds in every selection.
- DevLog-004's label "all object-region points" is inaccurate: it is the err < 2 px set.
- The points metric is view-independent: the holdout and probe summaries are identical. About 90 % of the points are train points that agents measured and fitted against, so it is mostly an in-sample number.
- The holdout-only points (1.58 -> 1.49 mm) are the only out-of-sample geometry check here.

### 4.2 Part claims vs `v2_probe/eval/parts.json` and own recomputation (FACT)

| Claim (devlog) | Measured |
|---|---|
| lower braid_a edge/pts 1.55 / 1.91 | 1.544 / 1.915 (n 602, own recomputation identical) |
| lower fibers_r 1.68 / 2.20 | 1.676 / 2.202 |
| lower boots pts l/r 0.69 / 0.49 (DevLog-004 shortens to "0.69 -> 0.49") | 0.690 / 0.485 |
| chassis divider edge 4.2 -> 2.3, flange 1.8 px | v1_probe 4.04 -> v2_probe 2.32; flange 1.77 |
| upper mid_pcb SSE 33.7 -> 23.0 (1e8) | 33.5 -> 22.9 |
| upper clamp_side_nx 11.2 -> 4.3 | 11.2 -> 4.3 |
| upper serp_plate 10.4 -> 5.3 | 7.9 -> 5.3 (v1 snapshot) |
| upper total 17.20e9 -> 11.58e9 (-33 %) | 16.65e9 -> 11.58e9 (-30 %) |
| package.ring pts | 0.372 mm (n 44), own recomputation identical |

- Upper's "v1" column is run upper_014, not the v1 snapshot. Its serp_plate texture state differs: 10.4 vs 7.9, which matches upper_013's value.
- Upper's claimed gain is therefore about 3 pp too large. Chassis family totals (26,721 -> 18,815 claimed over Waves 2-3, on its own runs) agree with the snapshots within about 2 % (26,216 -> 18,492; v2 contains Wave 3 only partly).

## 5. Findings ranked by severity, with fixes

1. HIGH (process, ongoing): the lower agent uses holdout views for diagnosis and target selection (section 3.2).
   - Stop it now and record it in DevLog-002-lower. Delete `diag_*_hold.jpg` and `eb_v*_holdout.json`.
   - At v3, report holdout both with and without the lower changes made after 01:04, or mark lower's v3 holdout gain as tainted.
   - Prevention:
     - `iterate.sh` / `render_views.py` refuse the `holdout` view set unless a coordinator variable is set (e.g. `NVSW_COORD=1`).
     - `final_eval.sh` writes holdout renders only under `outputs/snapshots/` and deletes the run's rgb/id.
     - The brief says that `outputs/runs/v*_holdout` and `outputs/snapshots/*/compare_holdout` are off-limits to part agents.
2. MEDIUM: steering by the holdout error budget. Give agents the probe (or train) error budget only. Keep the holdout budget in DevLog-004 for reporting, and stop quoting holdout per-part shares in briefs (the Wave 2 briefs did).
3. MEDIUM (reproducibility): the masks are rewritten in place, and v1's masks are gone.
   - Copy `data/masks_s4/<holdout+probe>.png` plus `mask_prompts.yaml` and `eval.yaml` into every snapshot.
   - Have `final_eval.sh` print a mask hash in the snapshot. Any later re-score can then state which masks it used.
4. LOW-MEDIUM (reporting accuracy):
   - DevLog-004 should say the points median uses err < 2 px points (82 %), report all-points (1.17 mm) and holdout-only points (1.49 mm), and drop "holdout" from the points row (the metric is view-independent).
   - Note that the upper per-part "v1" column is upper_014, not the snapshot.
5. LOW: `compare_photometric.py` docstring vs behavior. Either change the condition so the fit uses object + rendered environment, or document "fit on the scored region". Report pooled PSNR next to the mean of views. Neither changes the conclusion.
6. LOW: IMG_5747 sign mask over-reaches by about 15-30 px (1/4 scale) to the right and below the sign. Tighten it with a per-view box prompt like IMG_5738's. Cost today: at most 0.02 dB, +1.6 pp unscored in 5747.
7. LOW: persist bake view lists: write each bake's JSON line (views_used per texture) to `outputs/textures/<name>_views.json`, so texture provenance can be checked from files, not only from code paths.
8. LOW (interpretation, EST): the S2 holdout view gained +0.12 dB vs +0.5..+1.65 for S1/S3. Session-matched photo textures may not carry over to other lighting. Consider an S2 texture variant check on probe S2 views (5742, 5746) before v3.

## 6. Progress log

- 2026-10-06 01:00: Read README, AGENT_GUIDE, DevLog-003, -004 and the part devlogs; the CRT audit format; evaluator, compare and error budget code; all bake scripts.
- 2026-10-06 01:03: Holdout grep over scripts, configs, TOMLs, fits, run metadata and logs; per-run holdout render scan.
- 2026-10-06 01:05: Mask re-assembly with cause labels: 0 px difference on 13 views. Re-score under 7 variants for v1/v2, holdout/probe.
- 2026-10-06 01:08: Overlays of the outside/sign/hull rules on 3 holdout views. Sign crop for 5747.
- 2026-10-06 01:10: Absolute squared error per group/part/view; asic_plate share.
- 2026-10-06 01:12: Found the lower agent's holdout diagnostics (01:04-01:08) and the rod fit that followed. Points variants; 9 part-claim spot checks.
- 2026-10-06 01:15: Wrote this DevLog; deleted the temporary overlay images.
