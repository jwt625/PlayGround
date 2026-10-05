# DevLog-003: Fresh-context audit of the measurement and evaluation chain

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | Done (read-only audit; no project file changed except this DevLog) |
| Scope | Metric scale, world frame + undistortion, masks, evaluate.py, compare_photometric.py, render_3dgs.py |
| Author | Claude (auditor subagent), at Wentao Jiang's request |
| Blender runs used | 0 of 2 (existing ID/RGB renders were reused: `outputs/runs/crt_idtrain`, `rough_v1_holdout`) |

Method: independent scripts (scratch, deleted afterwards). They used only `tools/geom.py` camera loading,
COLMAP binaries, the photos, masks and existing render outputs. Pixel coordinates are at 1/2 scale unless marked s4.

## Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1 | Scale 52.776 mm per COLMAP unit | Confirmed to within about 1 percent. "About 12 cm" for the case width is ruled out (it needs +21 percent). |
| 2 | World frame, undistortion, cam 2, EXIF | Confirmed. No systematic offset. |
| 3 | Masks | Usable, with 4 systematic failure types. False object about 0.9 percent and false mat about 1.1 percent of object px (10 views). Two of the types bias the case metrics and fits. |
| 4 | evaluate.py / compare_photometric.py | The numbers reproduce. Problems: several biased metrics, holdout leakage, stale compare.json. |
| 5 | 3DGS baseline | The renderer is correct (aligned to 0.1-0.2 px). The comparison is optimistic for 3DGS because it trained on the holdout views. |

## 1. Metric scale: confirmed

All measurements below are in the current world frame (scale already applied). Mat points use rays to the z = 0 plane.

| Check (independent of the FFT) | Measured | Expected | Deviation |
|---|---|---|---|
| Ruler numerals "0" to "16" (IMG_1546 rectified at 6 px/mm) | 160.5 mm | 160 | +0.3 % |
| Ruler numerals "0" (IMG_1546) to "31" (IMG_1518) | 312.1 mm | 310 | +0.7 % |
| Numerals "6" to "16" / "21" to "31" (IMG_1518) | 100.8 / 101.6 mm | 100 | +0.8 / +1.6 % (reading +-1 mm) |
| Numeral spacing, all visible labels | 10 mm grid, 10 ticks per label | 1 cm / 1 mm | consistent |
| Mat near-left corner (IMG_1546) to near-right corner (IMG_1535, IMG_1556 agree) | x -189.7 to 270.8 = 460.5 mm | 450 printed | +2.3 % |
| Mat near edge y -110..-112.5 to far edge y +190.8 (IMG_1635, blurry) | about 302 mm | 300 printed | +0.7 % |
| JST-style 4-pin header pin 1 to pin 4, tri2 IMG_1628 + IMG_1629 (err 0.6-1.5 px) | 7.03 mm, pitch 2.34 mm | XH 2.50 | -6 % (weak check) |

- Mat outer edges were read on rectified images. The outer lip may sit slightly below z = 0 and the corners are rounded, so the +2.3 % length is an upper bound on any bias. The blue-point rectangle in `world_extra.yaml` (451.6 x 292.9) sits inside the true edge because sparse points do not reach it.
- Part dimensions cannot settle this alone. Electrolytic diameters (5 / 6.3 / 8 / 10 / 12.5) step by about 1.25x, so a 1.23x scale error would land on other standard sizes. The pin pitch is 6 % off 2.5 mm, and at 1.23x it would be 2.88 mm (no common 4-pin single-row pitch). Weak support only.
- Conclusion: the scale is right to within about 1 percent. If anything the world is 0.3-0.7 % too large by the numerals, which is inside the 0.95 % per-view FFT spread. A 120 mm case width would need the ruler's cm labels to be 12.1 mm apart and the 450 mm mat to be about 555 mm long. Both are contradicted. The case width of about 99.5 mm stands; Wentao's "about 12 cm" was an estimate.

## 2. World frame and undistortion: confirmed

- `load_points` and the transform reproduce `data/points_world.npy` exactly (max difference 0.0 mm). `load_views` composes the world camera correctly (checked algebraically).
- Sparse observations were undistorted analytically and compared with the projections in the s4 world cameras. In all 12 views tested, the median residual is 0.30-0.40 px s4 (p95 0.73-0.84). That equals the COLMAP raw reprojection (1.16-1.60 px full res) divided by 4. The 12 views were 3 cam-1 views, all 8 cam-2 views (1533, 1577, 1578, 1579, 1592, 1593, 1617, 1635) and the EXIF-6 views IMG_1505 and IMG_1622; cam-2 views 1533 and 1617 have EXIF 3.
- Image content check: patch NCC between the undistorted s4 image at the projection and the raw downscaled image at the observation has a median of 0.95-0.99 in every view. Raw pixels are not rotated (IMG_1505 raw 5712x4284 with EXIF 6). Principal points match `(W-1)/2` for both cameras, so the Blender shift is 0.
- Model renders: silhouette and outline shift search (+-6 px) over all 123 train ID renders in `crt_idtrain`, the 8 cam-2 views included, gives a median best shift of (0, 0) and a mean of (0.16, 0.27) px s4. The 14 holdout views in `rough_v1_holdout` give a median of (0, 0). Tile phase correlation of model RGB vs photo on IMG_1522 gives a median of (0.01, -0.32) px.

## 3. Masks: usable, with systematic errors

Sample: 10 random non-probe views (seed 20261005): IMG_1526, 1531, 1541, 1545, 1549, 1558, 1567, 1592, 1595, 1617 (two from cam 2). For each view I made an overlay and a candidate-error map against the model ID renders.
- False object: object px more than 30 px from any model pixel. 37.7 k of 4.11 M object px (0.9 %). It concentrates in wide views: 4.0 % in IMG_1541 and 9.4 % in IMG_1545.
- False mat: mat px inside the model silhouette eroded by 5 px. 45.6 k (1.1 % of object px; 0.3-4.6 % of the model interior per view). Some of this is true model excess. The visual check attributed most of it to the types below.

Systematic failure types:
1. Glossy case walls that reflect the bright mat are labeled mat (false mat). Seen in IMG_1541 (whole right wall strip), IMG_1592 and IMG_1595 (wall corner triangles). The mirror check only catches reflections much darker than the mat. Effect: correct case-wall geometry is scored as excess (fp). `fit_params.py` uses `w_fp = 10` with `w_fn = 0` by default, so case fits are pushed inward.
2. Light-blue CRT socket cap and blue wires are labeled mat (false mat). Seen in IMG_1558, 1617, 1592, 1526. Point evidence fills them only partially.
3. Mat lip lines, groove shadows and the parts-box edge are labeled object (false object): long thin lines along the mat border and the "Circuit board location area" groove. Seen in IMG_1541, 1545, 1558. They show up as `fn_unmodeled`.
4. Known-region truncation: in low views, the parts of the object seen against anything beyond the mat rectangle are marked 128 and never scored. This is 34 % of the model's pixels in IMG_1595, 13 % in 1592 and 8 % in 1567, so tall parts (yoke top, far case wall) are unscored in exactly the views that constrain their height. This follows from the design, but it is not reported anywhere.

## 4. Evaluation code

Recomputed independently on `rough_v1_holdout`:
- IoU on IMG_1522 is 0.9606 vs 0.9606 reported; on IMG_1583 it is 0.802 vs 0.8009. ID decode: 0 undecoded px. Palette minimum L1 distance is 16, above 2 x tol, so decoding is unambiguous at 115 objects.
- Points: median 0.4471 mm, 71.9 % <= 1 mm, 10.8 % > 4 mm. This matches `summary.json` exactly (computed with the selection before the mat-blue filter, which was added to evaluate.py during this audit).

Problems:
1. The per-part point median is truncated. `points_eval` uses only points within 4 mm (line ~294), so a part with many far points looks good. Example: `wires.white_pair` reports 0.43 mm, but 549 of its 1217 nearest points are more than 4 mm away and the untruncated median is 2.66 mm. Similar: ext_pot_2 (2.14 vs 2.68 mm), case.screen_box (0.82 vs 0.93 mm).
2. Nearest-surface attribution: each point belongs to whichever object surface is nearest. An oversized part takes points that belong to its neighbours and gets small distances. The point metric is also one-sided (data to model), so excess geometry is not penalized in 3D.
3. The edge chamfer is one-sided (model to photo, no recall) and truncated at 10 px. In cluttered regions it has little dynamic range. IMG_1522: model ID edges 2.47 px; random object pixels 5.6 px; photo-to-model recall 4.6 px. Internal ID boundaries between touching coplanar parts count as edges, which penalizes how finely parts are split.
4. The IoU 2 px band forgives silhouette offsets up to about 2 px (about 0.5-1 mm). IMG_1522 goes from 0.942 without the band to 0.961 with it. IoU barely responds to the small offsets that matter now.
5. Holdout leakage: `config/model/case.toml` line 35 (`far_height`) uses overlays on holdout IMG_1595. The env mat texture (`env_mat_texture_spec.json`, `views: auto`, per_texel) samples every view, the holdout views included. `texture_from_photos.py` "auto" never excludes holdout views. The crt and pcb textures checked (yoke script, pcb lists) use train views only.
6. `outputs/compare/rough_v1_holdout/compare.json` (02:55) predates the current `rough_v1_holdout/rgb` renders (03:58). Model PSNR for IMG_1522 recomputes to 13.91 dB raw vs 13.30 in the file; the 3DGS values reproduce exactly (24.33). The DevLog-001 photometric numbers are therefore not reproducible from the current run folder.
7. Context for the photometric numbers (IMG_1522, object mask): a constant mean color scores 10.6 dB; the photo shifted by 1 / 2 / 3 px scores 28.0 / 23.7 / 21.6 dB; the photo blurred with sigma 4 scores 22.3 dB. The model at 12.6-15.4 dB is close to the constant-color floor. A color fit on the object only gives 15.4 dB vs 14.4 dB when fitted over object + mat. The new `psnr_fit_blur4` is a reasonable addition.
8. Minor: the SSIM window (sigma 2) mixes in pixels outside the mask at mask borders. PSNR is averaged in dB across views rather than pooled by MSE (the usual convention, but say so).

## 5. 3DGS baseline renderer

- Correct: the SH basis and signs, f_rest channel-major layout, EWA covariance with a 0.3 px^2 low-pass, pixel-index convention and front-to-back compositing all match the reference. Alignment to the photos is 0.10 / 0.21 px median absolute tile shift (IMG_1522), so the Brush training handled the lens model consistently.
- Same views and same region: yes. The same eroded object mask is used, and the color-fit region (the model's alpha coverage) is the same for both methods.
- Optimistic for 3DGS: it trained on all 137 photos, so its "holdout" scores (22.1 dB / 0.79) are training-view fits. It also drops Gaussians whose radius exceeds half the image (no effect on the object region).

## Fixes ranked by impact

1. Masks, reflections: label glossy-wall reflections as object or unknown. Mark as 128 every pixel whose ray hits the mat within about 8 mm outside the current case-model footprint, or use the case ID render as a prior. Until then, run case fits with `--w-fn` > 0 or lower `w_fp`, and treat case fp as unreliable.
2. Per-part points: report the untruncated median and the fraction > 4 mm per part next to the truncated median. Flag parts where more than 10 % of their points are > 4 mm.
3. Holdout hygiene: make `texture_from_photos.py` "auto" and per_texel use `split["train"]` only. Rebuild `env_mat.png`. Remove the IMG_1595 dependency in case.toml or re-justify it with train views. Rerun compare for `rough_v1_holdout` so the numbers match the renders.
4. Masks, false object: drop thin elongated components that touch the mat-border band (or use the env_mat texture as a mat-appearance reference). Add the socket cap and blue wires to the evidence step: use the model ID as a prior, or lower the point z threshold near the neck.
5. Metrics: add a symmetric edge chamfer (photo-edge recall inside a dilated model region) and a no-band IoU (or a boundary F-score at 1 px). Report the fraction of model px in unknown per view so unscored geometry is visible.
6. 3DGS: label it as "trained on all views" in every table. For a fair holdout number, retrain Brush without the 14 holdout views.
7. Minor: assert palette min L1 > 2 x decode tolerance in `render_views.py` as the object count grows. Pool PSNR by MSE in summaries.

## Progress log

- 2026-10-05: Read AGENT_GUIDE, DevLog-001 and the prep/eval/tool scripts. Ran scale checks (ruler numerals, mat corners, connector pitch), reprojection and undistortion checks (12 views), silhouette shift tests (137 renders), mask sampling (10 views), metric recomputation (IoU, points, PSNR) and 3DGS alignment. Scratch in outputs/scratch/audit/ deleted.
