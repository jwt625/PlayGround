# DevLog-000: Direct 3D modeling from photos (instead of 3DGS) - kickoff and dataset pick

| Field | Value |
|---|---|
| Date | 2026-10-04 |
| Status | Done: Wentao picked B (CRT display) and answered Section 6; plan continues in DevLog-001 |
| Scope | Agents build an explicit, editable 3D model (Blender, bpy scripts) of a captured object directly from its photos, steered by visual inspection plus coded render-vs-photo evaluation at the COLMAP camera poses |
| Inspired by | `../20261002_interconnect_film` (bpy asset library, `ASSET_SPEC.md` conventions, agent-wave workflow) |
| Authors | Claude, at Wentao Jiang's request |

Path convention: paths are relative to this project folder. Source captures live outside the repo at `~/Documents/3DGS/<set>/` and are treated as read-only (nothing is written there).

## 1. Instructions (verbatim from Wentao, 2026-10-04)

```text
now take a look at ~/Documents/3DGS. [absolute path shortened for the repo]
Here is the idea: 3dgs is a dumb way for scene reconstruction, and I want you and your subagents to directly build the actual 3d model from the imagesm with a combination of direct visual comparison and inspection, as well as coding/eval assisted comparison, such as evaluate the diff of rendered vs actual images. Note there are existing colmap data that you could utilize for setting camera pose and properties for rendering.
Now pick two of the existing 3dgs projects to test this, and i will pick one from them. Ask any clarification questions you have. Create a new playground project folder for this.
```

## 2. Survey of `~/Documents/3DGS` (2026-10-04)

Each set has `images/`, a COLMAP `sparse/<n>/` model (cameras.bin, images.bin, points3D.bin), mostly a `database.db`, and a trained 3DGS `.ply` (exported from Brush; for the coherent laser that is 544,539 Gaussians, y-up). Photos are iPhone frames at 5712x4284. The table below comes from parsing the binary COLMAP models (scratch script, not kept).

| Set | Registered / images | Camera model | Sparse pts | Median reproj. err (px) | Subject | Metric scale in frame |
|---|---|---|---|---|---|---|
| 20250413_coherent_laser | 181 / 181 | SIMPLE_RADIAL, 1 cam, f 4612 px | 77,973 | 1.40 | Gold hermetic laser package, lid off: feedthrough pin rows, ceramic substrates with gold traces, optics, bond wires | Yes: cm grid plus cm/inch ruler mat |
| 20250414_coherent_laser_v2 | 127 / 128 | SIMPLE_RADIAL, 1 cam | 46,712 | 1.44 | Same package (appears to be), second capture with the label side visible | Yes: same mat |
| 20250420_spectra_physics_navigator | 114 / 136 | SIMPLE_RADIAL, 1 cam | 11,970 | 1.20 | Machined aluminum laser head housing, interior; many close-ups | Partial |
| 20250502_blueFors | 32 / 154 | OPENCV, 2 cams | 7,495 | 2.26 | Dilution fridge stages | No; COLMAP model broken (one camera center diverges) |
| 20250504_wirebonoder | 90 / 92 | SIMPLE_RADIAL, 1 cam | 6,875 | 1.30 | TPT HB16 wire bonder, large machine plus lab clutter | No |
| 20250619_Laser_Phosphor_Display | 139 / 139 | SIMPLE_RADIAL, 1 cam | 36,511 | 1.35 | Disassembled laser engine, about 6 loose parts | Yes: grid mat |
| 20251004_CRT_display | 137 / 137 | SIMPLE_RADIAL, 2 cams | 38,376 | 1.40 | Small CRT monitor, case open: CRT neck (glass), deflection yoke, flyback, PCB with electrolytics, wiring | No ruler seen in the sampled frames |
| innolight | 82 / 82 | SIMPLE_RADIAL, 82 cams (per-image intrinsics) | 27,229 | 1.24 | Three transceiver boards, nearly planar | Yes: grid mat |
| intel100G (sparse/1) | 46 / 47 | SIMPLE_RADIAL, 46 cams | 7,788 | 1.00 | Intel 100G CWDM4 module PCB, small in frame; sparse/0 is a 3-image fragment | Yes: mat, object small |
| innolight_round1, blueFors_DA3_gallery | n/a | n/a | n/a | n/a | No images/sparse model (3DGS ply and a DA3 export only) | n/a |

## 3. Proposed candidates (Wentao picks one)

### A. `20250413_coherent_laser` (recommended)

- Best registration of all sets (181/181, 78k points, 1.40 px).
- Mostly rigid, prismatic, machined geometry that suits parametric bpy modeling: box housing, lid, seal ring, pin rows, substrates, mounts. The detail runs all the way down to bond wires and printed traces, so how far down the modeling goes is a real choice (question 2).
- Metric scale and a ground plane come straight from the cm grid and ruler under the object.
- The v2 capture of the same package is an independent second photo set. It can serve as a true held-out validation set once registered to the v1 frame, which no train/test split inside one capture can match.
- Hard parts: specular gold and silver surfaces (view-dependent highlights inflate photometric error), thin wires, and seal solder.

### B. `20251004_CRT_display`

- Full registration (137/137, 38k points); a deliberately different test from A.
- Mixed shapes: an open black plastic case, a glass CRT neck and socket, a cylindrical yoke wrapped in tape, a flyback block, a PCB populated with many repeated parts (electrolytics, chokes, connectors), and loose wire bundles.
- It tests repeated-part instancing, curved and soft geometry (wires, tape) and transparency, where A tests precise machined geometry.
- Hard parts: no in-frame ruler found, so metric scale needs one measured dimension or a known part size; glass refraction; wire routing.

Runner-up: `20250619_Laser_Phosphor_Display` (multi-part assembly on a grid mat). It was not picked because the loose parts lie in arbitrary poses and many small plastic parts carry little geometry information.

## 4. Proposed method (draft; refined after the pick)

1. Cameras: undistort the chosen set with `colmap image_undistorter` (SIMPLE_RADIAL to PINHOLE) into `data/` here, because Blender cameras have no lens distortion. Build Blender cameras from the undistorted intrinsics and poses (COLMAP world-to-camera, +Y down / +Z forward, converted to Blender's camera frame).
2. World frame: fit the mat plane from sparse points to get Z up, scale from the grid pitch to mm, and put the origin at the object's footprint center. Same conventions as the film's `ASSET_SPEC.md`: 1 BU = 1 m, real size, `ROOT_` empty.
3. Ground-truth masks per view: object vs mat and background. The default is color keying on the blue/white mat plus manual polygon fixes; a segmentation model only if keying fails.
4. Modeling loop (agents, one part group each, at most 5 at a time): write or modify a bpy build script, render the views, run the eval, look at the side-by-side, overlay and diff sheets, and edit the script. The model stays parametric (dimensions in config, not hard-coded).
5. Eval (`scripts/eval/`), on about 10 percent of views held out from modeling plus the v2 capture if A is picked:
   - silhouette IoU against the masks;
   - edge alignment (chamfer distance in px between photo and render edge maps);
   - masked PSNR / SSIM / LPIPS on color;
   - geometric distance of COLMAP sparse points to the model surface in mm;
   - per-part error heatmaps so each agent can see where its part is off.
6. Baseline: the existing Brush 3DGS rendered at the same views. It was trained on all images, which favors 3DGS; see question 3.
7. Renders run at 1/4 resolution (1428x1071) during iteration, full resolution only for final figures. Blender 4.2.3 is installed at `/Applications/Blender.app` (not on PATH); `colmap` is at `/opt/homebrew/bin/colmap`. The 3-concurrent-Blender cap from the film project applies.

## 5. TODO

- [x] Survey the capture sets; propose two candidates
- [x] Wentao picks A or B and answers Section 6 (B, 2026-10-04)
- [ ] Write the plan into DevLog-001 (data prep, frame alignment, eval harness, agent split by part group)
- [ ] Data prep: undistort, world frame and scale, masks, holdout split
- [ ] Eval harness plus baseline numbers (3DGS and an empty or bounding-box model, as floor and ceiling references)
- [ ] Modeling waves with eval after each; fresh-context audit

## 6. Clarification questions

1. Materials and textures: should surfaces use procedural PBR only, or may photo-projected textures be used for flat printed content (labels, PCB silkscreen, trace artwork)? Projection makes the color metrics look much better without improving geometry, so if allowed it would be reported as a separate tier. Default: procedural only for geometry scoring, projection as an optional final layer.
2. Detail floor: how fine should the modeling go? Default for A: everything resolvable at about 0.3 mm, including individual pins, substrate outlines, components and bond wires as curves; printed traces only via question 1.
3. 3DGS baseline fairness: compare against the existing Brush ply as-is (it saw the held-out views), or retrain 3DGS on the same split (remote GPU box)? Default: as-is, with the caveat stated, plus the v2 cross-capture check if A is picked.
4. Library reuse: should the model follow the film project's `ASSET_SPEC.md`, so the result can drop into that asset library? Default: yes.
5. Only if B is picked: can you measure one dimension of the CRT unit (for example, case width), or should scale come from a known part such as the CRT neck diameter, flagged as an estimate?

### 6.1 Answers (Wentao, 2026-10-04, verbatim)

```text
lets do the CRT one. You can have up to 5 subagents and up to 2 blender instances. Test and explore whats the best workflow, esp the evaluation and feedback. Come up with good balance between speed and accuracy, e.g., use code to find regions to focus on more for your builtin visual inspection.
1. labels should or can be modeled properly as textures, should be as important as structural.
2. default sounds good.
3. as is. No use of remote GPU for this project.
4. yes but build it in here, do not share across projects. Later on we will consolidate the efforts that have synergy.
5. its width (shorter edge) is about 12 cm.
```

## 7. Progress log

- 2026-10-04: Read the interconnect film project for context. Surveyed `~/Documents/3DGS` (parsed all COLMAP models, made a 6-photo sheet per set). Proposed A (coherent laser) and B (CRT display). Created this folder and DevLog.
- 2026-10-04: Wentao picked B (CRT display); answers recorded in 6.1. Plan in DevLog-001.
