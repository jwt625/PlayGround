# DevLog-000: Direct 3D model of an NVIDIA CPO switch tray and its CPO package - kickoff and plan

| Field | Value |
|---|---|
| Date | 2026-10-05 |
| Status | Initial inspection done; plan drafted; waiting for Wentao's answers to Section 6 |
| Scope | Explicit, editable Blender model (bpy scripts + TOML parameters) of the "Spectrum-X CPO Switch Tray" and the CPO package on the stand in front of it, built from 44 photos with no prior COLMAP model and no 3DGS baseline |
| Predecessor | `../20261004_direct_3d_from_images` (CRT run): README, PLAYBOOK.md, AGENT_GUIDE.md, VIDEO_WORKFLOW.md, DevLog-000/001/003 |
| Authors | Claude (coordinator), at Wentao Jiang's request |

Path convention: paths are relative to this project folder. The source photos stay outside the repo and are
read-only (nothing is written there).

## 1. Instructions (verbatim from Wentao, 2026-10-05)

```text
read project 20261004_direct_3d_from_images, and now I want you to carry out the next one, and also slightly different one: I have a new set of photos in '[~/Downloads]/20261005-nvidia switch', and I want you to do the same kind of blender 3d reconstruction of the nvidia switch as well as the switch chip in front of it. Note this time there is no more colmap data or 3dgs baseline. You are building blender fron scratch. Read the prev project first, and create new folder for this project, and start writing down a devlog to record the plan. Ask me any clarification questions you have after initial inspection. Apply similar rules from the prev project to this project. Keep track of progress in various docs accordingly as you and your subagents proceed.
```

## 2. Data inventory (verified 2026-10-05)

- Source: `~/Downloads/20261005-nvidia switch/`, 44 JPEGs IMG_5711..IMG_5758 (gaps: 5716, 5718-5720), 3.7-6.0 MB each.
- Camera: one iPhone 15, main camera (EXIF "back dual wide camera 5.96mm f/1.6", 26 mm equivalent), all
  5712x4284 raw pixel order. EXIF orientation 6 for 38 photos, 3 for IMG_5735-5740 (landscape). The same rule as
  the CRT run applies: COLMAP and every loader use raw pixel order (`cv2.IMREAD_IGNORE_ORIENTATION`); EXIF
  rotation only for human-facing contact sheets. One shared intrinsic set is expected (same lens, same focal).
- EXIF capture times (2026-08-24), three sessions:

| Session | Time | Photos | Content |
|---|---|---|---|
| S1 | 12:31:41-12:37:13 | 5711-5715, 5717, 5721-5740 (26) | chip close-ups (5711, 5712), tray overviews and bay close-ups, low front views with chip and plinth (5735-5740) |
| S2 | 12:47:22-12:47:30 | 5741-5747 (7) | tray from its left side, steep angles, side wall and coolant fittings; 5747 shows both packages |
| S3 | 13:36:09-13:36:24 | 5748-5758 (11) | tray front-on from left to right, higher and lower |

- Views of the CPO package: 5711, 5712, 5717, 5735-5740, 5747 (about 10; 5711/5712 close-ups at a different
  scale from the tray views). The tray appears in 40+ views, mostly from its front-top side.
- Scene (from contact sheets and 6 close views):
  - Tray: open-top sheet-metal chassis displayed tilted on a black plinth, front panel down (green connector
    blocks left and right, 4 RJ45 jacks, USB, LEDs, pull handles/ears at both ends), long axis pointing up and
    back. Three bays: top bay with a copper-tube serpentine cold plate and a second plate, coolant tubes and
    two quick-disconnect fittings at the top end; middle bay with the switch ASIC area (copper cold block,
    connectors, electrolytics, black plastic baffles); bottom bay with blue fiber bundles, color-coded fiber
    organizers, a row of copper cold-plate fingers with copper tubes, a small black board with an NVIDIA logo
    and a cable connector, and black plastic ducts on both sides. Two stamped sheet-metal crossbars with an
    NVIDIA logo, a braided cable running the full length, white cable clips.
  - Package in front: large CPO package on a post stand, tilted toward the viewer. Silver stiffener ring with
    notched corners, blue substrate, optical-engine sites on four sides (gold-colored blocks: 8 top, 8 bottom,
    about 9 per side), center lid/die with markings "NVIDIA / T TW 2609 / E9X717.001 e1", reflecting the sky.
    An acrylic block labeled "Spectrum-6 CPO" in front of it.
  - Also in frame (not in scope by default): a smaller package without the optical ring on its own stand to
    the left (5738, 5740, 5747), a second tray at the left edge (5713, 5741), acrylic sign "Spectrum-X CPO Switch
    Tray", green NVIDIA backdrop, posters, a bag.
- Lighting: outdoors, direct sun with hard, dappled tree shadows on the tray; the sun moved between S1 and S3
  (64 min), so shadow patterns differ between sessions. Strong sky reflections on the package lid.
- No metric reference in frame (no ruler or mat). No COLMAP model, no 3DGS.
- Tools on this machine: COLMAP 3.11.1 (`/opt/homebrew/bin/colmap`, no CUDA), Blender 4.2.3 LTS. Disk: 23 GiB
  free on the data volume at kickoff.

## 3. What changes versus the CRT run

| CRT run had | Here | Consequence |
|---|---|---|
| COLMAP poses, 137 views, sparse points | nothing; 44 views | camera poses must be solved in this project (Section 6, question 1); fewer views per surface |
| mm ruler on the mat (scale to about 1 percent) | no reference | scale from a known dimension (question 2), flagged as an estimate until cross-checked |
| blue mat: color-keyed masks and a ground plane | cluttered background, black plinth and table | masks need a segmentation model or manual polygons; ground plane from the plinth/table top |
| one object, about 200 mm | a tray of order 0.5-1 m plus a package of order 0.1 m, at very different photo scales | two local frames (tray, package) placed in one scene frame; separate detail floors |
| indoor diffuse light, one session | hard sun, moving tree shadows, three sessions | textures carry shadows; holdout and probe must span sessions; photometric scores will be lower |
| 3DGS baseline | none | report against the photos only; floor reference = flat mean color per view |

## 4. Plan (draft; final after the answers in Section 6)

Rules carried over (PLAYBOOK.md, AGENT_GUIDE.md of the CRT run): copy scripts into this folder (no cross-project
imports); parametric models in TOML, mm in configs, meters and Z up in Blender; one agent per part group, each
owning `scripts/model/<group>/`, `config/model/<group>.toml` and `DevLog/parts/DevLog-002-<group>.md`; measure
first, fit second; code-ranked inspection crops; holdout views never used for measuring, fitting or texturing;
last-known-good copy per group; Blender only through `scripts/bslot.sh`; no git by agents; no emoji;
milestone bundles via `scripts/final_eval.sh`; fresh-context audits after significant work.

### Phase 0: poses, frame, scale, masks (coordinator)
- [ ] Project skeleton: uv env (Python 3.12, same deps), copy and adapt prep/eval/tool/blender scripts; move the
      capture-specific constants listed in the CRT PLAYBOOK section 3 table into `config/`.
- [ ] Camera poses (default, if allowed): own COLMAP SfM on the 44 originals: one shared SIMPLE_RADIAL (or
      OPENCV) camera, exhaustive matching (44 images), mapper; then check registration, reprojection error, and
      whether the package close-ups (5711, 5712) and S2/S3 register into the same model. Masking the tray
      region is not needed for SfM; the static background helps. Fallback for unregistered views: resection
      against the model (PnP on hand-picked correspondences, then refine by edge alignment).
- [ ] Static-scene check across sessions: per-session reprojection residuals on the tray and on the package
      stand; if the package moved, it gets its own pose per session.
- [ ] Scale: from the dimension chosen in question 2, triangulated in several views; independent cross-check
      by a second dimension (for example RJ45 jack pitch or opening, front panel height, published tray
      dimensions). Report scale with an uncertainty range.
- [ ] Frames: scene frame with Z up normal to the plinth/table top. Tray local frame: origin at the front panel
      bottom-left corner, +X along the front panel width, +Y along the tray length, +Z out of the open top
      (the tray tilt is a single transform, so the model is reusable as a level asset). Package local frame:
      origin at the substrate center, Z out of the lid.
- [ ] Undistorted 1/4 and 1/2 scale images; split: about 5 holdout views by farthest-point sampling of view
      directions, constrained to cover all three sessions and at least one package view; about 8 probe views
      for the fast loop, also spanning sessions.
- [ ] Masks (3 levels: object / background / unknown): default a segmentation model (SAM 2 class, run locally
      on the Mac, box prompts from the projected tray and package boxes), then visual check of every mask
      (44 views is few enough to check all). Shadows are not an issue for masks here (no colored mat), but
      the second tray and the plinth touch the silhouette.
- [ ] Edge maps; sparse points in the tray and package frames; environment: plinth and table top as textured
      planes (role of the mat in the CRT run).

### Phase 1A rough block-out, 1B detail and labels, 2 appearance, 3 refine (part agents)
Part groups (draft; each also a 3D region, so agents do not overlap):

| Group | Contents | Detail floor (draft) |
|---|---|---|
| chassis | tray shell, side walls, front panel (ports, green connector blocks, LEDs, handles/ears), crossbars with logo, black plastic ducts and baffles | about 1 mm |
| upper | top and middle bays: serpentine cold plate, second plate, ASIC area parts, capacitors, connectors, coolant fittings and tubes in those bays | about 1 mm |
| lower | bottom bay: fiber bundles and organizers, copper cold-plate fingers and tubes, the small board, braided cable and clips (full length) | about 1 mm; fibers as grouped tubes |
| package | CPO package (stiffener, substrate, optical engines, lid, markings as textures) and its post stand | about 0.2-0.3 mm |

Coordinator: environment (plinth, table), shared tools, evaluation, audits, milestone bundles. Up to 4 part
agents plus one audit slot (5 concurrent maximum), Blender at most 2 concurrent.

Phase budgets as in the CRT playbook (1A about 8 iterations, 1B about 15, 2 about 10-12, 3 about 10); the
iteration loop renders the probe views (about 8) at 1/4 scale.

### Evaluation
Same signals as the CRT run (silhouette IoU on known regions, edge chamfer, sparse points vs surface if SfM
gives them, per-view color-fit PSNR/SSIM plus blur-4 PSNR, error budget by part), minus the 3DGS comparison.
Reference floor: flat mean color per view on the object region. Additional caveat for this capture: sun shadows
differ between sessions, so a texture baked from S1 views cannot match S3 shadows; report holdout results per
session as well as pooled.

### Deliverables
`.blend`, glTF binaries (tray, package, scene), orbit renders, comparison sheets per milestone, README with
methodology and results; demo video per `VIDEO_WORKFLOW.md` of the CRT run only if asked.

## 5. Risks

- Registration: 44 views, glossy metal and copper, repeated structures (fingers, ports, connector blocks). The
  package close-ups may not register with the tray views; S2 side views share few features with S3.
- Scale: no ruler; an error in the reference dimension scales everything (the CRT's "12 cm" was 21 percent off).
- Coverage: the tray underside, the back of the front panel, the package underside, and deep bay interiors are
  barely or never seen.
- Appearance: sun shadows baked into textures; sky reflections on the package lid and polished parts.

## 6. Clarification questions (for Wentao)

1. Camera poses: "no more colmap data" - may I run COLMAP myself on these 44 photos (local, CPU) for poses and
   sparse points? Or should poses come only from the model itself (hand-picked correspondences, PnP and joint
   refinement with the parametric model, Facade style)? Default: own COLMAP run, model-based resection as the
   fallback for views that fail to register.
2. Scale: do you know any real dimension (tray width or length, front panel height, package or stiffener
   size)? Default: research published NVIDIA tray/package dimensions plus standard connector dimensions (RJ45
   jack), cross-check two of them, report as an estimate with a range.
3. Scope: the center tray plus the large CPO package with its post stand; plinth and table top as a textured
   environment like the mat; out of scope: the smaller package to its left, the acrylic signs, the second tray,
   backdrop and posters. Correct?
4. Did anything move between sessions (12:31-12:37, 12:47, 13:36), for example the package stand? Default:
   assume static and verify with per-session residuals.
5. Unseen surfaces (tray bottom, panel back, package underside): closed simple geometry with a flat color,
   marked as unobserved in the model. OK?
6. Limits: same as the CRT run (5 subagents, 2 Blender instances, no remote GPU), and the sources stay read in
   place from `~/Downloads/20261005-nvidia switch` (read-only)? If you would rather keep them safe from Downloads
   cleanup, tell me where to copy them.

### 6.1 Answers (Wentao, 2026-10-05, verbatim)

```text
1. default sounds good to me. Be mindful abt this machine's capability etc., and colmap might also already exist somewhere, avoid installing same thing twice if possible.
2. look up and cross check
3. basic environment (table, ground, background etc.) plus detailed switch tray and chip package. Ignore other packages and poster and signs etc.
4. maybe, environment shadows also moved because of time diff. Make your own judgement.
5. yes. You can hide it with a fake solud stand.
6. same limit. Copy them over to a source folder.
```

Decisions from these answers:
- Poses: own COLMAP run with the existing Homebrew COLMAP 3.11.1 (no new install); CPU threads capped so the
  machine stays usable; model-based resection only for views that fail to register.
- Scale: published dimensions researched by a subagent and cross-checked against standard parts measured in
  the photos.
- Scope: tray and package in detail; environment basic (plinth, table, ground, backdrop/walls as simple
  textured planes or boxes). Out: the smaller package, posters, signs, second tray.
- Sessions: treat geometry as static unless per-session residuals say otherwise; shadows differ per session
  (sun moved), so appearance is evaluated per session as well.
- Unseen surfaces: simple closed geometry, flat color; the package underside hidden by a solid stand.
- Limits: 5 subagents, 2 Blender instances, no remote GPU. Sources copied to `source/images/` in this project
  (gitignored, write-protected copy; the Downloads folder is left untouched).

## 7. Progress log

- 2026-10-05: Read the CRT project (README, PLAYBOOK, AGENT_GUIDE, DevLog-000/001). Inspected the 44 photos
  (EXIF, sessions, two contact sheets, six close views). Created this folder, `.gitignore` (copied from the CRT
  run) and this DevLog with the plan and questions.
