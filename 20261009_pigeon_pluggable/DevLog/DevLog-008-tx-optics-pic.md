# DevLog-008: TX optics + PIC subassembly (workstream T)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v1.1 delivered (audit corrections applied); blend, glb, parts JSON, 8 renders |
| Scope | True-scale TX subassembly for the hero's right eye: carrier, 2 CW DFB + Si microlenses + isolators, SiPh PIC with layout texture, HS differential and DC wirebonds, FAU-attach collimator, 45 deg staircase TIR prism, row-shift rhomboids, 2x4 Galilean eye plate |
| Owns | `scripts/assets/optics_tx/`, `assets/components/optics_tx/`, `outputs/assets/optics_tx/`, `research/tx_optics_refs.md`, this DevLog |
| Interface | DevLog-005 Section 3 |
| Depends on | DevLog-003 (hero TX eye), `research/hero_800g_osfp_dr8.md`, `research/hero_component_dimensions.md`, `research/tx_optics_refs.md` |

## 1. Request (owner, verbatim excerpt, 2026-10-10)

```text
Inspect and model against proper references esp. the 400G and 800G example pluggable teardown pictures. Include textures for the PIC or model more complex stuff on the layout, including electrical routings, wirebonds (low speed and high speed differential), alignment marks, dummy filling etc. Include micro optics components such as silicon lenses between the DFB and the SiPho, and include proper prism for FAU attach and optics needed to map 1x8 output facet to 2x4 fiber array.
```

## 2. Plan

1. References: owner teardown photos (800G PSM8, 400G DR4+, Cisco 400G DR4, Intel-era TOSA) + rate-limited literature -> `research/tx_optics_refs.md`.
2. PIC layout texture generator (Pillow, outside Blender): `scripts/assets/optics_tx/make_pic_texture.py` -> `assets/components/optics_tx/textures/pic_albedo.png`, `pic_mr.png` (G roughness, B metallic). Pad/port coordinates are shared through `tx_geom.py` (no separate layout JSON).
3. Builder (bpy, Blender 4.2): `scripts/assets/optics_tx/build_optics_tx.py` -> `optics_tx.blend`, `optics_tx.glb`, `optics_tx_parts.json` (parts, dims, grades, explode, hooks, lane map, optics design numbers).
4. Renders (EEVEE): `scripts/assets/optics_tx/render_optics_tx.py` -> overview 3/4, exploded (+ labels via `label_overlay_tx.py`), PIC top close-up, DFB-lens-PIC close-up, 1x8 -> 2x4 mapping close-up with beam lines.
5. Inspect every render, iterate at least once; fresh-context audit if budget allows.

Frame (DevLog-005): origin at the centre of the 2x4 eye plate outer surface (dome bases), +X out of the eye (light leaves along +X), +Z up. Mapping to the hero pigeon frame: subassembly +X = pigeon -X (outward, right eye), +Z = +Z, +Y = pigeon -Y (toward the beak). PIC face normal = +X; optical edge faces +Y.

## 3. TODO

- [x] Read DevLog-005, DevLog-003, research docs, current TX eye renders
- [x] Read owner teardown photos (PSM8, DR4+, Cisco DR4, TOSA)
- [x] Literature: Si microlens datasheet, DFB-lens-isolator-SiPh patent, GSSG TW-MZM, edge couplers, FAU attach, glass fan-out, stepped mirrors
- [x] research/tx_optics_refs.md
- [x] PIC texture generator + texture (4200 x 4500 px, 600 px/mm)
- [x] bpy builder: carrier, laser bench, DFBs, lenses, isolators, PIC, pads, wirebonds, output optics, eye plate, hooks, explode, JSON
- [x] GLB export check (59 nodes, 53 meshes, PIC base-color + metal/rough textures embedded, extras incl. explode on 58 nodes)
- [x] Renders (8) + label / tag / note overlay
- [x] Inspect + iterations (5 render passes: exposure, mirrored glyphs, DC field position, mapping camera, outlines, explode layout, callout labels)
- [x] Audit + corrections (Section 6)
- [x] Final report

## 4. What was built (v1, 2026-10-10)

### 4.1 Architecture

- Lasers and output share the PIC front (optical) edge, as read from the owner's 800G PSM8 photo (glass strip between two laser sites). Laser train per DFB (along -Y into the PIC): DFB facet -> 0.194 WD -> Si lens 1 (Focuslight 18-01342 class, collimator) -> 0.10 -> micro isolator (optional; polarizer / latching garnet / polarizer / half-wave plate) -> 0.10 -> Si lens 2 (18-01339 class, focuser) -> 0.30 WD -> laser-side SSC at u = +/-2.5 mm.
- PIC: 2 x (1x2 -> 2x 1x2 MMI tree) -> 8 TW-MZMs (GSSG, RF from the back-edge pads, light co-propagating, on-chip 2 x 50 ohm terminations) -> TiN heaters on both arms -> output MMI -> 2 % tap to a Ge MPD -> cosine S-bend fan-in -> 1x8 SSCs at 0.25 mm pitch; 2 loopback alignment SSC pairs on the 0.25 grid; seal ring opened at the facet; fiducial crosses, circle marks, verniers, edge arrows; dummy fill (6 um squares on 14 um grid with keep-outs); double seal ring; parody marking "ROCKDOVE PHOTONICS PGN-8TX r2 DR8".
- Output optics: FAU-attach fused-silica collimator block (0.80 mm, bonded to the facet, with a PIC edge lid) carrying a 1x12 lens array (8 lanes + loopbacks, R 0.247) -> 45 deg N-BK7 staircase TIR prism (one facet per 0.25 mm lane band, at the lane's eye column) -> window plate -> 6 row-shift rhomboids (+/-0.25, +/-0.5 mm; TX3 and TX6 pass straight) -> 2x4 Galilean eye plate (1.2 mm, M 3.5, concave R 0.161 in, convex R 0.563 out). Beams w 0.050 mm collimated, 0.176-0.204 mm out (z_R 74-100 mm).
- Lane map (TX1..TX8 = edge z -0.875..+0.875): TX1-4 -> bottom row (z -0.375), columns y -1.125..+1.125; TX5-8 -> top row (z +0.375). Per-lane beam paths in `optics_tx_parts.json` (`lane_beam_paths`).

### 4.2 Stack (subassembly X, mm)

eye outer 0.0 / eye inner -1.20 / rhomboids -1.45..-1.25 / window -1.50..-1.45 / staircase -1.85..-1.50 / PIC top = waveguide plane -1.70 / carrier top (pads) -1.75 / PIC bottom -2.45 / bench top -2.05 / carrier cavity floor -2.47 / carrier bottom -3.07. PIC and bench sit in a carrier cavity (Y -9.81..+1.71, Z +/-3.46) so PIC and carrier pads are level. PIC optical edge at Y = -2.25, back edge -9.75; carrier Y -11.35..+1.95, Z +/-4.6.

### 4.3 Parts (dims L x W x H as bbox X x Y x Z in mm; grades A spec, B vendor/paper, C photo, D estimate)

| Part | Size (mm) | Grade | Basis |
|---|---|---|---|
| Carrier (black laminate, cavity for PIC + bench) + RF pair traces + DC traces + vias | 1.32 x 13.3 x 9.2 (0.6 under the cavity) | D | sized to the parts; RF S pairs pulled to coupled lines |
| SiPh PIC (textured) | 0.75 x 7.5 x 7.0 | B (7.0 x 7.5 Carmel8) / D (t) | research |
| PIC pads | 25 RF (G S S G ..., 0.27 pitch, shared G) + 28 DC (14 per side) | D / C (3 bonded edges in photo) | |
| HS differential wire bonds | 16 S + 9 G x 2 = 34, ~0.4 mm span, pads level, 25 um Au | D | GSSG practice; Cisco photo paired traces (C) |
| DC wire bonds | 28 (heaters, MPD anodes, GND, MPD bias) | D | |
| Laser bias bonds | per laser 4 die->CoS + 4 CoS->bench + 4 bench->carrier | C/D | Cisco photo ~5 parallel bonds |
| AlN optical bench | 0.42 x 3.87 x 6.8 | D | |
| CW DFB die x2 | 1.0 x 0.3 x 0.1 (p-up) | B/D | research (Casela class) |
| AlN CoS submount x2 | 1.4 x 0.8 x 0.25 | D | |
| Si lens 1 (collimator) x2 | L 1.2 (Z) x W 0.6 (X) x T 0.5 (Y, beam), CA 0.5, EFL 0.34, WD 0.194 | A | Focuslight 18-01342 datasheet |
| Si lens 2 (focuser) x2 | L 1.0 (Z) x W 0.6 (X) x T 0.5 (Y), CA 0.5, EFL 0.45, WD 0.30 | B | Focuslight 18-01339 (search summary) |
| Micro isolator x2 (optional) | 0.6 x 0.6 x 0.55 (pol 0.10 / garnet 0.25 / pol 0.10 / half-wave plate 0.10) | C/D | Cisco 400G DR4 photo; US 9,787,054; none documented for DR8 |
| FAU-attach collimator block + 1x12 lens array | 0.63 x 0.80 x 3.2; lenslets R 0.247, pitch 0.25 | D | design calc |
| PIC edge lid | 0.3 x 0.6 x 3.2 | B/D | US 9,787,054 silica cover lid |
| Staircase TIR prism (N-BK7 class) | 0.35 x 2.65 x 2.0, 8 bands | D | stepped-mirror principle (fact 15) |
| Staircase pedestal, window plate, 4 eye-plate posts | - | D | mounting |
| Row-shift rhomboids x6 (N-BK7 class) | aperture 0.2, shifts 0.25 / 0.5 | D | |
| 2x4 eye plate + output lenslets | 1.2 x 3.2 x 2.6; pitch 0.75; apertures 0.22 in / 0.70 out | D | design calc; hero lenslet pitch 0.75 (D) |
| Beam cores x8 (per-lane colour) + laser cores + envelopes | viz only, collection `tx_beams_viz` | viz | |

Total 37.4k tris, 55 mesh objects.

### 4.4 Hooks (mm, subassembly frame; local +X of each empty = direction; local +Z as close to world +Z as possible, world +Y for vertical hooks)

| Hook | Location | Direction |
|---|---|---|
| HOOK_eye_aperture | (0, 0, 0) | +X |
| HOOK_pic_mount | (-2.45, -6.00, 0) | -X |
| HOOK_flex_rf | (-1.75, -11.35, 0) | -Y |
| HOOK_flex_dc | (-1.75, -7.405, -4.6) | -Z |
| HOOK_carrier_mount | (-3.07, -4.698, 0) | -X |

### 4.5 Deliverables

- `assets/components/optics_tx/optics_tx.blend`, `optics_tx.glb`, `optics_tx_parts.json`, `textures/pic_albedo.png`, `textures/pic_mr.png`, `textures/pic_layout_preview.png`
- `outputs/assets/optics_tx/`: `optics_tx_overview.png`, `optics_tx_exploded_labeled.png`, `optics_tx_pic_top.png`, `optics_tx_pic_zoom_mzm.png`, `optics_tx_pic_bonds.png`, `optics_tx_laser_coupling.png`, `optics_tx_mapping_1x8_2x4.png`, `optics_tx_mapping_oblique.png`; raw renders + label JSON in `work/`.

### 4.6 Rebuild

```text
uv run --with pillow --with numpy python scripts/assets/optics_tx/make_pic_texture.py assets/components/optics_tx/textures
scripts/bslot.sh -b --factory-startup --python scripts/assets/optics_tx/build_optics_tx.py -- assets/components/optics_tx
scripts/bslot.sh -b assets/components/optics_tx/optics_tx.blend --python scripts/assets/optics_tx/render_optics_tx.py -- outputs/assets/optics_tx
bash scripts/assets/optics_tx/compose_optics_tx.sh
```

### 4.7 Notes for integrators (H, S)

- Hero mapping: subassembly +X -> pigeon -X (outward from the right eye), +Y -> pigeon -Y (beak), +Z -> +Z. The hero's current PIC plane sits ~5.1 mm behind the eye surface; this subassembly puts it 1.70 mm behind (needed for the collimated rearranger); the carrier extends 13.3 mm toward the tail and 4.6 mm up/down.
- Glyphs on the PIC texture are mirrored in image space so they read correctly on the model (viewed from +X with the optical edge up, screen right = -Z).
- Beams: hide `tx_beams_viz` in background shots; beam shots carry the 1311 nm note.

## 6. Fresh-context audit (2026-10-10) and corrections

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | PIC HS and DC bonds passed 0.09-0.22 mm through the die (PIC top 0.77 mm above the carrier pads); bonds ~0.94 mm long vs "~0.45" in docs | high | PIC and bench now sit in a carrier cavity; carrier pads 0.05 mm below the PIC top; bonds ~0.4 mm span; bench-to-carrier bonds looped over the cavity wall; build-time clearance check added (`BOND_CLEARANCE_OK bonds=86`) |
| 2 | No driver EIC (Cisco 400G DR4 shows a flip-chip EIC) | high | Not added: research claim 2 locks DSP-integrated direct drive for this DR8 (BCM87800, Keystone); documented in research 3.3 |
| 3 | Isolator without half-wave plate sends 45 deg polarization to a TE SSC | med | 0.10 mm half-wave plate added after the output polarizer |
| 4 | Seal ring crossed the edge couplers | med | ring opened across the facet zone (u +/-2.85) |
| 5 | Splitter tree and laser feed under the RF feed metal | med | documented as a layout simplification (research 3.3) |
| 6 | Loopbacks off the 0.25 grid and outside the collimator | med | moved to +/-1.125 / +/-1.375; collimator lens array widened to 1x12; -Y eye posts moved clear |
| 7 | PIC size vs PSM8 photo; no big clear lid with X/V fiducials; photo bond rows denser | med | size choice justified in research 3.3; lid and denser rows left as open issues |
| 8 | DC run m=13 overlapped the laser feed trench | med | laser channel moved to |u| 3.04 |
| 9 | Doc vs code: rhomboid aperture wording, eye output w, DFB w0 basis, RF pitch, lens bbox axes, missing pic_layout.json, true-scale exception for drawn waveguide widths | med | all corrected (research 3.2/3.3, this DevLog, JSON scale statement) |
| 10 | Laser-side SSC facet normal, no angle/AR | low | open issue (8 deg angled facet + AR not modeled) |
| 11 | Staircase TIR margin 1.3 deg in fused silica | low | staircase changed to N-BK7 class (3.3 deg margin) |
| 12 | Render tags collided; exploded leaders ambiguous; laser bonds hang in explode | low | tag collision avoidance added; callout columns; bonds explode with the submount (left as is) |
| 13 | Dummy-fill corner check only on the anchor pixel | low | all four corners checked |
| 14 | macOS system font path | low | kept (build tool, not committed data) |
| 15 | OPTICS_TX_ROOT lacked `explode` | low | added [0, 0, 0] |

Verified by the auditor (recomputed): collimator R and beam sizes, Galilean R_in/R_out/M, laser-train WD/EFL consistency with the datasheets, staircase facet orientation and band containment, rhomboid displacement and no Z overlaps, lane map consistency, all bond landings on pads, hooks/axes/GLB contents, no emoji / home paths, beam note on all beam renders.

## 7. Open issues

1. Large clear cover lid / encapsulant with etched X/V circle fiducials (PSM8 photo 031929_1) not modeled.
2. Laser-side SSC facet drawn normal-incidence; real designs use an 8-10 deg angled facet and/or AR.
3. Splitter tree routed under RF feed metal (simplification).
4. PIC 7.0 x 7.5 (Carmel8) differs from the PSM8 photo PIC (~8.3 x 4.4).
5. Mapping 1x8_2x4 view: the carrier front wall hides the lower staircase; the oblique view shows it fully.
6. Hero integration: this subassembly puts the PIC 1.70 mm behind the eye surface and needs ~13.3 x 9.2 x 4.3 mm of head volume (hero currently 5.1 mm deep PIC placement).

## 8. v3 FX layer (DevLog-018 items T-a, T-b; 2026-10-10)

Owner text (verbatim, DevLog-018): "add fast and traveling glowing effect to the mzi to show the electrical signal coming in, they should come in from the CB contacts and thru the wirebonds and then onto the traveling wave electrodes. Use different color than the optical beam color" / "the PIC circuits should also light up accordingly, e.g., DFB turns on as camera moves in, light propagates follwoing camera panning left, and light up the circuits and MMI tree accordingly, and thru and coming out of the MZI modulated."

### 8.1 What it is

- Module `scripts/assets/optics_tx/tx_fx.py` (Blender). It builds 4 FX objects (collection `tx_fx`) under any parent, so it works in `optics_tx.blend` and after the scene's GLB import (parent = the imported `OPTICS_TX_ROOT`; macro-set scale x1000 is inherited):
  - `fx_tx_optical_pic`: emission ribbons 12 um wide, 4 um above the PIC, on every lit waveguide plus the MMIs (loopbacks stay dark).
  - `fx_tx_optical_free_space`: tubes along the two laser trains (DFB -> Si lens 1 -> isolator -> Si lens 2 -> SSC) and the 8 output paths (collimator -> staircase -> rhomboids -> eye -> 1 mm out).
  - `fx_tx_electrical_rf`: per lane, both signal lines: flex contact -> carrier coupled pair -> GSSG wire bond -> PIC pad -> feed -> traveling-wave electrode -> termination.
  - `fx_tx_dfb_on`: facet glows.
- Every vertex carries `fx_dist` (physical path length in mm from the DFB facet, or from the flex contact for RF), `fx_lane`, and `fx_mod`. The on-chip distances come from a breadth-first walk over the `tx_geom` waveguide network starting at the laser SSCs.
- Each material uses emission x transparency. The keyed node values are `front_mm`, `master`, `mod_on`, `rate`, `seed`, `sym_per_mm`; time comes from a `frame / fps` driver. There is no per-frame Python and no heavy geometry (ribbons and 6-sided tubes only).
- PAM4: 4-level symbols from 3D white noise of (symbol index, lane, seed), travelling along the path. RF uses 55 % duty dashes plus a dim base glow.
- Colours: optical = TX red (1.0, 0.08, 0.05), matching `scripts/vfx/materials.py` TX_RED. Electrical = amber (1.0, 0.60, 0.13).
- Base-asset emitters follow the light-up when present (also after a GLB import): the `tx_dfb_facet_glow` emission keys on with the DFB, and the `tx_eye_lens_red` emission keys on when the front reaches the eye.

### 8.2 API (absolute frames)

```text
import sys; sys.path.insert(0, "scripts/assets/optics_tx"); import tx_fx
fx = tx_fx.build_tx_fx(parent)                       # parent = OPTICS_TX_ROOT (blend or imported GLB)
tx_fx.tx_light_up(fx, f_on, f_front_end, f_mod_on=None, dfb_ramp=4, f_off=None)
tx_fx.tx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6)
track = tx_fx.tx_front_track(fx, f_on, f_front_end, world=True)   # [(frame, world xyz)] of the optical front
```

- `tx_light_up` keys the DFB on at f_on. The front then travels 0 -> 26.06 mm, linear between f_on and f_front_end. The MZM outputs start blinking at f_mod_on, which defaults to the moment the front passes the output MMIs (fraction 15.30 / 26.06).
- `tx_signal_glow` sends an arrival front over 5.05 mm (flex -> termination) in `arrive_frames`, then travelling pulses at speed = rate / sym_per_mm mm/s (13.3 mm/s of subassembly length at the defaults), and fades out at f1.
- `tx_front_track` returns a smoothed path (Z centred) that the scene camera can follow. Order of travel:
  1. DFB facets (Y ~ 0, Z +/-2.5)
  2. lenses / isolator
  3. PIC front edge (Y -2.25)
  4. side feeds to the back (Y ~ -9.3)
  5. MMI trees
  6. MZM arms travelling toward +Y
  7. fan-in to the 1x8 SSCs (Y -2.25)
  8. staircase / rhomboids
  9. eye (X 0)

  With the hero's right eye seen from outside, this is the "pan" order.

### 8.3 Path distances (mm, from `optics_tx_fx.json`)

| Point | Distance |
|---|---|
| DFB facet -> PIC laser SSC | 2.24 |
| -> MZM output MMIs | 15.30 |
| -> output SSCs (longest lane) | 19.49 |
| -> eye plate (first lane) | 20.84 |
| total incl. 1 mm out | 26.06 |
| RF flex contact -> termination | 5.05 |

### 8.4 Test timing used for the strips (30 fps, 150 frames)

| Event | Frames |
|---|---|
| DFB on | f10 |
| Optical front | f10 -> f80 |
| MZM outputs modulated from | f51 |
| Eye lit at | f66 |
| RF glow | f40 -> f150 (arrival over 12 frames, 40 symbols/s, seed 7) |

The light-up camera follows `tx_front_track` from +X with the optical edge up.

### 8.5 Deliverables

- `assets/components/optics_tx/optics_tx_fx.blend`: `optics_tx.blend` + FX + the test keys; test cameras and lights removed. `optics_tx.blend`, `.glb` and `_parts.json` are unchanged.
- `assets/components/optics_tx/optics_tx_fx.json`: API, colours, distances, test timing, front track.
- Strips (540x674 frames):
  - `outputs/assets/optics_tx/optics_tx_fx_strip_lightup.png`: f9-f110.
  - `outputs/assets/optics_tx/optics_tx_fx_strip_signal.png`: RF close-up, f41-f100.
- Scripts:
  - `scripts/assets/optics_tx/make_tx_fx_test.py`: build + test render.
  - `scripts/assets/optics_tx/compose_tx_fx_strip.py`
- Run commands:

```text
scripts/bslot.sh -b assets/components/optics_tx/optics_tx.blend --python scripts/assets/optics_tx/make_tx_fx_test.py -- assets/components/optics_tx outputs/assets/optics_tx/fx_test
uv run --with pillow python scripts/assets/optics_tx/compose_tx_fx_strip.py outputs/assets/optics_tx/fx_test outputs/assets/optics_tx
```

### 8.6 Notes / open

- The red tint of the eye lenslets is their base colour, not emission, so the lenslets read faintly red even before light-up.
- Scene usage: call `build_tx_fx` after the GLB import with the imported `OPTICS_TX_ROOT` as parent, then key it with scene frames. Hide `tx_beam_*` while the FX run.

## 5. Progress log

- 2026-10-10: references read; research doc written; architecture fixed (lasers + output share the PIC front edge per the PSM8 photo; collimate-then-rearrange free-space output optics).
- 2026-10-10 01:46: tx_geom.py (shared layout + stack) and make_pic_texture.py; texture v1 inspected (tree, MZM/heater/MPD region, SSC region crops).
- 2026-10-10 01:52: build v1 OK (35 parts); first renders overexposed; found PIC texture glyphs would read mirrored from the face side -> mirrored glyphs in image space.
- 2026-10-10 01:55: lights toned down; +Z DC field moved inside the carrier (was past the back edge); gold roughness up.
- 2026-10-10 01:57: per-lane beam colours; pedestal glass, rails -> 4 slim posts; lasers split per LD with outward explode; mapping cameras moved off the occluding rail.
- 2026-10-10 02:02: render-only wireframe outlines for glass optics; camera fitting; lane tags; beam note; all 8 views rendered and composed.
- 2026-10-10 02:06: explode layout reworked (Y-spread laser train, X-stacked optics column), callout-column labels with auto-crop; exploded view re-rendered.
- 2026-10-10 02:20: fresh-context audit (15 findings); corrections applied: carrier cavity + level pads + bond clearance check, isolator half-wave plate, seal-ring facet opening, on-grid loopbacks + 1x12 lens array, laser channel shift, N-BK7 staircase, fill corner check, root explode, tag collision avoidance, doc fixes. Rebuilt (55 parts, 37.4k tris), all 8 views re-rendered and inspected; laser camera re-aimed over the cavity wall.
- 2026-10-10 02:40: integration fix (coordinator): hook empties re-oriented so local +X = documented direction (was local +Z; HOOK_eye_aperture local X lay along +Y, flex hooks along +X). Positions unchanged; build now asserts each hook's local +X and location (`HOOK_AXIS_OK 5`); parts JSON hooks carry `local_axis: +X`. Rebuilt .blend/.glb/_parts.json; part centres and bboxes diffed against the previous JSON: no part moved. GLB hook axes verified after the Y-up conversion.
- 2026-10-10 03:10: v3 FX layer (tx_fx.py): optical light-up + amber RF signal glow, keyed API, front track; optics_tx_fx.blend/.json; two test strips at 540x674 rendered and inspected (2 passes: distance fix for f_mod_on, smoothed Y-only front track, amber strength down, deeper optical PAM4 contrast). Raw test frames deleted after composing.
