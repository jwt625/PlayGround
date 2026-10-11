# DevLog-009: RX optics + TIA subassembly (workstream X)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v4: changing-bits electrical VFX (DevLog-020, Section 9); FX layer Section 8; open decisions Section 6 |
| Scope | True-scale RX subassembly for the hero's left (RX) eye: 2x4 lenslet array -> 2x4-to-1x8 fan-in micro-optics -> 1x8 PIN PD array -> GSG wirebonds -> octal TIA (spiral inductors, pads, decoupling) -> GSSG output bonds -> carrier -> flex pads |
| Owns | `scripts/assets/optics_rx/`, `assets/components/optics_rx/`, `outputs/assets/optics_rx/`, `research/rx_optics_refs.md`, this DevLog |
| Interface | DevLog-005 Section 3 (origin = centre of the 2x4 lenslet array outer surface, +X out of the eye, light arrives travelling -X, +Z up; hooks; `explode` property on every part) |
| Authors | Claude (workstream X subagent), at Wentao Jiang's request |

## 1. Request (verbatim excerpts, 2026-10-10, from DevLog-005)

```text
the Rx should have proper micro optics for mapping the 2x4 array to 1x8 PD array, proper high speed differential wirebonds to a multi-channel TIA chip, and proper inductor coil etc. on the TIA chip. Find good ref pic to build it or use as texture.
Inspect and model against proper references esp. the 400G and 800G example pluggable teardown pictures.
```

## 2. Plan (written before implementation)

1. References: owner teardown photos (OFS-64 Innolight 400G DR4+ / 800G PSM8 RX blocks; 100G post TIA close-ups), PD array datasheet with mechanical drawing, TIA die micrographs with inductors, octal TIA facts. Record in `research/rx_optics_refs.md`.
2. Optical design (physics exact, design grade D): rectangular 2x4 lenslet grid (column pitch 1.00 mm, row pitch 0.75 mm) -> two tilted silicon plates (one per row, +/-30 deg, thickness solved for a +/-0.25 mm lateral shift) interleave the rows in Y -> interleaved 45 deg comb mirror (step mirror; top-row teeth and bottom-row teeth at the same X, alternating in Y) turns all 8 beams down onto one row -> top-illuminated 1x8 PIN PD array at 0.50 mm pitch (real part class: 1x8, 500 um pitch, 300 x 4000 x 150 um die, 20 um aperture). Row-specific lenslet radii equalise the two path lengths. Beam lines traced exactly (Snell at the plates, mirror reflection, marginal rays solved to land on the PD centre).
3. Electrical: PD GSG pads (N-P-N, 85 um spacing per datasheet drawing) -> 3 short ball-wedge bonds per lane to TIA input pads (VPD-IN-VPD); TIA outputs GSSG (G, OUTP, OUTN, G) -> differential CPW on the AlN carrier -> flex pad row; DC/control pads on the TIA ends (VCC, GND, VPD; SCL, SDA, RSTn) via single-layer decoupling caps to carrier DC pads -> DC flex pads.
4. TIA die: 4.6 x 1.3 x 0.20 mm (estimate), 8 lanes at 0.50 mm pitch; per lane: input pads, ESD, PD-bias RC filter, input series inductor, TIA core, two shunt-peaking pairs, output buffer with series inductors, output pads; top-metal supply rails, dummy fill with inductor keep-outs, seal ring, alignment marks, parody logo. Fine layout as procedural textures (base color, height, roughness) from a layout JSON; the same JSON drives 3D spiral-inductor ribbons and pad geometry so texture and geometry match.
5. Build `scripts/assets/optics_rx/build_optics_rx.py` (bpy, meters) -> `.blend`, `.glb`, `optics_rx_parts.json`. Render script: overview 3/4, exploded + labels, TIA close-up, PD-to-TIA bond close-up, optics close-up with traced beams and the invisible-beam note. Inspect each render; iterate at least once; fresh-context audit.

## 3. TODO

- [x] References collected and graded (`research/rx_optics_refs.md`)
- [x] TIA + PD layout/texture generator (`gen_rx_textures.py`)
- [x] bpy build script with hooks, explode props, parts JSON, GLB
- [x] Render script + label overlay
- [x] Renders: overview, exploded labeled, TIA close-up, bond close-up, optics close-up with beams (+ orthographic top view)
- [x] Iteration pass after inspecting renders (4 passes, see log)
- [x] Fresh-context audit and corrections (Section 6)
- [ ] Final report (paths, hooks, parts with dimensions and grades, open issues)

## 4. Deliverables (v2)

| Path | Content |
|---|---|
| `assets/components/optics_rx/optics_rx.blend` | true-scale subassembly (textures packed); collections `rx_micro_optics`, `rx_pd`, `rx_tia`, `rx_carrier`, `rx_wirebonds`, `rx_passives`, `rx_hooks`, `viz_beams` (traced chief + marginal rays, viz only) |
| `assets/components/optics_rx/optics_rx.glb` | same without `viz_beams`; Y-up glTF; node extras carry `explode` (Blender Z-up, m), `explode_gltf` (glTF Y-up, m), `part_label`, `grade`, hook `doc` |
| `assets/components/optics_rx/optics_rx_parts.json` | parts (dims, grade, source, explode), hooks (location, rotation, doc), bbox, optics numbers, lane map, build log |
| `assets/components/optics_rx/textures/` | `tia_basecolor/height/rough.png` (7360 x 2080, 1.6 px/um), `pd_basecolor/height.png` (8000 x 600, 2 px/um), `rx_layout.json` |
| `outputs/assets/optics_rx/optics_rx_{overview,exploded,tia_closeup,bond_closeup,optics_closeup,optics_top}.png` | renders (Cycles 64 spp, 1920 x 1200); raw frames + label anchors in `work/` |
| `scripts/assets/optics_rx/` | `gen_rx_textures.py` (uv), `build_optics_rx.py` (bpy), `render_optics_rx.py` (bpy), `finalize_renders.py` (uv) |

Commands (project folder):

```bash
uv run --with pillow python scripts/assets/optics_rx/gen_rx_textures.py
scripts/bslot.sh -b --factory-startup --python scripts/assets/optics_rx/build_optics_rx.py
scripts/bslot.sh -b assets/components/optics_rx/optics_rx.blend --python scripts/assets/optics_rx/render_optics_rx.py   # [-- shot ...]
uv run --with pillow python scripts/assets/optics_rx/finalize_renders.py
```

Scale statement: everything is true scale in metres (parts in mm x 1e-3). Bounding box X -6.95..+0.035, Y -3.20..+3.20, Z -1.83..+0.85 mm; only the lens domes (sag 0.031/0.035) sit ahead of the X = 0 eye plane. 31.8k polygons in the .blend including `viz_beams`; 62.2k triangles in the GLB (no beams), 3.5 MB. The render script scales the root x1000 at render time only (EEVEE/Cycles numerics); the saved asset is untouched.

Hooks (subassembly frame): `HOOK_eye_aperture` (0,0,0), local +X out of the eye; `HOOK_pd_mount` (-3.90, 0, -1.20) PD aperture row, beams arrive -Z, carries the PD's explode offset; `HOOK_flex_rf` (-6.70, 0, -1.447) centre of the 8 x GSSG pad row, local +X = flex exit (global -X); `HOOK_flex_dc` (-6.70, -2.62, -1.447) power group GND/VPD_BIAS/VTIA; `HOOK_flex_ctrl` (-6.70, +2.62, -1.447) RSTn/SDA/SCL (extra hook beyond DevLog-005); `HOOK_carrier_mount` (-3.475, 0, -1.83) carrier bottom, mount normal -Z.

Interface notes for H and S: lenslet grid is 1.00 (Y) x 0.75 (Z) mm pitch, not the hero's 0.75 x 0.75 (the column pitch must be 2 x the 0.50 mm PD pitch for the interleaver); the eye block is 4.4 x 1.7 mm (fits a 1.85 mm iris radius only if the iris is enlarged to ~2.4 mm or the lens substrate is cropped). Hero RX flex was 23 traces single-ended; this carrier exposes 8 x GSSG (32 pads) + 6 DC pads.

## 5. Progress log

Times are taken from the timestamped logs in `outputs/logs/rx_*.log` (corrected per audit item 12; earlier entries had estimated times).

- 2026-10-10 ~01:20-01:35: read DevLog-005/003 and hero research; inspected owner photos (OFS-64 20250913_030659_0..3, 031929_0..2; 100G post 20250122_045451_2/3, 20250127_000458_1); found the PD-array datasheet and TIA micrograph references; plan written.
- 01:38-01:41 (rx_textures_013834..014140): texture generator v1 (56 octagonal spirals, 66 TIA pads; PD layout from the datasheet drawing); densified circuit blocks, coil leads, lane numbers.
- 01:45 (rx_build_014526): build v1; plate t = 0.6664 mm; all chief rays land on their PD centres; DC routing redone without crossings (-Y TIA pad order VCC, GND, VCC, GND, VPD).
- 01:46-01:48 (rx_render_014634..014846): render pass 1 (EEVEE): optics close-up smeared by refraction, overview hidden by the comb spine, bond close-up too tight -> alpha glass + translucent Si for beam shots, new cameras, smooth domes, copper coils, smoother bond landing.
- 01:50-01:58 (rx_build_015037..015307, rx_render_015038..015818): orthographic top view + 1 mm scale bar; die text found mirrored (die UV is a mirror of the top view) -> text tiles mirrored; exploded regrouped.
- 01:59-02:00 (rx_render_015902..020050): EEVEE jagged edges at mm scale not fixed by shadow resolution, probe tracing or x1000 scaling -> all shots in Cycles (64 spp + denoise).
- 02:02-02:03 (rx_build_020256, rx_render_020258): carrier trimmed to the X = 0 eye plane; .blend backups disabled. v1 complete.
- 02:05-02:20: fresh-context audit (read-only subagent), 19 findings (Section 6).
- 02:28-02:43 (rx_textures_022811, rx_build_022812..023837, rx_render_022856..024305): audit corrections applied (Section 6); turning block rebuilt as a watertight solid (0 non-manifold edges, asserted in the build); renders re-checked.

## 6. Audit (2026-10-10) and corrections

| # | Finding (severity) | Action |
|---|---|---|
| 1 | 0201 caps shorted VPD_BIAS to GND / both VTIA terminals on one net (HIGH) | one 0201 on VTIA, long axis along Y, terminal A on the VTIA bus, terminal B on a via-stitched GND land; VPD keeps its SLC only |
| 2 | SDA trace through a die-site fiducial; VTIA bus touching another (MED) | die-site fiducials moved to X = -3.60, |Y| = 2.95 |
| 3 | Output bonds did not read as differential; PD input is GSG, not differential (MED) | OUTP/OUTN pulled to 0.10 mm centre-to-centre (G at +/-0.19), CPW S 45 um at 0.10 mm; GSG input kept (owner's 100G photo) and listed as an open decision below |
| 4 | GLB explode extras in Blender frame (MED) | added `explode_gltf` (Y-up) next to `explode`; frames stated in the parts JSON |
| 5 | Au comb on a grey Si spine did not match the glass turning blocks in the teardowns (MED) | replaced by a stepped fused-silica turning block with Au-coated 45 deg facets (coating required: TIR would leak); lens radii recomputed for the glass path |
| 6 | Exploded labels crowded / anchors on hidden parts / lens label covering the part (MED) | electrical stack explodes 1.6 mm, turning block 4.2 mm; anchors ray-tested (hidden parts skipped); shim/submount labels dropped; label placed beside the part when the column would cover it |
| 7 | S2 citation partly unsupported (MED/LOW) | re-fetched: RSSI is on the GN1814 page (not GN1825); "wire-bondable bare die" quoted from the GN1814 release |
| 8 | PD pad rows maybe inverted vs Detail A (LOW) | drawing re-read: ambiguous; kept and recorded as open |
| 9-11 | focus "<<" overstated; bond span; "unlit 0.25 mm gap" (LOW) | replaced with computed numbers in research doc |
| 12 | progress-log times not real (LOW) | log rewritten from log-file timestamps |
| 13 | face count ambiguous (LOW) | both counts stated |
| 14 | render docstring said EEVEE (LOW) | fixed |
| 15 | render artifacts (LOW) | Si plate height 0.80 so the bottom plate sits on the bench; turning block replaces the blown-out teeth; lens array still renders dark from behind in the overview (Cycles glass against a dark world) - not fixed |
| 16 | VCC rails and GND guards same colour, read as shorts (LOW) | guards drawn in a lower-metal colour and passing under the redrawn top-metal rails |
| 17 | TIA GND pad unconnected; hooks without explode (LOW) | GND pad bridged to both GND strips with vias; HOOK_pd_mount follows the PD explode |
| 18 | ~5 bonds per lane in the 400G photo vs 3; one bond per SLC side (LOW) | kept (owner's 100G note supports GSG); noted |
| 19 | macOS font paths (LOW) | kept (machine-local tool) |

Open owner decisions:

1. (Decided 2026-10-10, see Section 7) PD -> TIA input is differential.
2. Lenslet pitch 1.00 x 0.75 mm vs the hero's 0.75 x 0.75 eye (H must enlarge the iris to ~2.4 mm radius or crop the lens substrate).
3. Octal wire-bonded TIA vs the real flip-chip GN1825 class (built wire-bonded per the request).

## 7. v3: differential PD -> TIA input (2026-10-10)

Coordinator decision (relayed, following the owner's request "proper high speed differential wirebonds to a multi-channel TIA chip"): per lane, PD anode (P) -> TIA IN+ and PD cathode (N) -> TIA IN- as a paired G-S-S-G group with flanking ground bonds, short and matched; PD bias moves to the TIA side; hooks and part positions unchanged; beam-landing assertion kept.

Changes:
- PD texture/layout: per lane P pad at lane centre -0.05 mm, N pad at +0.05 mm (65 um, 0.235 mm from the optical edge); P stem from the p-ring, N arm from an n-contact arc with a gap where the P stem leaves. 16 PD pads (was 24). Aperture position unchanged.
- TIA texture/layout: input row G (-0.17), IN+ (-0.05), IN- (+0.05), G (+0.17); two input series inductors per lane (dy -/+0.10, 72 um) instead of one; PD-bias RC filter now feeds IN-; input grounds tied to the lane guards. 74 TIA pads, 64 inductors (8 per lane).
- Geometry: PD submount extended toward the TIA to 5 um from the TIA shim (X0 -4.12 -> -4.155) with a new Au ground strip `rx_pd_submount_gnd_strip` (X -4.155..-4.085, 3 um) landing the ground bonds. Bond object renamed `rx_wirebonds_pd_to_tia_diff`: 32 wires (16 signal, equal 0.21 mm spans; 16 ground, TIA G pad -> strip). No other part moved; all 6 hook positions verified identical in the parts JSON.
- Docs: PD bias now via the TIA (IN-), noted in `research/rx_optics_refs.md`.
- Cosmetic: overview draws the lenslet array clear (alpha) instead of the black refractive slab; Cycles bounces raised to 24; exploded-label leaders untangled by swapping crossing label slots; bond close-up caption updated.
- Verified: build log shows all 8 lanes landing on their PD centres (error < 0.1 um), turning block watertight; GLB 68.2k triangles, 3.55 MB, `explode`/`explode_gltf` on every mesh node; re-rendered all six shots (rx_render_20261010_025117, overview rx_render_20261010_025312) and inspected the bond close-up, exploded view and overview.

## 8. v3 wave FX layer (DevLog-018 item X, 2026-10-10)

Owner text (verbatim, DevLog-018): "~0:35, cleanup the black fixtures so that the lens and light path is more visible. And add glowing waves on the PD and TIA and wirebonds and PCBs to show electrical signals flowing."

Cause of the "black fixtures": in the film's EEVEE draft, the fused-silica parts (bench, plate shelf + posts, turning-block body) and the Au facets render as dark slabs (refractive glass and mirrors reflecting a dark world). Geometry and mechanics are unchanged; only the look is changed, through a scene-callable function.

### 8.1 What it is

- Module `scripts/assets/optics_rx/rx_fx.py`, which follows the pattern of T's `tx_fx.py`. The mesh and material code is copied from T's module rather than imported, so the two layers stay independent.
  - `rx_clean_fixtures(parent, si_plates=True, lenslets=False, facets=True)` swaps materials on matching objects only; it moves no geometry. Name prefixes are used, so it works on a GLB import.
    - Bench, shelf/posts and turning-block body: near-clear ghost glass (alpha 0.05) with a facing-ratio edge glow, and no shadow casting.
    - Au facets: translucent gold (alpha 0.35).
    - Si plates: translucent violet-grey with lit edges.
    - Lenslets, optional: clear with lit edges.
    - It returns the touched objects, so the scene can also hide them.
  - `build_rx_fx(parent)` builds three emission objects in collection `rx_fx`:
    - `fx_rx_optical_beams`: a faint, unmodulated converging cone, radius 0.30 mm at the lens shrinking linearly to the focus.
    - `fx_rx_optical_cores_pd_spots`: a thin bright PAM4 core, plus a spot and halo on each PD aperture that light as the front lands.
    - `fx_rx_electrical_signal` (amber), per lane, P and N lines:
      1. PD aperture to pad
      2. differential bond to IN+ / IN-
      3. on-die route past the input, peaking 1, peaking 2 and output spirals; each spiral flashes as the pulse passes
      4. OUTP / OUTN bond
      5. carrier CPW
      6. flex pad
  - The optical paths are re-traced in the module with the build constants and asserted to land on the PD apertures (< 1 um). Bond and pad positions come from `textures/rx_layout.json`.
- Colours: optical = RX cyan (0.1, 0.85, 1.0) (`scripts/vfx/materials.py` RX_CYAN); electrical = amber (1.0, 0.60, 0.13), the same as T.

### 8.2 API (absolute frames)

```text
import sys; sys.path.insert(0, "scripts/assets/optics_rx"); import rx_fx
touched = rx_fx.rx_clean_fixtures(root)                     # root = RX_OPTICS_ROOT (blend, or the imported GLB root)
fx = rx_fx.build_rx_fx(root)
rx_fx.rx_light_arrive(fx, f_on, f_front_end, f_off=None, rate=30.0, seed=7)
rx_fx.rx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6)
track = rx_fx.rx_front_track(fx, f_on, f_front_end, f_sig0=None, f_sig_end=None, world=True)
```

- `rx_light_arrive`: the front travels from 3 mm in front of the eye to the PD, linearly in path length.
  - The light is already PAM4-modulated on arrival (it comes from the far-end TX).
  - Bottom-row lanes land at 7.77 mm and top-row lanes at 8.52 mm. The lens is at 2.97 mm.
- `rx_signal_glow`: an arrival front covers 3.20 mm (PD aperture to flex pad) within `arrive_frames`. After that, PAM4 dashes travel at rate / sym_per_mm mm/s.
- `rx_front_track`: returns the optical front (eye to PD) and then, optionally, the electrical front (PD to TIA to CPW to flex), for the camera.
- Scene note for S: the film currently adds its own straight `rx_in` / `rx_int` / `rx_down` beams (red, not traced). The FX beams replace them with the traced paths in RX cyan. The film's lenslet override can stay, or `lenslets=True` can be used instead.

### 8.3 Test (30 fps, 150 frames, EEVEE Next, macro scale x1000, 540x674)

| Event | Frames |
|---|---|
| Light on | f10 |
| Front reaches lens | f27 |
| Front reaches PDs | f56-f60 |
| Signal glow | f50-f150 (arrival over 14 frames, 40 symbols/s, seed 7) |

Inspected:
- `outputs/assets/optics_rx/optics_rx_fx_strip_arrival.png`: lenslets, plates and turning block are visible through the ghost glass; the beams fold down onto the PD row.
- `outputs/assets/optics_rx/optics_rx_fx_strip_signal.png`: view from behind the TIA. The PD spots blink, amber pulses cross the bonds and the TIA, spirals flash, and the output bonds and CPW glow.

The first pass had an opaque cone, a bright bench, dark Au facets and too-wide cameras; all were corrected in the second pass. Log: `outputs/logs/rx_fx_test_20261010_153307.log`.

### 8.4 Deliverables

- `assets/components/optics_rx/optics_rx_fx.blend`: base, plus cleanup, FX and test keys; `viz_beams` hidden; no test cameras or lights.
- `assets/components/optics_rx/optics_rx_fx.json`: API, colours, object names, distances, test timing, front track.
- Base files `optics_rx.blend`, `.glb` and `_parts.json` are unchanged (md5 checked before and after).
- Scripts:
  - `scripts/assets/optics_rx/rx_fx.py`
  - `scripts/assets/optics_rx/make_rx_fx_test.py`
  - `scripts/assets/optics_rx/compose_rx_fx_strip.py`

```text
scripts/bslot.sh -b assets/components/optics_rx/optics_rx.blend --python scripts/assets/optics_rx/make_rx_fx_test.py -- outputs/assets/optics_rx/fx_test
uv run --with pillow python scripts/assets/optics_rx/compose_rx_fx_strip.py outputs/assets/optics_rx/fx_test outputs/assets/optics_rx
```

## 9. v4: "changing bits" on the RX electrical path (DevLog-020 item X, 2026-10-10)

Owner text (verbatim, DevLog-020): "use similar changing bits VFX for Rx when signals are on the TIA similar to the Tx VFX." Also "no need to label spiral inductors at 0:36". The FX layer adds no labels; S drops its label.

### 9.1 What was wrong in film v3

At ~0:34-0:36 the RX amber glow read as a steady blob:
- The bond glow tubes (r 17 um) were well outside the 12.5 um wires.
- The emission strength of 4 saturated under fog glow.
- The spirals flashed at full strength.

The material formula was already the same as T's `tx_signal_glow`. The problem was scale and saturation.

### 9.2 Changes (`rx_fx.py`; the optical cyan layer is unchanged)

- Same "bits" statistics as TX electrical (T's `fx_material(electrical=True)`):
  - amber (1.0, 0.60, 0.13)
  - 40 symbols/s, 3 symbols/mm, so 13.3 mm/s travel
  - 55 % duty dashes
  - 4 random levels per symbol from (symbol, lane, seed), with floor 0.15 and base glow 0.12
- Differential pairs: `fx_mod` carries the polarity. The N line shows the complementary PAM4 level (3 - level) of the P line, on the PD stems, the input bonds, both on-die lines, the output bonds and the CPW pair.
- The spiral flashes are dimmed to 45 % (`fx_mod` coil flag), so the travelling symbols dominate.
- Readability:
  - emission strength 4.0 -> 2.6
  - bond glow tube r 17 -> 14.5 um (just outside the wire)
  - on-die ribbons 10 -> 16 um wide
  - gold facet ghost alpha 0.35 -> 0.16 (less foreground clutter)
- New `rx_land_frames(fx, f_on, f_front_end)` returns the frames where the light lands (bottom row first, then top row), so the bits start exactly on landing.
- New `intensity` argument on `rx_signal_glow`, keyed on `master`.

### 9.3 API (absolute frames)

```text
rx_fx.rx_land_frames(fx, f_on, f_front_end) -> (first, last)
rx_fx.rx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6, intensity=1.0)
```

### 9.4 Suggested film keys

These are relative to the S4 RX macro in `scripts/build_film.py` (scene 3 starts at frame 901 = 30.0 s) and keep S's current light keys.

| Key | Value | Film time |
|---|---|---|
| `rx_light_arrive(fx, F(3, 1.95), F(3, 3.4))` | unchanged | 31.95-33.40 s |
| Light lands, bottom row | F(3, 3.27) | 33.27 s |
| Light lands, top row | F(3, 3.40) | 33.40 s |
| `rx_signal_glow(fx, F(3, 3.27), F(3, 6.95), arrive_frames=12, intensity=1.0)` | bits start on landing | 33.27-36.95 s |

The v3 film started the glow at F(3, 4.0), which is 0.7 s after the light landed. Use `f0 = round(rx_fx.rx_land_frames(fx, F(3, 1.95), F(3, 3.4))[0])`.

### 9.5 Test

Same 150-frame test as Section 8.3: light f10-f60, land f55.6 / f60, signal f56-f150. Log: `outputs/logs/rx_fx_test_20261010_171407.log`.

Strips:
- `outputs/assets/optics_rx/optics_rx_fx_strip_signal.png`: view from behind the TIA.
- `outputs/assets/optics_rx/optics_rx_fx_strip_bits_tia.png`: new low angle along the PD row, close to the film's 0:34 view.
- `outputs/assets/optics_rx/optics_rx_fx_strip_arrival.png`: 5 frames.

Inspected: discrete amber dashes change level between frames f60 / f64 / f68 / f72; P and N lines differ; the arrival front reaches the flex pads by f68; the spirals flash dimmed.

The low-angle test camera sits close to the glass bench, so its grazing edge glow adds light sheets in the foreground. The film camera (S) differs.

`optics_rx_fx.blend` and `.json` are regenerated (JSON adds `land_frames` and `intensity`). Base `optics_rx.blend`, `.glb` and `_parts.json` are unchanged (md5 checked).

