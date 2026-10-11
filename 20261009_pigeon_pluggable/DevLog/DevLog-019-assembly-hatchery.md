# DevLog-019: Assembly hatchery v3 (workstream M)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v4 delivered 2026-10-10 (DFB off during pick and place, +1 s, varying PAM4 eyes, free-space test heads); v1 integrated by S in film v3 |
| Workstream | M (DevLog-018 section 3) |
| Owns | `scripts/assembly/`, `assets/components/assembly/`, `outputs/assets/assembly/`, `research/assembly_refs.md`, `references/assembly/`, this file |
| Inputs | DevLog-018 (owner feedback), DevLog-013 (E's hatchery v2 + API), DevLog-003 (hero), DevLog-008 (TX), DevLog-009 (RX); frames of `outputs/film_v2_1_20261010_av.mp4` 46.5-51 s |
| Consumer | S (scene assembly, DevLog-011) |

## 1. Request (owner, verbatim, 2026-10-10)

```text
0:50: come up with a better assembling animation showing actual isolators lens getting put in place and wirebonds getting connected and enclosure moved in place etc. DYOR on reference tools and equipments images as needed. Use ounolight instead of outnolight, too weird.
```

Coordinator brief (summary): research real assembly equipment (die/flip-chip bonders, active alignment with multi-axis stages + UV epoxy + UV cure, ball-wedge wire bonders, dispensers, lid placement, test); build an EEVEE-friendly assembly-line set at hall scale (birds 2.6x) with recognizable equipment using the real T/X/H parts; 2-3 s sequence for ~48.2-50 s with an API (`append_assembly(loc, rot_z)`, `key_assembly(f0)`) and suggested camera keys; signage "OUNOLIGHT HATCHERY"; fast-tier previews 540x674, look, iterate once.

## 2. What v2.1 shows now (frames 46.5-51 s)

Wide 3/4 shot of E's conveyor line (orange SCARA robots, gantry, PASS display) for ~1.6 s between white flashes; parts are too small to read; no lens, isolator, bond or lid action visible; sign and VO say OUTNOLIGHT.

## 3. Plan (written before implementation)

1. Research (rate-limited, one request at a time): vendor pages for die bonders, active alignment, UV cure, wire bonders and capillaries, dispensers, test; open-access packaging reviews. Record URLs and facts with grades in `research/assembly_refs.md`; a few Commons (licensed) reference images in `references/assembly/` with licence notes; vendor images reference-only (URLs, not downloaded).
2. Set (`scripts/assembly/build_assembly.py`, bpy, Blender 4.2, metres): line along local +X, five stations on a bench line, each a recognizable machine at human scale with tooling scaled 2.6x like the parts it handles:
   - S1 DIE ATTACH: gantry bond head with vacuum collet and bond-head camera, heated chuck, waffle-pack tray; places the PIC and then a DFB on the TX carrier.
   - S2 ACTIVE ALIGN + UV CURE: hexapod-style 6-axis stage, tweezer gripper places Si lens 1, the micro isolator and Si lens 2 in front of the lit DFB with small alignment dithers; dispense needle dot; UV light guides flash violet; power readout rises.
   - S3 WIRE BOND: ball-wedge bonder head (transducer horn, capillary, wire clamp, EFO wand spark), heated workholder; HS GSSG bonds appear one by one as the capillary loops each.
   - S4 DISPENSE + LID PRESS: dispenser needle runs a bead on the bird chassis; press lowers the shell halves (top, bottom) and wings onto the bird.
   - S5 TEST: test socket / probe arm, eye-check camera; eyes light up (TX red, RX cyan), PASS; finished bird hops off.
   - Parts appended from `assets/components/optics_tx/optics_tx.blend` and `assets/hero/pigeon_osfp_dr8.blend`, scaled x2.6 (true relative size to the birds).
   - Signage: neon "OUNOLIGHT HATCHERY", station labels; parody only.
3. Animation (`scripts/assembly/assembly_api.py`): one-shot sequence, every station acting in parallel, default 84 frames (2.8 s at 30 fps); `append_assembly(path, loc, rot_z)`, `key_assembly(objs, f0)` (re-keys every action to start at f0), `camera_keys(root, t0)` (suggested shots); events table for SFX.
4. Previews (`scripts/assembly/render_previews.py`): 540x674 EEVEE fast tier stills per station + short mp4 with the suggested camera; look, iterate once.

## 4. TODO

- [x] Research + refs doc (`research/assembly_refs.md`) + 3 Commons reference images (`references/assembly/`)
- [x] Probe T/H blends (object names, sizes): `outputs/logs/M_probe_tx_*.log`, `M_probe_hero_*.log`
- [x] Build set + parts + stations (`scripts/assembly/build_assembly.py`)
- [x] Animation + API + camera keys (`scripts/assembly/assembly_api.py`, sidecar JSON)
- [x] Previews round 1, look, fix (4 still rounds)
- [x] Previews round 2 + mp4 (2 animation rounds)
- [x] API test in a fresh scene (append at (600, 0, 0), retime x1.36, camera keys)
- [x] Final report
- [ ] S integration (place, retime, camera, beam note, SFX events)

## 5. As built

### 5.1 Files

| Path | Content |
|---|---|
| `scripts/assembly/build_assembly.py` | bpy builder -> `assets/components/assembly/assembly_hatchery.blend` + `.json` (about 25 s) |
| `scripts/assembly/assembly_api.py` | `append_assembly`, `key_assembly`, `camera_keys`, `apply_camera`, `event_frame` |
| `scripts/assembly/render_previews.py` | fast-tier stills / animation frames (540 x 674, EEVEE 1 sample, no shadows / AO / ray tracing) |
| `scripts/assembly/make_eye_texture.py` | synthetic PAM4 eye-diagram screen texture (uv; no measured data) |
| `scripts/assembly/probe_parts.py` | read-only blend probe (object names, bboxes, tris) |
| `assets/components/assembly/assembly_hatchery.blend` (6.8 MB, textures packed) | collection `ENV_assembly` (root empty `assembly_root`; `ASM_set`, `ASM_stations`, `ASM_lights`, `ASM_hooks`); 854 objects, 804 k tris (linked mesh data: two birds share the hero meshes) |
| `assets/components/assembly/assembly_hatchery.json` | stations x, events, camera plan, hooks, build log |
| `outputs/assets/assembly/` | 13 stills + `assembly_preview_20261010.mp4` (540 x 674, 30 fps, 90 frames, H.264 yuv420p, faststart) |

Rebuild:
```text
uv run --with numpy --with pillow python scripts/assembly/make_eye_texture.py assets/components/assembly/textures/eye_pam4.png
scripts/bslot.sh -b --factory-startup --python scripts/assembly/build_assembly.py
scripts/bslot.sh -b assets/components/assembly/assembly_hatchery.blend --python scripts/assembly/render_previews.py -- stills outputs/assets/assembly
scripts/bslot.sh -b assets/components/assembly/assembly_hatchery.blend --python scripts/assembly/render_previews.py -- anim <frames_dir>
```

### 5.2 Set and stations (local to `assembly_root`; line along +X, camera side -Y, back wall y 1.25)

| Station | x | Equipment (D: styled after the refs) | Action (frame offsets from the sequence start) |
|---|---|---|---|
| S1 DIE ATTACH | -2.2 | cabinet + granite slab + stage, gantry bond head with vacuum collet and head camera, waffle-pack trays (spare PICs, DFB-on-submounts), microscope + ring light | PIC carried in and placed (t28), head to the DFB tray, picks (t60), places the LD1 DFB (t88) |
| S2 ACTIVE ALIGN + UV CURE | -1.1 | two hexapod stages with arms and tweezers (A from the back, B from the front-left), two UV light guides, COUPLED POWER monitor, gel-pak with spare isolators / lenses | LD1 laser on (viz beam), tweezers A lowers the micro isolator (t13) with an alignment dither, B places Si lens 2 (t22), power bar rises 0.12 -> 1.0, UV flash t30-36 (tips + epoxy glow violet) |
| S3 WIRE BOND | 0.0 | ball bonder head: horn, capillary (66 um tip, 1/16 in shank), wire clamp, Au wire, spool, EFO wand + spark, heated workholder | T's 34 HS GSSG bonds split per wire; wires 0-7 pre-bonded, then one bond every 5 frames from t2 (18 inside 90 frames): above pad -> touch (ball) -> loop -> stitch -> rise, wire grows from the first bond, balls pop at touch, EFO spark at t+4.5 |
| S4 DISPENSE + LID PRESS | 1.15 | bird bench + pallet, needle dispenser on an arm, press frame with cylinder, platen and vacuum cups | TIM bead on the DSP thermal pad t12-40 (curve grows, needle follows), top shell + wings + tail (H `shell_top` + `fins`) lowered t58-70 |
| S5 TEST / EYES ON | 1.95 | test socket under the feet, eye-check camera arch, eye-diagram monitor, PASS lamp, DISPATCH step at x 2.62 | eyes on t73-77 (H glow rings + lenslets + added eye discs), monitor eye + "TX EYE PASS" t77-78, crouch t78, hop 1 t80-89 to the bench end, hop 2 t92-101 onto DISPATCH |

Scale: parts x2.6 (true relative to the birds). TX parts are copies of T's objects (lenses, isolator stack, DFB, PIC with texture, bonds); birds are copies of H's hero. Tool ends true size inside the x2.6 work frame; machine bodies human scale. Sign: neon "OUNOLIGHT HATCHERY"; station plates "S1 DIE ATTACH" ... "S5 TEST / EYES ON"; machine badges generic (DB-26, AA-6, WB-26). H's leg band text (OUTNOLIGHT) is hidden on both birds.

### 5.3 API (S)

```python
import sys; sys.path.insert(0, "<repo>/scripts/assembly"); import assembly_api as AA
root, objs = AA.append_assembly(AA.DEFAULT_BLEND, loc=(600, 0, 0), rot_z=0.0)
AA.key_assembly(objs, f0, speed=1.0)            # all keys -> start at film frame f0; speed 1.36 squeezes 90 -> 66 frames
keys = AA.camera_keys(root, f0, speed=1.0)      # [(frame, loc, target, lens, cut)] world space, shots A, B, C
AA.apply_camera(cam_obj, keys)                  # or feed S's cam() / cam_jump() with the same tuples
AA.event_frame("S5_eyes_full", f0, speed=1.0)   # SFX / VO sync
```

Events (offsets): S1_pic_place 28, S1_dfb_pick 60, S1_dfb_place 88; S2_iso_place 13, S2_lens2_place 22, S2_uv_on 30, S2_uv_off 36; S3 bonds every 5 from 2 (18 in window); S4_dispense 12-40, S4_press_down 58, S4_press_closed 70; S5_eyes_on 73, S5_eyes_full 77, S5_pass 78, S5_hop1 80, S5_land1 89, S5_hop2 92, S5_land2 101.

Verified in a fresh scene (`outputs/logs/M_api_test_*.log`): 854 objects, 1150 keys retimed, 8 camera keys at (600, 0, 0), tweezers and camera land where expected.

### 5.4 Suggested camera (sidecar `camera_plan`, local; hard cuts between shots or S's white fills)

| Shot | Frames | Subject | Lens (36 mm horizontal sensor) |
|---|---|---|---|
| A_align_macro | 0-32 | S2 side view along the LD1 laser train: DFB -> lens 1 -> isolator placed -> lens 2 placed -> UV flash; slow push | 560 -> 600 (camera ~0.19 m away, frame ~11 mm) |
| B_bond_macro | 33-58 | S3 capillary forming bonds, camera trucks along the bond row | 450 |
| C_press_to_test | 59-89 | S4 press closing the shell (t58-70) -> truck right to S5: eyes on, PASS, hop off | 55 -> 45 |
| overview (optional) | - | whole line | 26 |

Total 3.0 s at speed 1.0; at the current 48.2-50 s slot (~1.8 s) use speed ~1.6, or drop shot B.

## 6. Open issues (for S / coordinator)

1. Macro shots: the camera sits 0.12-0.2 m from the work; S's caption overlay plane is 0.06 m in front of the lens, so it stays clear. Lens values 450-600 mm are unusual; check the overlay scaling if S ties overlay size to the lens.
2. The S2 laser beam is viz only (1311 nm invisible): shot A needs the beam note.
3. Fast-tier look: macro shots are bright and flat (no shadows / AO); detail reads (lens, isolator stack, bonds, capillary). In the final tier, shadows and bloom will help the UV flash, EFO spark and eye glow.
4. The tweezer bodies are large in shot A (true-size bodies on 0.6 mm parts); they read as tools but cover part of the frame for ~10 frames each.
5. H's leg band text still says OUTNOLIGHT (hidden here); H/coordinator to rename in the hero.
6. The set is a separate room (back wall, ceiling, own lights in `ASM_lights`); the S scene places it (E's hatchery sat at (600, 0, 0)). Remove E's hatchery from S5 when this replaces it.
7. Machines are simplified massing (no cable chains, no enclosures with glass); fine at the shot durations, not for long wide holds.

## 7. Progress log

- 2026-10-10 15:00: read DevLog-018/013/003/008/009, `hatchery_api.py`, S's S5 hatchery staging in `build_film.py` (hatchery at (600, 0, 0), shot 8.09-9.78 s of S5); contact sheet of film v2.1 46.5-51 s viewed. Plan written.
- 2026-10-10 15:20: research (ficonTEC AL2000, PI hexapod alignment, Excelitas LX500 UV LED, MRSI die bonders, K&S 4124 SOP, capillary patents, Nordson jetting, Keysight DCA-M); 3 Commons images (PD x2, CC BY-SA x1); `research/assembly_refs.md` written.
- 2026-10-10 15:25: probes of T's and H's blends; eye-diagram texture generated and viewed (first version had sampling stripes, fixed).
- 2026-10-10 15:32: build v1; stills round 1: TX copies collapsed to the origin (orphan appended objects carry a stale matrix_world). Fixed by linking the templates to a temp collection and updating the view layer; stage plate replaces the round chuck (trays were floating); granite darker; sign spill 60 -> 20 W.
- 2026-10-10 15:35: rounds 2-4: macro lenses 150 -> 240 -> 560-600; S2 re-staged (side camera along the laser train, tweezers from back / front-left, slimmer arms, visible viz beam), S5 bird yaw flipped (TX eye to camera), hop turns toward +X, eye glow discs added; key lights per micro station.
- 2026-10-10 15:39: animation round 1 (90 frames): tweezer bodies black (metal without reflections in fast tier), UV guides crossing shot A. Fixed (non-metal steel, guides behind/above), lights lowered, shot C lens 55.
- 2026-10-10 15:42: animation round 2 viewed (sheet + 4 full-res frames): isolator + lens placement, UV flash, bonds forming with EFO spark, press closing, eyes on, PASS, hop all read. Final stills + mp4 to `outputs/assets/assembly/`; scratch frames deleted.
- 2026-10-10 15:45: API test in a fresh scene; fixed `camera_keys` (root matrix needed a view-layer update).

## 8. v4 wave (DevLog-020, owner feedback 2026-10-10)

Owner (verbatim excerpt): "0:48: during pick and place DFB should be off, once the tip leaves the Si lens turn on the DFB. 0:49 make the assembling line scene 1 s longer, so that the closing the container is visible after the fade out. Eye diagram should also be varying a bit similar to the interconnect film's scope. And the final test stand should have IO couplers coming in next to the bird eyes, running test with eye diagrams on, and then eye diagram off and couplers moving away."

TODO:
- [x] DFB LD1 off (facet glow, viz beam, T beam core, COUPLED POWER bar at 0) until tweezer B leaves Si lens 2 (rise starts t29); DFB on at t30; UV cure moved after it (t34-39); LD2 dark in S2
- [x] Sequence 90 -> 144 build frames; shot A 0-42, B 43-70, C 71-143; lid close t74-86 inside shot C, platen rises again t90-102 so the closed enclosure is seen
- [x] Varying PAM4 eye: `scripts/assembly/make_eye_atlas.py` -> `assets/components/assembly/textures/eye_pam4_atlas.png` (24 frames, 6 x 4 cells of 384 x 256; amplitude / noise / jitter wander, persistence, seamless loop); screen shader picks cell floor(phase) mod 24, phase keyed linear 1 cell per build frame (retimes with key_assembly); two screens TX EYE / RX EYE (offset 11 cells)
- [x] Test stand: socket (electrical side) + gantry with two free-space optical test heads (2x4 lens-array window, ring LED, carriage); in t98-106, eyes on t104, eye diagrams + viz beams eye <-> head t107-124, PASS t120, diagrams off t124, heads out t125-133, hop t133-142, hop 2 t145-154
- [x] Refs: `research/assembly_refs.md` section 5; `references/assembly/commons_semiauto_probe_station.jpg` (CC BY-SA 4.0)
- [x] Keyed emissives (beams, PASS, lamp, eye discs, UV tips, spark, DFB facet copy, S5 eye rings) get dark base colours so they read OFF at emission 0 (round-1 check showed PASS and the beam visible while "off")
- [x] Preview (fast tier) checked at DFB off / on, lid closing / closed, heads in / eyes on / eye off / heads out, hop; content clear of the bottom 18 % caption band in shot C (bench front only)
- [x] API test (fresh scene, x1.36): 861 objects, 1331 keys retimed, 9 camera keys

Framing note: no fiber on the birds; the test "IO couplers" are free-space optical test heads docking at the 2x4 lenslet eyes (see refs 17-19).

Timing (build frames from the sequence start; S retimes with `key_assembly(objs, f0, speed)`): DUR 144 build frames = 90 film frames (3.0 s) at speed 1.6 (old 56.25 frames, +1.125 s), or 120 film frames (4.0 s) at speed 1.2. Shots: A 0-42 (43 frames), B 43-70 (28), C 71-143 (73). Events: see the sidecar `events` (S2_dfb_on 30, S2_uv_on 34, S3 bond every 5 from 2, S4_press_down 74, S4_lid_closed 86, S4_platen_up 90, S5_heads_in 98, S5_heads_docked 106, S5_eyes_on 104, S5_eye_diagram_on 107, S5_pass 120, S5_eye_diagram_off 124, S5_heads_out 125, S5_heads_away 133, S5_hop1 133, S5_land1 142; alias S5_eyes_full 108).

Outputs: `outputs/assets/assembly/assembly_preview_v4_20261010.mp4` (540 x 674, 144 frames) + 19 stills (the v1 stills / mp4 were replaced).

Progress:
- 2026-10-10 17:10: read DevLog-020; S's integration (speed 1.6, slot 47.85-49.75 s); interconnect film eye generator (tex_gen.py eye_frame, s02_retimers make_eye_sequence) read.
- 2026-10-10 17:12: refs (EXFO MCB, FormFactor CM300xi-SiPh, PI electro-optical probing); Commons probe-station image.
- 2026-10-10 17:15: eye atlas v1 too closed (ISI sigma 0.30 UI); sigma 0.17, lower noise: 3 open PAM4 eyes, varying.
- 2026-10-10 17:25: builder v4 (DFB timing, DUR 144, platen separate from the lid, S5 test heads, eye screens, camera plan); stills: PASS / beam / eyes visible while off (base colours) and a cable read as an antenna; fixed.
- 2026-10-10 17:35: 144-frame preview rendered and checked at the requested moments; final stills; API re-test; alias event S5_eyes_full added (S's v3 code may query it).

## 9. v4.1: final test stand rebuilt (owner rejection, 2026-10-10)

Owner (verbatim): "for the assembling scene, Tx and Rx eyes separate makes no sense, unless you also have a panel for full link eye. Also refine the test stand, find proper references and come up with your own adaptation for this arrangement. The feet should also be connected. Delegate to a subagent to build this properly. This is unacceptable as of it is right now."
Coordinator: rebuild shot C test stand from references (HCB/MCB + BERT, TX -> reference receiver -> DCA, RX <- reference transmitter + VOA, full link loopback), feet visibly connected, parody instruments, plain screens (one accent), same slot (144 build frames = 92 film frames at speed 1.5652), same API; fix floor z-fighting and yellow strip glitches.

TODO:
- [x] Refs: `research/assembly_refs.md` section 6 (facts 21-29, adaptation table, images); `references/assembly/commons_bert_vs_scope_eye_diagram.jpg` (CC BY-SA 4.0)
- [x] Screens: `scripts/assembly/make_test_screens.py` -> `scr_tx.png`, `scr_rx_base.png` + `scr_rx_curve.png`, `scr_link_base.png` + `scr_link_rows.png` (dark grey, light grey text, green accent only)
- [x] MCB socket (audit: module under test sits on an MCB, not an HCB): board, 52 pogo pins under the foot contacts (pad centres taken from the hero's foot pad meshes), belly pogo pedestal (6 pads = 4 LS + 2 merged VCC = 8 contacts), socket nest + posts, traces to 2 x 8 coax launches, coax cables to the BERT, LS/VCC header
- [x] Instruments (parody): KEYSIGHTLESS sampling scope (Keysight), ANRITSOON BERT PPG/ED 16 x coax (Anritsu), VIAVOID VOA + optical switch (VIAVI); generic grey boxes
- [x] Test heads on linear stages (towers): "REF RX > DCA" at the TX (right) eye, "REF TX + VOA" at the RX (left) eye; viz beams eye <-> head while testing
- [x] Screens TX | FULL LINK | RX on a bar: live PAM4 optical eye + TDECQ / OMA / ER; FULL LINK lanes fill 1-8 + live host RX eye + PASS; RX waterfall drawn by the VOA sweep + KP4 limit + SENSITIVITY PASS; off after the test (verdict text hidden when off)
- [x] Choreography: bird waits on the bench, hops into the socket and seats; heads dock; screens; PASS; screens off; heads retract; hop off
- [x] Shots re-split inside the same 144 frames: A 0-38, B 39-60, C 61-84 (S4 enclosure close), D 85-143 (test stand); `camera_keys` default now includes D (S's call unchanged)
- [x] Fixes: bench top plate 1 mm proud of the cabinet top (was coplanar -> floor streaks), yellow strip offset from the pallet faces, S5 pallet replaced by a fixture base, S4 bird root on the pallet top, S4 dispenser arm/park moved to -X (was in front of the TX screen)
- [x] Fast preview checked: socket + feet close-up, heads docked, three screens with readouts, PASS, off + retract, hop
- [ ] Fresh-context audit: launched 2026-10-10 21:40, result not yet applied at handback
- 2026-10-10 21:45: preview `outputs/assets/assembly/assembly_preview_v4_1_20261010.mp4` (144 frames) + 22 stills; v4 mp4 replaced

### 9.1 Audit corrections (fresh-context audit, 2026-10-10) - applied
- Board is an MCB (module under test), not an HCB: silkscreen "MCB  800G  MODULE COMPLIANCE BOARD", docstring / comments / refs 6.2 updated, object names S5_mcb_*.
- RX waterfall now steepens with rising power; x label "LOG BER  vs  RX OMA, dBm (VOA)" (no numeric ticks); "SENSITIVITY PASS" moved below the curve.
- TX readout: "TDECQ 1.6 dB PASS" (no limit shown).
- Refs: fact 25 marked "search summary, not read"; fact 29 reworded (passive / DSP active loopbacks characterize and test host ports). KEYSIGHTLESS kept (film parody convention).

### 9.2 Board upgrade (owner add-on, verbatim): "further improve the test board, use proper ref, add both low speed control & power channels, and high speed signaling channels you already have. Use proper vertical launchers people usually have, and also include various components and ICs onto the board. Low speed and power supplies should also be connected to the board."
- Refs: `research/assembly_refs.md` section 7 (OSFP spec low-speed / VCC pins read locally; Wilder and MultiLane OSFP MCBs; Southwest Microwave 1.85 mm vertical launch). Board parts not seen on a specific vendor board are marked D.
- Built: 32 vertical-launch connectors for 16 matched differential pairs, 32 coax to the BERT (now 2 x 16 ports); LS block (USB-I2C bridge, USB-C to a CMIS HOST box, EEPROM, oscillator, I2C header, DIP + jumpers for LPWn/PRSn and INT/RSTn, LEDs on at seating, test points) with traces from the 4 LS belly pads; power block (banana jacks with leads to a RIGOLD bench supply showing 3.30 V / 3.8 A, fuse, TVS, sense resistor + monitor, DC-DC + inductor, bulk + MLCC caps) with wide pours to the 2 VCC pads; mounting holes; silkscreen refdes. Board 136 x 125 mm true (x2.6 in the hall).
- Unchanged: frames, shots, API, event times. Checked stills: board overview (launches, LS, power), socket close-up (D key 1), stand wide with cables to the BERT / supply / CMIS host, screens with the audit fixes.
- Not done: a second fresh-context audit of the board (time); the stand-wide still is a check view only.
- 2026-10-10 22:10: rebuilt; stills + preview mp4 re-rendered; READY marker written for S.

## 10. v4.2: assembly scene at 2x duration (owner, 2026-10-10)

Owner (verbatim): "double the innolight assembling full scene (aligning isolators and lens and turning on dfb, enclosure, and link testing) duration by 2x so that all the details are given sufficient time and attention (slow down the moving handlers a bit, show the turned on DFB longer, run the wirebonding at the same speed but 2x more bonds, show more link tests steps etc.). Adding more narratives accordingly describing the process."

Slot: 184 film frames (6.133 s) at speed 1.5652 -> DUR = 288 build frames exactly (288 / 1.5652 = 184.0).

| Shot | Build frames | Film frames from slot start (speed 1.5652) | Content |
|---|---|---|---|
| A_align_macro | 0-71 | 0.0-46.0 | isolator (place t18) + Si lens 2 (place t31) with slower tweezer moves and dither; DFB off; tool leaves lens t39; DFB on t40 held to the cut, COUPLED POWER bar rises to its peak t54; UV cure t56-64 |
| B_bond_macro | 72-121 | 46.0-77.9 | same 5 frames per bond; all 34 HS GSSG wires bonded in sequence (t2-t172), 10 bonds inside the shot (was 5); lens 330 mm shows more of the RF pad row and the DC bonds |
| C_press_to_test | 122-167 | 77.9-107.3 | dispense close-up (TIM bead t124-150), needle out, press down t154, lid closed t164 (visible), platen up t168-178 |
| D_test_stand | 168-287 | 107.3-184.0 | seat (hop t168-178, seated t180); power on t182 (board LEDs, supply current 0 -> 0.9 -> 2.1 -> 3.2 -> 3.8 A); CMIS / I2C: link up t184, module ID t190, ModuleReady t194, DDM t198-199 (TEMP 42.5 C, VCC 3.29 V, BIAS 58 mA, TXPWR 1.2 dBm); heads in t198-206; TX optical eye lanes 1-8 every 5 frames t210-250 (lane label + TDECQ / OMA / ER per lane + PASS); RX sweep t250-262 (+ SENSITIVITY PASS); full link t260-270; PASS t270; screens off + heads out t275-281; hop t280-287 |

Readout values are inside the DR8 datasheet limits read locally (FS 800GBASE-DR8 / Approved Networks DR8): TDECQ max 3.4 dB, OMA outer -0.8..4.2 dBm, ER >= 3.5 dB, average launch power -2.9..4.0 dBm per lane, case 0-70 C, VCC 3.135-3.465 V, TX bias DDM range 0-100 mA, max power 16.5 W (FS); CMIS ModuleReady per OIF-CMIS 5.4 6.3.2.5.6. Per-lane TX values: L1 1.6/1.9/4.6, L2 1.5/2.1/4.4, L3 1.7/1.8/4.7, L4 1.4/2.2/4.5, L5 1.6/2.0/4.6, L6 1.8/1.7/4.3, L7 1.5/2.0/4.8, L8 1.6/1.9/4.5 (TDECQ dB / OMA dBm / ER dB; illustrative). Instruments moved: RIGOLD supply + CMIS HOST (with screen) on the BERT top; leads and USB rerouted.

Events (build frame | film frame | film s from the slot start, speed 1.5652; also in `assembly_hatchery.json` "events"):

| Event | Build | Film frame | Film s |
|---|---|---|---|
| S3_bond_first | 2 | 1.3 | 0.043 |
| S3_bond_01 | 3 | 1.9 | 0.064 |
| S3_bond_02 | 8 | 5.1 | 0.170 |
| S3_bond_03 | 13 | 8.3 | 0.277 |
| S2_iso_place | 18 | 11.5 | 0.383 |
| S3_bond_04 | 18 | 11.5 | 0.383 |
| S3_bond_05 | 23 | 14.7 | 0.490 |
| S3_bond_06 | 28 | 17.9 | 0.596 |
| S2_lens2_place | 31 | 19.8 | 0.660 |
| S3_bond_07 | 33 | 21.1 | 0.703 |
| S3_bond_08 | 38 | 24.3 | 0.809 |
| S2_tool_leaves_lens | 39 | 24.9 | 0.831 |
| S2_dfb_on | 40 | 25.6 | 0.852 |
| S3_bond_09 | 43 | 27.5 | 0.916 |
| S3_bond_10 | 48 | 30.7 | 1.022 |
| S3_bond_11 | 53 | 33.9 | 1.129 |
| S2_power_peak | 54 | 34.5 | 1.150 |
| S2_uv_on | 56 | 35.8 | 1.193 |
| S3_bond_12 | 58 | 37.1 | 1.235 |
| S3_bond_13 | 63 | 40.3 | 1.342 |
| S2_uv_off | 64 | 40.9 | 1.363 |
| S3_bond_14 | 68 | 43.4 | 1.448 |
| S3_bond_15 | 73 | 46.6 | 1.555 |
| S3_bond_16 | 78 | 49.8 | 1.661 |
| S3_bond_17 | 83 | 53.0 | 1.768 |
| S3_bond_18 | 88 | 56.2 | 1.874 |
| S3_bond_19 | 93 | 59.4 | 1.981 |
| S3_bond_20 | 98 | 62.6 | 2.087 |
| S3_bond_21 | 103 | 65.8 | 2.194 |
| S3_bond_22 | 108 | 69.0 | 2.300 |
| S3_bond_23 | 113 | 72.2 | 2.407 |
| S3_bond_24 | 118 | 75.4 | 2.513 |
| S3_bond_25 | 123 | 78.6 | 2.619 |
| S4_dispense_start | 124 | 79.2 | 2.641 |
| S3_bond_26 | 128 | 81.8 | 2.726 |
| S3_bond_27 | 133 | 85.0 | 2.832 |
| S3_bond_28 | 138 | 88.2 | 2.939 |
| S3_bond_29 | 143 | 91.4 | 3.045 |
| S3_bond_30 | 148 | 94.6 | 3.152 |
| S4_dispense_end | 150 | 95.8 | 3.194 |
| S3_bond_31 | 153 | 97.8 | 3.258 |
| S4_press_down | 154 | 98.4 | 3.280 |
| S3_bond_32 | 158 | 100.9 | 3.365 |
| S3_bond_33 | 163 | 104.1 | 3.471 |
| S4_lid_closed | 164 | 104.8 | 3.493 |
| S3_bond_34 | 168 | 107.3 | 3.578 |
| S4_platen_up | 168 | 107.3 | 3.578 |
| S5_seat_hop | 168 | 107.3 | 3.578 |
| S4_platen_home | 178 | 113.7 | 3.791 |
| S5_feet_seated | 180 | 115.0 | 3.833 |
| S5_power_on | 182 | 116.3 | 3.876 |
| S5_psu_current_step_1 | 183 | 116.9 | 3.897 |
| S5_eyes_on | 184 | 117.6 | 3.919 |
| S5_psu_current_step_2 | 185 | 118.2 | 3.940 |
| S5_psu_current_step_3 | 187 | 119.5 | 3.982 |
| S5_eyes_full | 188 | 120.1 | 4.004 |
| S5_psu_current_step_4 | 189 | 120.8 | 4.025 |
| S5_cmis_id | 190 | 121.4 | 4.046 |
| S5_cmis_ready | 194 | 123.9 | 4.132 |
| S5_cmis_ddm | 198 | 126.5 | 4.217 |
| S5_heads_in | 198 | 126.5 | 4.217 |
| S5_heads_docked | 206 | 131.6 | 4.387 |
| S5_eye_diagram_on | 210 | 134.2 | 4.472 |
| S5_tx_lane_1 | 210 | 134.2 | 4.472 |
| S5_tx_lane_2 | 215 | 137.4 | 4.579 |
| S5_tx_lane_3 | 220 | 140.6 | 4.685 |
| S5_tx_lane_4 | 225 | 143.8 | 4.792 |
| S5_tx_lane_5 | 230 | 146.9 | 4.898 |
| S5_tx_lane_6 | 235 | 150.1 | 5.005 |
| S5_tx_lane_7 | 240 | 153.3 | 5.111 |
| S5_tx_lane_8 | 245 | 156.5 | 5.218 |
| S5_rx_sweep_on | 250 | 159.7 | 5.324 |
| S5_tx_lanes_done | 250 | 159.7 | 5.324 |
| S5_full_link_on | 260 | 166.1 | 5.537 |
| S5_rx_sweep_done | 262 | 167.4 | 5.580 |
| S5_full_link_lanes_done | 270 | 172.5 | 5.750 |
| S5_pass | 270 | 172.5 | 5.750 |
| S5_eye_diagram_off | 275 | 175.7 | 5.857 |
| S5_heads_out | 275 | 175.7 | 5.857 |
| S5_hop1 | 280 | 178.9 | 5.963 |
| S5_heads_away | 281 | 179.5 | 5.984 |
| S5_land1 | 287 | 183.4 | 6.112 |
| S5_hop2 | 290 | 185.3 | 6.176 |
| S5_land2 | 299 | 191.0 | 6.368 |

Checks: fast stills at DFB off / on / power peak / UV, bonding, dispense, lid closing / closed, hop in, power on, CMIS + DDM, heads docked, TX lane 3 and 8, RX + full link, PASS, retract, hop off; preview `outputs/assets/assembly/assembly_preview_v4_2_20261010.mp4` (288 frames). Fixes during the pass: banana leads no longer rise vertically in the socket close-up (routed low behind the bird); CMIS view moved right of the RX stage tower; USB plugs into the host's side; no knobs over the CMIS screen.
- 2026-10-10 22:50: v4.2 built (937 objects, 873 k tris), checked, delivered.
