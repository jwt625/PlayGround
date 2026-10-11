# DevLog-006: Storyboard rev 3 and crude v0.2 (workstream S)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v0.2 - v0.2.3 rendered (2026-10-10); waiting for review |
| Depends on | `DevLog-005-feedback-wave-rev3-orchestration.md` (owner feedback verbatim, workstreams, interfaces), `DevLog-004-crude-v0_1.md` (v0.1 build), `research/*.md` |
| Generator | `scripts/build_crude_film.py` (v0.2.3); v0.2.1 frozen as `scripts/build_crude_film_v0_2_1.py`, v0.2 as `scripts/build_crude_film_v0_2.py`, v0.1 as `scripts/build_crude_film_v0_1.py`; blend `scenes/crude_v0_2.blend` (built by the current script) |
| External assets | Hero GLB (H, `assets/hero/pigeon_osfp_dr8.glb`); racks + layout (R, `assets/components/datacenter/`, DevLog-007); TX / RX subassemblies (T / X, `assets/components/optics_tx|optics_rx`, DevLog-008 / 009) |
| Authors | Claude (scene/camera subagent S), at Wentao Jiang's request |

## 1. Brief (coordinator, 2026-10-10)

a. Outdoor night-street opening like the ending; the camera follows one bird flying into the datacenter (bird flap) to charge and exchange data with other pigeons at the landing pad on the bird racks; the family tree placed in this opening without breaking the continuous camera; narration lines unchanged (timing only).
b. Tree: no years; balance a sparse real bird tree with a simple pluggable evolution tree; hero node prominent; facts only from `research/*.md`.
c. Background birds: constant random head motion (bobs, turns, tilts), repeated eye contact (TX right eye to another bird's RX left eye, hold, move on); all ambient beams hidden.
d. Any shot showing beams carries "* beams invisible (1311 nm); rendered for viewing".
e. TX -> RX switch: zoom out of the hero's TX eye, the hero turns its head, zoom into its RX eye (same bird); replaces the partner-bird beam ride.
f. Ending: natural night street with objects; birds casually perched at varied positions and heights (no polygon), all observing the Subject and communicating.
g. Datacenter: placeholders until R delivers, but laid out now with bird racks about 1/3 of all racks and pigeonholes filled top to bottom.
h. Keep the S2/S5 explaining scenes; use the hero GLB for the hero (hall scale); T/X subassemblies in S3/S4 when delivered; leash out of the TAIL (owner decision, 2026-10-10); in S2 the Subject pulls the bird out by the tail leash.

## 2. Storyboard rev 3 (60 s + 2 s)

Camera: one continuous path (Bezier keys); colour fills only for the scale jumps into and out of the eyes; the only hard cut is to the S7 card. Narration text unchanged from rev 2 (DevLog-002 section 4); timing adjusted where noted.

### S1 Street, tree, bird flap, landing pad (0-10)

| t | Action | Camera |
|---|---|---|
| 0.0-3.6 | Night street in front of the datacenter facade (the same street as S6). A hologram family tree stands over the street. The hero flies up beside the trunk; leaves pass close. Below, the Subject walks into the datacenter door with his coffee. | Rises with the bird, about 1.2 m in front of the hologram |
| 3.6-4.6 | The bird passes the top: CYBER PIGEON node (largest, red, pulsing); the pluggable lineage line ends there | Tops out over the node |
| 4.6-6.0 | The bird turns to the facade and flies through a bird flap high in the wall; the flaps swing | Follows behind it through the flap |
| 6.0-7.3 | Inside above the aisle: compute racks (two thirds) and bird racks (one third, pigeonholes filled top to bottom). The bird glides down to the landing pad on top of its bird rack; the pad glows (charging); neighbours trade one-eyed glances (data exchange, no beams) | Follows, descending |
| 7.3-8.2 | It drops off the pad, turns, and backs into the last empty pigeonhole of its rack (docks) | Eases in |
| 8.0-10.0 | The Subject walks down the aisle; heads track him with one-eyed stares; settle on the hero | Settles on the hero |

VO unchanged: "Birds evolved from dinosaurs. / Or so they told you. / Meet the OSFP: Overhead Surveillance Feathered Platform. / It is watching you."
Labels: tree (below), "LANDING PAD: CHARGE + SYNC", "800G OSFP DR8 / DR8: DRONE RECON, 8 LANES", "HEADS TRACK THE SUBJECT".

Tree (no years; balance of a sparse real bird tree and a simple pluggable lineage). Bird clades only where Stiller 2024 and Prum 2015 agree (DevLog-001 section 4); leaves trait-matched as before. A gold lineage line runs through the pluggable names (simplified, by lanes and data rate; lane counts from `research/form_factors_and_companies.md` A/B):

| Leaf (bird) | Pluggable | Lineage marks |
|---|---|---|
| Theropod (root) | 1x9 soldered | start |
| Archaeopteryx | GBIC | 1 lane |
| Palaeognaths: moa (extinct), elephant bird (extinct), emu, ostrich | XENPAK, CFP, CFP2-DCO, XPO | coherent branch: CFP -> CFP2-DCO; XPO: 64 lanes |
| Galloanserae: chicken | SFP | 1 lane |
| Neoaves (unresolved): hummingbird, albatross, parrot | QSFP-DD, 400ZR, compatibles | QSFP-DD: 8 lanes; 400ZR on the coherent branch |
| Columbaves: cuckoo, passenger pigeon (legacy), rock pigeon, cyber pigeon (hero), dodo ghost | AEC (copper egg), QSFP+, QSFP28, 800G OSFP DR8, CPO | QSFP: 4 lanes; OSFP: 8 lanes |

Lineage line: 1x9 -> GBIC -> SFP -> QSFP+ -> QSFP28 -> QSFP-DD / OSFP -> XPO; side branches COHERENT (CFP -> CFP2-DCO -> 400ZR), COPPER (AEC), CO-PACKAGED (CPO). Title "BIRD TREE (TRAIT-MATCHED) + PLUGGABLE LINEAGE (SIMPLIFIED)". Extinct = gray (legend).

### S2 Outside: tail leash, beak latch, eyes, feet (10-20)

| t | Action | Camera |
|---|---|---|
| 0.0-1.8 | The Subject pulls the leash loop that hangs from the cell lip; the leash runs from the bird's tail; the beak opens (latch release); the bird slides out and dangles by its tail leash, head down | Orbit as it comes out |
| 1.8-2.2 | He turns it upright in his hand | Push onto the face |
| 2.2-4.5 | Lateral eyes: the head turns to present the right eye (red rim, TX), then the left eye (cyan rim, RX); dark (unpowered); beak snaps | Face close-up |
| 4.5-7.0 | Tilted nose-up: feet (4 toes each, gold pads) and belly pads | Under the belly |
| 7.0-10.0 | Plugged back in (cutaway): claws (GND), palms (power), toe pads (signal) touch in order; eyes light; beak bites the cage lip | Low macro, then up to the beak |

VO unchanged. Labels as v0.1.

### S3 Right eye: transmit (20-30)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | In its cell the hero turns its head: right eye to the aisle; push in; red fill | Push along the eye axis |
| 2.0-8.3 | TX macro set as v0.1 (lasers, splitter tree, 8 MZMs, monitor PDs, edge couplers, prism 1x8 -> 2x4, 2x4 lenslets, 8 free-space beams). T subassembly replaces it when delivered | As v0.1 |
| 8.3-10.0 | Red fill; zoom out of the TX eye back to the hero's head; 8 beams leave its right eye across the aisle | Pull back from the eye |

Beam note on screen whenever free-space beams are visible (macro 7.3-8.4, hall 8.6-10).

### S4 Left eye: receive (30-40)

| t | Action | Camera |
|---|---|---|
| 0.0-1.4 | The hero turns its head the other way; its left (RX) eye faces the aisle; a bird across the aisle sends its 8 beams into it (note) | Holds on the head, then pushes in |
| 1.4-2.5 | Cyan fill into the RX eye | Push |
| 2.0-6.4 | RX macro set as v0.1 (lenslets, prism 2x4 -> 1x8, 8 PIN PDs, octal TIA). X subassembly replaces it when delivered | As v0.1 |
| 6.4-10.0 | Bird POV from the hero's RX eye: the Subject boxed and tracked | POV push |

### S5 Exploded view and hatchery (40-50)

As v0.1, with the hero GLB (H) exploded along its own `explode` offsets (ghost shell), labels on the GLB parts; I2C stream; parts snap back; band "HATCHED BY OUTNOLIGHT"; hatchery fade.

### S6 Street: casual perches, observation, links (50-60)

| t | Action | Camera |
|---|---|---|
| 0.0-2.2 | The Subject walks out of the door into the night street; the hero and two birds follow | Follows behind him through the door |
| 2.2-5.0 | Birds already perched casually: lamp-post tops and arms, overhead wires, a railing, AC units, ledges, a sign, a bin, a parked scooter, rooftop edges, a CCTV pole. All heads track him with one-eyed stares; pairs trade glances. A few eye-to-eye links shown (note) | Rising crane past the perches |
| 5.2-6.8 | A bird on a rooftop edge uplinks to a satellite (note) | Tilt up to the sky |
| 6.8-8.8 | Street CCTV cameras link to birds (note) | Down to the cameras |
| 8.8-10.0 | Heads turn one eye to the camera; the hero blinks its TX eye (TX disable) | Push into the hero's eye |

### S7 Disclaimer (60-62)

Unchanged.

## 3. Implementation notes

- Eye model now matches the hero GLB: eye centres at (47.5, -+12.8, 80.5) mm in the bird frame, axes purely lateral (v0.1 used a 20 deg forward tilt).
- Hero = GLB under the hero rig root (rotated +90 deg about Z: GLB -Y beak -> rig +X), hall scale `S_BIRD = 2.6`; head parts (centroid in the head) reparented to the rig's neck pivot so the head can turn; GLB leash hidden, replaced by a stretch leash from the tail to an animated loop; lids added at the GLB eye positions.
- Hall layout (placeholder until R): 2 x 13 racks; bird racks = 9 of 26 (about 1/3), 7 rows x 3 pigeonholes from 0.16 m to 2.17 m, every cell filled; compute racks = GB300-style placeholders (tray stripes, LEDs, cable bundles). Landing pad on top of each bird rack front.
- Crowd heads: random head turns, tilts and bobs every 0.35-1.1 s; eye-contact events between nearby birds (link solver from v0.1) held 0.8-1.6 s; no ambient beams.
- New overlay kind NOTE (bottom left) for the beam note.

## 4. As built

| Item | Value |
|---|---|
| Videos | `outputs/crude_v0_2_20261010.mp4` (v0.2: R racks, placeholder macro sets) and `outputs/crude_v0_2_1_20261010.mp4` (v0.2.1: + T/X subassemblies); both 540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p CRF 18, +faststart, no audio |
| Rebuild | `scripts/bslot.sh -b --python scripts/build_crude_film.py -- scenes/crude_v0_2.blend` (about 3 min; imports hero GLB, R rack blends, T/X GLBs) |
| Render | draft settings unchanged; one Blender process; about 2.5 min for the full film |
| Checks | `scripts/check_camera_clearance.py` every frame (3 mm): 0 real hits; 1 frame (S1 6.1 s) inside the bounding box of R's flap-vent frame while the camera is centred in the 0.7 x 0.5 m opening (box test of a ring, not a collision). Checker now tests collection instances with the instanced collection's bounding radius (racks). Stills every 0.5 s of every scene (5 rounds) + 1 fps sheets of both videos |

### 4.1 External assets integrated

| Asset | Workstream | Used in | How |
|---|---|---|---|
| `assets/hero/pigeon_osfp_dr8.glb` (polish pass, file time 01:34) | H | hero in every scene | imported under the hero rig root, rotated +90 deg about Z, hall scale 2.6; parts with centroid in the head (x > 30 mm, z > 68 mm in the bird frame) reparented to the neck pivot so the head turns; GLB leash hidden (replaced by a stretch leash from the tail to an animated loop); lids added at the GLB eye hooks; S5 explode uses each part's `explode` property (210 top-level parts) with ghost copies of the `skin_*` shells; S5 labels parented to GLB parts; contact glow on unique copies of the talon-tip (GND), ankle-pad (power) and toe edge-pad (signal) materials |
| `assets/components/datacenter/*` + `hall_layout.json` | R | hall S1-S6 | `ASSET_rack_gb300_nvl72` and `ASSET_rack_bird_q3400` collections appended and instanced at the 30 layout positions (10 bird racks = 1/3); 180 cells from the layout fill every bird rack top to bottom (crowd = 177 instances + hero + 2 companions); `ASSET_bird_flap_vent` at the layout entrance (0, 12.94, 3.25), curtain flap keyed open about local Y as the bird and camera pass; hero lands on the L07 landing pad from the layout. Layout shifted +1.0 m in y (LAYOUT_DY) so the hero rack L07 is at y = 7.0, keeping the v0.1 framing of S2-S5 |
| `assets/components/optics_tx/optics_tx.glb` (v1) | T | S3 macro | imported at x = 200 m, scaled x1000 (1 mm -> 1 m); camera keys retargeted to part centres from `optics_tx_parts.json` (lasers + Si lenses, PIC face with MZM layout texture, HS bonds, FAU collimator + staircase prism, 2x4 eye); T's `tx_beam_*` viz kept, plus 8 output beams from `lane_map`; beam note on screen |
| `assets/components/optics_rx/optics_rx.glb` (v1) | X | S4 macro | imported at x = 400 m, x1000, carrier near the floor; camera over lenslets -> interleaver plates -> comb mirror + PD array -> TIA with spiral inductors and bonds; 8 incoming + internal beams drawn by S (rows shifted -+0.25 mm onto the 0.5 mm PD pitch, from X's optics log); beam note on screen |

### 4.2 Deviations

- Leash and "pulled out backwards": birds sit facing out (needed for eye contact and the explaining shots), so the tail leash runs from the tail along the cell floor and hangs over the cell lip; the Subject pulls it and the hero comes out and dangles head-down by its tail leash before he turns it upright. A literal backwards pull would need birds sitting head-in (eyes inside the cage). Decision for the owner.
- Fill order: R's layout fills cells top to bottom by `fill_index`; v0.2 fills every bird rack completely (owner: "fill the bird rack from top to bottom"); only the hero's cell (L07, unit 1 lower row, middle) is empty until it docks.
- Eye axes are now purely lateral (hero GLB); one-eyed stares turn the head about 90 deg.
- RX internal beams are drawn by S from X's optics log, not taken from X's asset (no beam objects in the RX GLB).

## 4.3 v0.2.2 (coordinator pass, 2026-10-10)

| Item | Change |
|---|---|
| Video | `outputs/crude_v0_2_2_20261010.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p CRF 18, +faststart) |
| Crowd birds | Crude blob crowd replaced by a light proxy derived from the hero GLB at build time: shell, wings, tail fan, legs and feet, beak, eyes (bezels, rims, lenslets) kept; internals, screws, seams, prisms, thermal straps and band dropped; body and head joined separately and decimated (body 32,584 -> 5,865 faces, head 8,824 -> 3,969); one shared mesh each for 177 crowd + 17 street instances and the 2 companion rigs; head stays a separate instanced part on the neck pivot (head motion and eye-contact rig unchanged); tail strap kept as separate parts |
| S1 street grade | Darker sky (world 0.006/0.007/0.016), neutral moon and fills at about a third of v0.2.1, warm (not pink) lamp light, darker street, sidewalks and facade, fake wet-street streaks under the neon signs and warm lamp pools on the ground; the hero no longer reads pale |
| S4 RX | Top light 3000 -> 600 W, front fill 900 W (carrier and gold traces no longer clip); silica lenslet array shown with a lit glass material (EEVEE draft has no refraction); camera holds on the lenslet array 2.0-3.4 s from the side, clear of the incoming beams |
| S5 labels | Labels laid out in two columns beside the exploded hero (head parts right of the beak, body parts left), each with a leader line to its part (stretch-to strap from the column point to an anchor parented to the part, so it follows the explode); cameras pulled back so the columns stay in frame |
| S6 framing | Crane keys reworked to pass close to the scooter bird, the lamp-post pair (link), the wire birds and the rooftop uplink bird (30 mm lens, 1-2 m away); placement unchanged |
| S3/S4 retarget | Rebuild picks up the latest T (02:38) and X (02:51) GLBs and the latest hero GLB (02:54); T part positions unchanged, so S3 keys unchanged; X hook/part positions unchanged |

Open decisions for the owner (coordinator, 2026-10-10: keep for now, flag):
1. Leash staging: tail leash runs over the cell lip and the bird dangles by its tail when pulled (not a literal backwards pull).
2. Bird racks fully filled (only the hero's cell empty until it docks) instead of R's partial top-to-bottom fill order.

## 4.4 v0.2.3 (coordinator pass, 2026-10-10)

- Video `outputs/crude_v0_2_3_20261010.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p, +faststart); assets unchanged from v0.2.2 (hero 02:54, X 02:51, T 02:38).
- S5 top-left label "EXPLODED VIEW (HERO MODEL v1; POLISH IN PROGRESS)" -> "EXPLODED VIEW".
- S5 leader lines: 0.9 mm -> 0.35 mm wide, dimmer cyan-grey (emission 1.2) instead of near-white; they now read as thin lines.

## 5. Known issues

- S1 (fixed in v0.2.2): lilac street and pale hero.
- S1 tree: labels are about 15-20 px tall at 540 px; the left (screen-right) leaves are partly behind the hero during the rise.
- S4 (fixed in v0.2.2): carrier overexposure, short lenslet view. The RX incoming and internal beams are still drawn by S, not by X.
- S5 (fixed in v0.2.2/v0.2.3): label overlap, white-bar leaders; the wing and feet leaders still run a long way across the frame.
- S6: birds read better at the reframed keys but are still small in transitions; flying birds do not flap (FX label).
- Proxy crowd: decimation softens the eye lenslets; heads of turning crowd birds can intersect the cell walls (no collision test).
- Crowd straps: every cell bird has a hanging yellow pull-tab loop; busy at a distance.
- No collision test between heads/beaks of turning crowd birds and the cell walls (visual only).

## 6. TODO

- [x] Freeze v0.1 (`scripts/build_crude_film_v0_1.py`)
- [x] Storyboard rev 3 (this file)
- [x] Hall layout (1/3 bird racks, full-height cells), landing pads, facade with door and bird flap (R assets)
- [x] Street set shared by S1 and S6 with natural objects; tree hologram (no years, numbered lineage)
- [x] Crowd head motion + eye contacts; ambient beams removed; beam note
- [x] Hero GLB integration (head pivot, lids, tail stretch leash, GLB explode in S5)
- [x] S1 opening, S2 tail-leash pull, S3/S4 TX -> RX switch, S6 casual perches
- [x] Verify: clearance check, stills, contact sheets; full render (one process); encode; 1 fps sheet
- [x] v0.2.1: T / X subassemblies in S3/S4
- [x] v0.2.2: proxy crowd from the hero GLB, S1 grade, S4 exposure + lenslet hold, S5 label columns with leaders, S6 reframing
- [ ] Owner decisions: leash staging and full racks (Section 4.3)
- [ ] Re-integrate after the R / T / X / H audits (rebuild picks up new files automatically; camera keys in S3/S4 assume the v1 part positions)

## 7. Progress log

- 2026-10-10 01:23: DevLog-005 read; v0.1 frozen; hero GLB inspected (165 objects, 232,878 tris, -Y beak, eyes at (-+12.78, -47.5, 80.5) mm, pure lateral; leash behind the tail); R / T / X assets not yet present.
- 2026-10-10 01:40: storyboard rev 3 written.
- 2026-10-10 01:55: v0.2 script assembled (placeholder racks, street, tree, heads, hero GLB, stretch leash); first build OK (3 min), clearance clean.
- 2026-10-10 01:59: stills round 1: tree lineage zigzag lines cluttered (replaced by numbered gold/magenta branches), beam note too small, street too bright, S6 perches not framed. Fixed.
- 2026-10-10 02:03: R racks delivered (coordinator message); layout, rack instances, flap vent and cells integrated (layout +1.0 m in y); clearance checker extended to rack instances; stills round 3 OK.
- 2026-10-10 02:09: v0.2 full render + encode (`outputs/crude_v0_2_20261010.mp4`); 1 fps sheets checked; frames removed; script frozen as `build_crude_film_v0_2.py`.
- 2026-10-10 02:13: T and X v1 delivered; S3/S4 macro sets replaced by the subassemblies (x1000), camera keys retargeted; RX camera moved over the TIA after stills.
- 2026-10-10 02:23: v0.2.1 full render + encode (`outputs/crude_v0_2_1_20261010.mp4`); 1 fps sheets checked; frames removed.
- 2026-10-10 02:35: v0.2.2 started (coordinator review); v0.2.1 frozen as `build_crude_film_v0_2_1.py`.
- 2026-10-10 02:45: proxy crowd, street grade, RX exposure, S5 label columns, S6 keys implemented; build OK (45 s); stills every 0.5 s: label columns cut at the frame edge, RX lenslets invisible (refraction), S6 wire box flagged. Fixed (columns moved in, cameras pulled back, lit glass, extra crane key).
- 2026-10-10 02:55: rebuilt with the latest T / X / hero GLBs; clearance: only the known vent-frame box frame (S1 6.1 s); stills every second checked.
- 2026-10-10 03:00: v0.2.2 full render (one process, 0 errors), encoded `outputs/crude_v0_2_2_20261010.mp4`; 1 fps sheets checked; frames removed.
- 2026-10-10 03:11: v0.2.3: S5 label text and leader lines fixed; rebuilt, full render (one process, 0 errors), encoded `outputs/crude_v0_2_3_20261010.mp4`; S5 1 fps sheet checked; frames removed.
