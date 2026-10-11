# DevLog-004: Crude v0.1 (pigeon-shaped birds, lateral eyes, v0 fixes)

| Field | Value |
|---|---|
| Date | 2026-10-09 (finished 2026-10-10) |
| Status | Crude v0.1 rendered; waiting for Wentao's review |
| Depends on | `DevLog-002-storyboard-rev2-and-crude-v0.md` (rev 2 storyboard, v0 build, known issues), `DevLog-001` sections 2-4 (mapping), hero shape from `outputs/hero/pigeon_osfp_dr8_profile.png` (DevLog-003, read-only here) |
| Generator | `scripts/build_crude_film.py` (v0.1); v0 frozen as `scripts/build_crude_film_v0.py` |
| Authors | Claude (scene/camera subagent), at Wentao Jiang's request |

## 1. Brief (from the coordinator, 2026-10-09)

1. Birds pigeon-shaped (box birds rejected): primitive silhouette with the hero's proportions (small round head, short beak with cere, iridescent neck, round breast, folded fin-feather wings, tail fan, legs, 4-toe feet), gunmetal cyberpunk drone look, glowing eyes; one hall scale factor for every bird; faceplate cells resized to fit.
2. Lateral eyes: right eye = TX (red 2x4), left eye = RX (cyan 2x4). Hall: one-eyed stares, cross-aisle beams from a TX right eye to an RX left eye; S3->S4 8-beam bundle likewise; S6 ring with birds perched around the Subject so each right eye points at the next bird's left eye; keep satellite uplink and CCTV links. Beak = latch (bites the cage lip in S2). Feet = edge contacts, 4 toes per foot, gold pads; S2 shows mating GND -> power -> signal.
3. Fix DevLog-002 section 6 issues (S3 lenslets hidden by the prism, small tree labels, S4 camera through the beam bundle, dark S6 street); re-check S2 framing (head turns to present each eye) and S5 exploded view (placeholder organs inside the pigeon silhouette).
4. One fluid camera (Bezier keys, color fills only for eye scale jumps, hard cut only to S7); subjects in the central 1:1 square; eased moves; no camera through geometry.
5. Captions, labels, FX notes and time code as in v0.
6. Verify with stills at every second, then one full render (one Blender process), encode `outputs/crude_v0_1_20261009.mp4`, 1 fps contact sheet check.

## 2. As built

| Item | Value |
|---|---|
| Rebuild | `scripts/bslot.sh -b --python scripts/build_crude_film.py -- scenes/crude_v0_1.blend` (about 100 s) |
| Camera check | `scripts/bslot.sh -b scenes/crude_v0_1.blend --python scripts/check_camera_clearance.py -- 1 0.003` (new; about 10 s): camera position against every visible mesh/curve bounding box, every frame, 3 mm margin; crowd instances checked per instanced part |
| Stills | `scripts/render_frames.py` (unchanged) |
| Render | Draft settings unchanged (EEVEE Next, 1 sample, 540x674, shadows off, persistent data); one Blender process, `-a` to `outputs/frames_v0_1/` (removed after encoding) |
| Video | `outputs/crude_v0_1_20261009.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p CRF 18, +faststart, no audio, 26 MB); full render about 5 min in one process |

### 2.1 Bird primitive (all birds)

- Modelled in millimetres in the hero's frame (hero profile render, 124 mm beak to tail): origin on the ground between the feet, +x = beak, +z up, -y = pigeon's right. Head centre 44 mm ahead of the feet, 81.5 mm up, radius 12 mm.
- Parts: tilted ellipsoid body, round breast, iridescent neck (thin-film metallic, green over purple), round gunmetal head, short dark-chrome beak (upper mandible + hinged lower mandible), off-white cere, folded wings (covert plate + 5 fin feathers + a magenta glow strip), 7-blade tail fan with a dark terminal band, red-metal legs, 4 toes per foot (hallux back, inner/middle/outer forward; angles and lengths from the hero script's TOES table), yellow pull-tab leash with loop under the chest, orange leg band on the right leg.
- Contacts: toe pads (gold, signal) at z = 0, palm pads (gold, power) 1.2 mm lower, claws (brass, GND) 3 mm lower, so a bird lowered onto a flat contact touches GND, then power, then signal. 4 low-speed pads on the belly between the feet.
- Eyes: lateral, axis 20 deg forward of pure lateral (stylized assumption, not on screen). Each eye: coloured rim (red TX / cyan RX, visible even unpowered), black iris, 2x4 lenslets (4 along the head, 2 high), eyelid (scale-keyed disc).
- Scale: one uniform `S_BIRD = 2.6` for every bird (hero, partner, companions, 230 crowd instances, 8 ring birds, 3 lamp birds): 0.32 m beak to tail, about 0.25 m tall on screen.
- Hall: racks now have open pigeonhole cells (0.20 m wide x 0.27 m high x 0.32 m deep; shelves and dividers instead of black hole plates); birds stand 6 cm inside the rack face so head, neck and breast poke out. Ceiling raised from 3.6 m to 4.2 m so the tree's top node and the dive fit under it.
- Hero extras (S5 only): ghost shell (blended copy of body, breast, neck and skull at alpha 0.16), head board + DSP + thermal pad (brain) and MCU in the head, TX PIC with 8 MZM strips + 2 DFBs behind the right eye, PD array + TIA behind the left eye, prism blocks behind both lenslet arrays, sternum board with PMIC, inductor, load switch, EEPROM, ticking crystal and MLCCs in the chest, flex ribbons as curves from the head down the neck into both legs, plus a power/I2C ribbon to the belly pads. Explode: skull lifts, wings and tail move back and up, legs/feet drop; organs stay inside the silhouette.

### 2.2 How the lateral-eye beams are computed

1. Eye position in world: `p = pos + Rz(yaw) Ry(pitch) [S_BIRD * (P_H + Rz(h) e_k)]`, with `P_H` the neck pivot (36, 0, 70) mm, `h` the head yaw about the neck, `e_k` the eye centre in head coordinates (`k` = tx right, rx left).
2. Aim one eye at a target (`aim()`): the eye axis bearing in the head frame is -70 deg (right eye) or +70 deg (left eye); `h = atan2(target - p_eye) - yaw - bearing`. Because the eye moves when the head turns, this is iterated 5 times (converges to well below 0.1 deg).
3. A link (`solve_link()`): alternately aim the TX bird's right eye at the RX bird's left eye and vice versa, 4 rounds. Both birds then hold that head yaw (one-eyed stare across the aisle).
4. Hall web: 18 links between cells in the top row and in the bottom row (TX side alternates L/R, RX partner up to 2 racks away), so the camera corridor at 1.3-1.7 m stays free of beams. The 36 link birds hold the link stare from S1 6.4 s; the beams switch on one by one from 6.4 s. The other 194 crowd birds track the Subject with whichever eye needs the smaller head turn (clamped at 1.9 rad), after random idle head twitches.
5. Hero to partner bundle: the hero turns its head at S3 0-0.8 s to the solved link yaw (the partner turns too); 8 parallel beams, each from a TX lenslet's world position to the same position shifted by (RX eye centre - TX eye centre), so the 2x4 grid lands on the facing 2x4 grid.
6. S6 ring: 8 perches on a 2.2 m circle around the Subject (rotated 22.5 deg so he walks in between two perches). Each bird faces the Subject; its right eye then looks along the ring to the counter-clockwise neighbour's left eye. For 8 birds the neighbour bearing is 67.5 deg from the beak, close to the 70 deg eye axis, so one head yaw serves both eyes (solver splits the small difference). With 3 birds the bearing would be 30 deg, which lateral eyes cannot serve; that is why the ring has 8 birds (3 flying in: hero + 2 companions; 5 already waiting).
7. Uplink and CCTV: 3 lamp-post birds outside the ring. Lamp bird 0 rolls its head so its right eye tilts up to the satellite (uplink); CCTV cameras feed the left eyes of lamp birds 1, 2 and 0.

### 2.3 Scene and camera changes vs v0

| Scene | Change |
|---|---|
| All | Pigeon birds; open cells; overlay plane moved from 0.8 m to 6 cm in front of the lens (v0 labels were hidden whenever the camera was within 0.8 m of a wall); macro sets moved from x = 1000 / 2000 to x = 200 / 400 (at 1000-2000 m float32 precision garbled the overlay text once the plane was close to the lens); hall lights rebalanced (dark gunmetal birds, neon accents) |
| S1 | Tree labels 0.062 m (were 0.042) with fewer words (extinct = gray, one "GRAY = EXTINCT" legend), camera 1.15 m from the tree (was 1.45 m); clade names written vertically along the trunk; dive passes above the node (v0 passed through the trunk top); beams switch on as the link birds turn their heads; crowd heads do one-eyed stares at the Subject |
| S2 | Hero head returns forward, then turns to present the right (red rim) and then the left (cyan rim) eye to the camera; beak snaps; belly shot tilted nose-up; cutaway: hero lowers 12 mm onto a gold host pad on a green host board; claws (GND), palms (power), toe pads (signal) glow in that order with numbered labels; beak closes on a cage-lip bar; whole hero rack (cells and neighbour instances) hidden during the cutaway |
| S3 | Hero turns its head to the link stare; camera pushes into the red eye along the link axis from the aisle; macro: lenslets now red-emissive with rims, 1x8 row beams into the prism and 2x4 beams out of it; camera ends on a 3/4 front view of the lens array (8.1-8.4 s) before the red fill; back in the hall it rides 7.5 cm beside the bundle |
| S4 | Rides beside the bundle and enters the cyan eye 32 deg off-axis (v0 flew through the bundle); RX macro path moved off the prism; partner turns its left eye to the Subject for the POV, then back to the link before S5 |
| S5 | Exploded view inside the silhouette (ghost shell); larger labels (0.017 m), fewer and spread out; camera pushes to the head (DSP), then arcs to the chest |
| S6 | 8-bird ring + 3 lamp birds; street brighter (sidewalks, lane marks, lit windows, larger neon, area fill, neon spill lights, stronger moon) but still night; flight path lowered to pass under the door top; crane routed through the door (v0 path would have clipped the end wall); heads turn one eye into the lens; hero blinks the right (TX) eye; push into it |

## 3. Deviations and open points

- Leash: the hero model (DevLog-003) hangs the pull tab under the tail; v0.1 keeps it under the chest so the Subject can pull it from the aisle. Needs a decision before hero integration.
- Cage lip: exists only on the hero's cutaway cage (a bar in front of the cell at beak height). Crowd cells have no lip, and birds that hold a link stare have their beak turned away from where a lip would be.
- Eye axis 20 deg forward of lateral is a stylization; real pigeon eye-axis angle not in `research/birds.md` (not on screen).
- Uplink bird uses the same right eye as the ring would; lamp birds are separate from the ring to keep one link per eye where possible (lamp bird 0 uses its right eye for the uplink and its left eye for a CCTV feed).
- Clearance check is a bounding-box test (conservative for spheres and cones); the camera still gets within 2-3 cm of the eyes in the push-ins by design.

## 4. Remaining issues (seen in the v0.1 stills)

- S1 5.0-6.0 s dive: frames look down at rack tops for about 0.5 s (dull).
- S5 exploded labels still overlap in places (DSP / TX PIC / MCU around the head at 3-5 s); wings and tail ghost shapes partly cover the crowd behind.
- S6 3.5-5 s crane frames are steep and tilted-looking (24 mm from high up); ring birds are small until 6 s.
- S6 flying birds read pale (specular from the area fill on the coat) and there is no flapping (FX label).
- S2 belly shot: legs and toes read, but the belly pads are tiny.
- Hall still uses the Standard view transform with no bloom/haze, so beams are thin lines (FX label as in v0).

## 5. TODO

- [x] Freeze v0 script (`scripts/build_crude_film_v0.py`)
- [x] Pigeon bird primitive + hero internals inside the silhouette
- [x] Lateral-eye beam solver; hall links, S3/S4 bundle, S6 ring
- [x] Scene and camera fixes S1-S6
- [x] Camera clearance script; stills every 0.5 s (4 rounds), contact sheets, fixes
- [x] Full render, encode, 1 fps contact sheet
- [ ] Wentao reviews v0.1
- [ ] Decide leash position (chest vs under the tail) before hero GLB integration
- [ ] Hero GLB integration (after DevLog-003 delivers), same `S_BIRD` and hooks

## 6. Progress log

- 2026-10-09 23:26: read DevLog-001/002/003, v0 script, v0 film contact sheets, hero profile renders; v0 script frozen; plan written.
- 2026-10-09 23:40: v0.1 script written (pigeon primitive, open cells, beam solver, 8-bird ring, scene and camera rework); first build OK.
- 2026-10-09 23:42: stills round 1 (120 frames): hall overexposed by the face-fill lights, hero white in S5, S6 birds clip the wall above the door. Fixed.
- 2026-10-09 23:45: stills round 2: lighting OK; S3 lenslets still weak, RX labels in the camera path, S2 contact labels cut off, leash too long. Fixed.
- 2026-10-09 23:53: `check_camera_clearance.py` written; first run flagged the S6 crane through the end wall (rerouted through the door) and false hits on crowd instances (stale matrices and margin in local units; fixed in the checker).
- 2026-10-10 00:01: clearance clean at every frame; overlay plane moved to 1.2 cm (labels hidden near walls in v0 and round 3); spot stills OK.
- 2026-10-10 00:08: final build, clearance clean (0 hits, every frame, 3 mm); full render started (one process).
- 2026-10-10 00:11: first full render + encode; 1 fps sheet showed garbled overlay text in the S3/S4 macro sets (float32 precision at x = 1000-2000 with the 1.2 cm overlay plane). Macro sets moved to x = 200 / 400, overlay plane to 6 cm; spot stills clean; clearance clean.
- 2026-10-10 00:18: final full render (one process, 1860 frames, 0 errors), encoded `outputs/crude_v0_1_20261009.mp4` (540x674, 30 fps, 62.0 s, yuv420p, +faststart); 1 fps contact sheets checked; PNG frames removed.

## Owner decisions after v0.1 (2026-10-10)

- Leash (pull tab) comes out of the TAIL, not the chest. Verbatim: "why is the pull tab coming out of the front? it is more natural coming out from the tail". Stylized vs the real OSFP (pull tab at the front/optical end). v0.2 staging: the birds still face out of their pigeonholes (tail inside); in S2 the Subject reaches into the hole and pulls the bird out backwards by its tail leash. The hero model keeps its tail leash (DevLog-003).
