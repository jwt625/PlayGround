# DevLog-002: Storyboard rev 2 (fiber-free birds) and crude v0 build

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Status | Rev 2 storyboard locked for the crude build; crude v0 in progress |
| Depends on | `DevLog-001-mapping-and-storyboard-v1.md` (rev 1 mapping, tree, audit), `research/*.md` |
| Parallel work | Hero GLB model by subagents: `DevLog-003-hero-pigeon-model.md`, `research/hero_component_dimensions.md` |
| Authors | Claude, at Wentao Jiang's request |

## 1. Wentao's feedback on rev 1 (verbatim, 2026-10-09)

```text
feedback:
1. do not mention the word "pluggables", birds = pluggables. Narrators should just say birds. Come up with surveillance related acronym for OSFP and DR, like Surveillance Feathered Platform for SFP, and O could be Overhead.
2. sixteen fibers from the beak does not make sense. it is FSO with its eyes. Come up with different functional mapping for the beak.
3. the eyes' fiber arrangement should be a CBC-like array, hence each collimated separately. Or maybe we should add an additional mux (i.e. more complex than the actual DR8) so that it is a single spatial channel
4. there should not be any fiber involved. The hatched by outnolight shot could have the background scene fade in and out the factory/hatchery showing short animation of the bird getting assembled on an assembly line.
5. adapt S6, there should be no fiber to begin with. Or add it to be talking to satelite and actual cameras on the street in addition to between them.
Update the devlog accordingly, and start drafting crude models, scenes and camera movements and render a v0 for me to review and give further feedback. i.e., follow the same workflow and progression as done in the interconnect film project.
```

Follow-up (same day): "at the same time, also spin off subagents to build the hero model of the OSFP 800G DR8 pigeon's 3d blender model. This can be done in parallel as the scene draft. Ask them to use or build components as close to actual images and pictures as possible. I have a teardown blog in jwt625.github.io repo with images of a 800G pluggable. And/or look up dimensions of core components like the dfb laser dies, isolators, FAUs etc. Deliverable should be a glb file, a picture of it with its shell half transparent, and an explosion view of its interiors and components."

## 2. Rev 2 decisions

| # | Decision |
|---|---|
| R1 | The narrator never says "pluggable(s)"; they are birds. Real acronyms stay on labels; the narrator expands them as surveillance backronyms (table below) |
| R2 | Beak = retention latch (stylized): it bites the cage lip to hold the bird in place; pulling the leash (pull tab) opens the beak to release. The real OSFP retention/delatch mechanism is simplified (standard, not in H) |
| R3 | Eyes = free-space lenslet arrays, CBC-like tiled apertures: each eye is a 2x4 microlens array, one lens per lane, each lane collimated separately. Chosen over a WDM mux because a single spatial channel would need 8 wavelengths, which changes the architecture away from DR8 (8 parallel lanes at 1311 nm). A small prism block in each eye rearranges the chip edge row (1x8) into the 2x4 grid (stylized) |
| R4 | No fiber anywhere. The eye positions are where the MPO's TX and RX halves would be (TX = pigeon's right eye, RX = left eye). Disclaimer: real DR8 uses 16 fibers through an MPO connector |
| R5 | Hatchery: during "Hatched by Out-no-light" the hall fades to an assembly line (PCB in, DSP placed, optics placed, shell and feathers closed, eyes light up, bird hops off) and back |
| R6 | S6 has no fiber: the Subject walks out to a night street; the birds follow and form the 3-bird free-space ring around him, plus links up to a satellite and to street CCTV cameras |
| R7 | The data hall is crossed by free-space beams between birds from S1 on (cyberpunk laser web) |

### Surveillance backronyms

| Acronym | Narrated as | Used |
|---|---|---|
| OSFP | Overhead Surveillance Feathered Platform | VO S1 |
| DR8 | Drone Recon, 8 lanes | label |
| SFP | Surveillance Feathered Platform | tree label (later shorts) |
| QSFP | Quad Surveillance Feathered Platform | tree |
| QSFP-DD | Quad Surveillance Feathered Platform, Deep Disguise | tree (hummingbird) |
| CFP | Covert Feathered Platform | tree |
| GBIC | Ground-Based Intelligence Collector | tree |
| XPO | eXtreme Payload Ostrich | tree |
| AEC | Avian Egg Counterfeit | tree (cuckoo) |
| ZR | Zone Recon | tree (albatross) |
| CPO | Caged Pigeon Operations | tree (dodo ghost) |

## 3. Mapping changes vs rev 1 (DevLog-001 Section 2)

| Part | Rev 1 | Rev 2 |
|---|---|---|
| Beak | MPO-16 APC, 16 fiber teeth | Retention latch; bites the cage lip; leash opens it |
| Beak muzzle / dust cap | Dust cap on the MPO | Eye caps: one cap per eye, popped off in S2 |
| Right eye (TX) front | Fiber array unit, 8 fibers | 2x4 lenslet array + prism block (1x8 edge row to 2x4); 8 collimated beams |
| Left eye (RX) front | Fiber array / lens | 2x4 lenslet array + prism block focusing onto the 8-ch PIN array |
| Links | Fibers between modules | Free-space beams eye to eye (TX right eye to the partner's RX left eye) |
Everything else (lasers, MZMs, monitor PDs, PIN array, TIA, DSP, MCU, EEPROM, crystal, power, 60-pad feet, I2C spine, feathers, EMI ruff, leash) as rev 1.

## 4. Storyboard rev 2 (60 s + 2 s)

Camera: one fluid path; scale jumps inside the eyes use "fill" transitions (the lens glow fills the frame, the macro set emerges from the same color), never scope transitions or hard cuts.

### S1 Tree of life to dovecote (0-10)

| t | Action | Camera |
|---|---|---|
| 0.0-4.5 | Hologram tree rises in the data hall aisle: root theropod (1x9), Archaeopteryx (GBIC), moa (XENPAK), elephant bird (CFP), emu (CFP2-DCO), ostrich (XPO), chicken (SFP), hummingbird (QSFP-DD), albatross (400ZR), parrot (compatibles), cuckoo (AEC), passenger pigeon (QSFP+), rock pigeon (QSFP28, RFC 1149 plaque), dodo ghost (CPO), cyber pigeon node on top | Rise up the trunk, leaves passing close |
| 4.5-6.5 | Dive from the cyber-pigeon node down to the hero's pigeonhole in the left rack; the tree fades | Swooping dive |
| 6.5-8.0 | Dolly along the faceplate: rows of birds, red right eyes; free-space beams criss-cross the aisle | Dolly |
| 8.0-10.0 | The Subject walks down the aisle; every head tracks him; settle on the hero | Settle close on the hero |

VO (20): "Birds evolved from dinosaurs. Or so they told you. Meet the O-S-F-P: Overhead Surveillance Feathered Platform. It is watching you."
Labels: leaf names + years; "800G OSFP DR8"; "DR8: DRONE RECON, 8 LANES".

### S2 Outside: beak latch, eyes, feet (10-20)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | The Subject pulls the hero out by the leash; the beak opens to release | Orbit as it comes out |
| 2.0-4.5 | Front close-up: eye caps pop off; two 2x4 lenslet eyes (right red TX, left cyan RX); the beak snaps (latch) | Push onto the face |
| 4.5-7.0 | Under the body: 60-pad feet, right foot TX, left foot RX, low-speed and power pads in the middle | Swing under the belly |
| 7.0-10.0 | Plugged back in (cutaway): ground claws touch first, then power, then signal (glow pulses); the beak bites the cage lip | Low macro inside the cage |

VO (22): "Feathers: heatsink. Beak: a latch that bites the cage. Right foot and eye transmit. Left foot and eye receive. Ground lands first."

### S3 Right eye: transmit (20-30)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | Push into the red right eye; red fill | Push |
| 2.0-4.5 | Macro set: fly back along the waveguides to the 2 CW DFB lasers, then forward with the light through the splitter tree | Fly back, then forward |
| 4.5-7.0 | 8 Mach-Zehnder modulators blink PAM4; driver traces from the DSP twitch; monitor PDs wink | Glide over the MZMs |
| 7.0-8.6 | Light leaves the chip edge, the prism block regroups it, 8 lenslets launch 8 collimated beams | Follow the light out |
| 8.6-10.0 | Back in the hall: ride along the 8-beam bundle across the aisle | Beam ride |

VO (18): "Right eye: transmit. Two D-F-B lasers glow deep in its skull. Eight Mach-Zehnder modulators. Eight lenses. Eight beams."

### S4 Left eye: receive (30-40)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | The beams enter the partner bird's cyan left eye; cyan fill | Ride continues |
| 2.0-4.5 | Macro set: 8 lenslets focus through the prism onto 8 PIN photodiodes, each flashing | Over the PD array |
| 4.5-6.5 | The octal TIA amplifies; signals rush toward the DSP | Follow signals |
| 6.5-10.0 | Bird POV in the hall: wide view with a surveillance HUD; the Subject boxed and tracked | POV push |

VO (20): "Left eye: receive. Eight lenses, eight photodiodes. A T-I-A, a transimpedance amplifier, turns tiny currents into voltage. It sees you."

### S5 Exploded view and hatchery (40-50)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | Pull out of the partner's eye, whip back along the beams to the hero; the hero floats out; parts separate | Pull-out, whip, rise to 3/4 |
| 2.0-4.5 | DSP brain under the thermal pad; stamp flicks NARROWCOM / MERVALL / MINLINEAR | Push to DSP |
| 4.5-6.5 | MCU, EEPROM, crystal ticking, power parts | Arc across the board |
| 6.5-8.0 | I2C spine lights and streams telemetry fields to the switch (temperature, voltage, laser bias, TX power, RX power, then "SUBJECT: HEART RATE ELEVATED"; no numbers) | Follow the stream |
| 8.0-10.0 | Parts snap back; leg band "HATCHED BY OUTNOLIGHT"; the hall fades to the hatchery assembly line (PCB, DSP, optics, shell, eyes on, bird hops off), then back | Hold on the hero |

VO (20): "The brain: a D-S-P from Narrow-com or Mer-vall. A microcontroller reports temperature, laser current and light levels. Hatched by Out-no-light."

### S6 Street: ring, satellite, cameras (50-60)

| t | Action | Camera |
|---|---|---|
| 0.0-2.0 | The Subject walks out of the hall door into a neon night street; three birds follow | Follow behind him through the door |
| 2.0-5.0 | The birds land on three lamp posts in a triangle around him | Rising, circling crane |
| 5.0-8.0 | Free-space ring: each right (TX) eye beams into the next bird's left (RX) eye; one bird uplinks to a satellite; street CCTV cameras link to the birds | Top-down, then descend into the ring |
| 8.0-10.0 | The Subject freezes; heads rotate to camera; the hero blinks (eyelid = TX disable) | Snap push to the hero's eye |

VO (21): "They follow you outside. Eye to eye, bird to bird. Up to the satellites. Down to every camera on the street."

### S7 Disclaimer (60-62)

Card: "Pigeons are real. Parody; not affiliated with any company or bird. One typical 800G DR8 build shown. Real DR8 modules use 16 fibers through an MPO connector; here the eyes do free-space optics with stylized lenslet arrays. Modules cannot see. 1311 nm beams drawn visible. Real pigeons have 4 toes." + sources. VO (6): "Parody. Typical build. Not real birds."

## 5. Crude v0 build plan

Same workflow as the interconnect film's crude builds:
- `scripts/build_crude_film.py` (scenes) + `scripts/blender_lib.py` (copied from the interconnect film: primitives, keyed visibility, camera-attached overlay text, narration chunks) + new helpers for the continuous camera (Bezier keys), fill transitions and the crude bird.
- One shared data-hall set (S1-S6 start) in world coordinates so the camera stays continuous; macro sets for the TX eye (S3) and RX eye (S4) far away, entered through color fills; the street set beyond the hall's end door; the hatchery fades in behind the hero.
- Crude bird = primitives (body box with fins, head, two 2x4 lenslet eyes, beak halves, leash, feet with toes); crowd birds as collection instances (body + head, heads keyed to track the Subject). The detailed hero GLB from DevLog-003 replaces it after review.
- Text: narration captions (white, lower third), labels (cyan, top-left), `[FX: ...]` notes where effects will go, time code `S# m:ss` for timestamped feedback.
- Render: EEVEE, 720x900 (4:5), 12 samples, 30 fps, 1860 frames; PNG frames to `outputs/frames_v0/` (removed after encoding), H.264 yuv420p `+faststart`, no audio: `outputs/crude_v0_<date>.mp4`.

## 6. Crude v0 as built

| Item | Value |
|---|---|
| Generator | `scripts/build_crude_film.py` + `scripts/blender_lib.py` (copied from the interconnect film); rebuild: `scripts/bslot.sh -b --python scripts/build_crude_film.py -- scenes/crude_v0.blend` (about 1 min) |
| Stills | `scripts/render_frames.py` (selected frames to PNG, for checks) |
| Blend | `scenes/crude_v0.blend` (git-ignored) |
| Render | Draft speed (Wentao, 2026-10-09: fastest settings for iteration): EEVEE Next, 1 sample, 540x674, shadows off, persistent data; about 0.12 s/frame, full film about 2 min in two parallel chunks through `bslot.sh` (was 12 samples at 720x900, about 1 s/frame) |
| Video | `outputs/crude_v0_20261009.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p, +faststart, no audio, 16 MB) |

How it is built:
- One camera with a Track-To target, Bezier keys through every scene; scene boundaries share the same camera pose (no cuts). Scale jumps into the eyes are teleports hidden under a full-frame color fill (red for TX, cyan for RX); S6 to S7 is the only hard cut (disclaimer card).
- Shared data hall in absolute coordinates: 2 x 13 racks, 9 pigeonholes each; 230 crowd birds as collection instances (body + head), heads keyed to track the Subject; 18 cross-aisle beams from top-row right (TX) eyes to left (RX) eyes.
- Full rigs: hero (with crude internals and explode groups), partner (link target across the aisle), two companions (fly out in S6). Hero eyes go dark when unplugged (no power) and light up after the signal contacts mate.
- Macro sets: TX eye at x ~ 1000 (2 DFBs, splitter tree, 8 MZMs with gold electrodes, PAM4-keyed output brightness, monitor PDs, driver wires from the DSP, edge couplers, prism, 2x4 lenslets, 8 beams); RX eye at x ~ 2000 (lenslets, prism, 8 PDs flashing, wire bonds, octal TIA with bumps, traces to the DSP).
- Hatchery: assembly line with robots and items gaining parts station by station, faded in by material alpha behind the hero in S5 8.2-9.95.
- Street: wet ground, buildings with neon, three lamp posts in a triangle around the Subject, three CCTV cameras, a moving satellite, stars.

Deviations from rev 2 decided while building:
- Eye caps dropped: unplugged birds have no power, so the eyes go dark instead (and light again after the signal contacts mate in S2). Lenslets stay visible as dark glass.
- Known v0 issues: in S3 the lenslets sit behind the prism from the camera's path, so the beams read but the lenses barely do; tree labels are small at 540 px; S4 0-2 s the camera flies through the beam bundle; S6 street is still dark.
- Hero internals are crude placeholders positioned by eye; the detailed hero GLB (DevLog-003, using `research/hero_component_dimensions.md`) replaces them after review.
- S4 point of view is a 16-22 mm lens with a red HUD box, not a true 340-degree fisheye (EEVEE limitation; labelled FX).
- No volumetric haze, bloom, flapping or rain (labelled FX).

Checks done: 3 rounds of test stills (32 + 15 + 3 frames) through contact sheets. Fixed from them: S2 face framing (both eyes), belly shot on the feet, cutaway lighting and angle (cage floor removed), S4 POV camera that sat inside the rack, S5 exploded framing and the Subject occluding it, street lighting, hero dark-eye material.

## 7. TODO

- [x] Rev 2 storyboard (this file)
- [x] Crude v0 build script and blend
- [x] Render and encode crude v0; 1 fps contact-sheet check (S3 8.0 s camera passed through the prism block; rerouted over it and re-rendered frames 805-880)
- [ ] Wentao reviews crude v0
- [ ] Hero GLB integration (after DevLog-003 delivers)

## 8. Progress log

- 2026-10-09: rev 2 written from Wentao's 5-point feedback; hero model and dimension agents launched in parallel.
- 2026-10-09: crude v0 built and rendered (`outputs/crude_v0_20261009.mp4`); switched to fastest draft render settings on Wentao's request; frames removed after encoding.
- 2026-10-09: Wentao on the hero model (verbatim): "why is the hero model not looking like a pigeon? is this a dummy/comparison model? This is nonsense. It should look like a pigeon, but with various components themselves faithfully modeled. They should get assembled into a pigeon shape, and then wired properly." Cause: the coordinator's brief required the OSFP box envelope; corrected (pigeon silhouette, true-mm organs, lateral eyes, 4 toes x 2 lanes, wiring). Then: "the exploded png looks great. Now assemble it into actual pigeon shape please." Then: "iterate and polish the flex cable routing and the pins on the feet, should match the actual contact counts of the OSFP pluggable. Also remember you should have subagents working on this or working on the scene and camera shots in parallel.."
- 2026-10-09: parallel work: hero agent (exact 60-contact OSFP pin map on the feet, flex routing polish; DevLog-003) and scene agent (crude v0.1: pigeon-shaped birds, lateral-eye beam geometry, v0 known issues, camera composition; DevLog-004). Coordinator reviews both.
