# DevLog-001: Narrowed mapping, assumptions, bird tree, storyboard v1 (short 1)

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Status | v1 for Wentao's review; nothing built |
| Depends on | `DevLog-000-pigeon-pluggable-kickoff.md` (instructions, decisions D1-D12, parody names), `research/*.md` |
| Authors | Claude, at Wentao Jiang's request |

Every technical number on screen traces to `research/hero_800g_osfp_dr8.md` (H), `research/form_factors_and_companies.md` (F) or `research/birds.md` (B). "Typical" = one architecture chosen among real variants; the disclaimer says so.

## 1. Hero: 800G OSFP DR8, typical silicon-photonics build

| Spec | Value | Ref |
|---|---|---|
| Data | 8 x 100G PAM4 each way (host 8 x 106.25G, 53.125 GBd) | H5 |
| Optics | 1311 nm, single-mode, 500 m, one MPO-16 APC (Tx1-8 left, Rx8-1 right, seen from the front) | H3 |
| Body | 22.58 x 13.00 x 100.4 mm, closed-top integrated finned heatsink | H7 |
| Power | max about 16-16.5 W (datasheets); OSFP class 8 is >14 W | H7 |
| Edge | 60 pads: 8 TX pairs, 8 RX pairs, 4 low-speed, 4 VCC, 20 GND; GND mates first, then power, then signal | H7 |

## 2. Narrowed anatomy mapping (locked for v1)

Orientation rule (H7, derived from MSA figures, med-high): seen from the module front, TX is on the viewer's left at the MPO and at the card edge. The pigeon faces out of the faceplate toward the viewer, so **TX = the pigeon's right eye and right foot; RX = left eye and left foot**.

| Bird part | Component | Notes / ref |
|---|---|---|
| Head | Front third: optics | Real layout: optics near the receptacle, DSP mid, edge connector at the rear |
| Beak | MPO-16 APC receptacle; 16 fiber "teeth" in one row; the beak tip is cut at the APC 8 deg angle | H3 (8 deg: standard APC, not in H) |
| Beak muzzle | Dust cap | |
| Leash | Pull tab and latch | Handler's leash |
| Neck ruff | EMI fingers where the body meets the cage | H mechanics |
| Feathers | Integrated finned heatsink (closed top), 2-piece metal shell | H7 |
| Skull cap | Thermal gel/pad from the DSP to the lid | H mechanics |
| **Right eye (TX)**, outside to inside | Fiber array unit (8 fibers, 127 or 250 um pitch), edge couplers, TX silicon photonics PIC | H1-H3 |
| Right retina | 8 traveling-wave Mach-Zehnder modulators (MZMs) with phase-shifter bias | H2 |
| Back of the right eye socket | 2 x 1311 nm CW DFB lasers, uncooled, 70-100 mW, butt-coupled to the PIC (lensless per Credo/DustPhotonics; no isolator found in DR8 products; typical); 1:4 splitter tree per laser (inferred) | H1 |
| Eye in the back of the head | On-chip monitor photodiodes on the MZMs | H2 |
| Eye muscles | Driver integrated in the DSP (direct drive): the brain wires straight into the eye | H2 |
| **Left eye (RX)**, outside to inside | Fiber array / lens, then a separate 8-channel PIN photodiode array | H4 |
| Left retina | PIN photodiode array | H4 |
| Retinal ganglion cells | 8-channel TIA (500 um pitch, flip-chip) | H4 |
| Brain | PAM4 DSP, 8:8: CDR, equalization, line driver; one per pigeon, vendor stamp swaps in the exploded view (NARROWCOM / MERVALL / MINLINEAR) | H2, F-C |
| Hypothalamus | MCU: CMIS firmware, ADC for DDM (H6); laser bias control (standard, not in H) | H6 |
| Memory / leg band | EEPROM / memory (H6); ID, serial, vendor coding (standard, not in H); the leg band shows the module serial and hatchery (OUTNOLIGHT / OPTOELINK) | H6 |
| Eyelid | TX disable: a CMIS management control in OSFP (no dedicated TX-disable pin among the 4 low-speed pads; standard, not in H) | H7 |
| Heartbeat | DSP reference crystal | H6 |
| Heart and vessels | 3.3 V load switch, PMIC and buck rails (DSP core, TIA/laser bias) | H6 |
| Torso | Paddle-card PCB | H mechanics |
| **Right foot (TX)** | TX pad groups (8 pairs) on the right lateral half of the card edge; 8 cyber toes = 8 lanes (augmented: real pigeons have 4) | H7, B9 |
| **Left foot (RX)** | RX pad groups (8 pairs) on the left half | H7 |
| Talon order | 3 pad lengths: ground claws touch first, then power, then signal | H7 |
| Spinal cord to the handler | I2C (SCL, SDA) + INT/RSTn + LPWn/PRSn: the 4 low-speed pads sit between the two feet, in the middle of the edge | H7 |
| What it reports | Temperature, Vcc, per-lane TX bias, TX power, RX power | H6 |

Not in short 1 (reserved): VCSEL species with prism + thin-film-filter MUX/DEMUX and the oil-droplet gag (pigeon cones have long-pass oil droplets, B10); supply-chain corkboard; evolution march.

## 3. Assumptions and choices (label "typical" where marked)

| # | Assumption | Why / risk |
|---|---|---|
| A1 | SiPh TX with 2 CW DFBs (Credo/DustPhotonics style) | Real alternatives: 4 lasers (Lumentum 2xDR4), 8 EMLs. Typical |
| A2 | Single MPO-16 APC | Many DR8s use dual MPO-12; MPO-16 exists (Cisco DR8P) and gives one beak. Typical |
| A3 | Separate PIN array + octal TIA for RX (not Ge PDs on the TX PIC) | Gigalight block diagram; Credo Kfir is TX-only. Typical |
| A4 | No isolator in the CW path | Not found in any DR8 product |
| A5 | DSP with integrated driver | Broadcom BCM87800, MaxLinear Keystone, Marvell Spica Gen2 |
| A6 | TX = right eye/foot | Derived from MSA figures (med-high); flips if the pigeon faces away from the viewer |
| A7 | Bird-tree topology from where Stiller 2024 and Prum 2015 agree; leaves trait-matched | Jarvis 2014 differs on pigeons; no strict-lineage claim made |
| A8 | Free-space links between pigeons are a gag (real modules have no FSO); the 1311 nm beams are drawn visible | Disclaimer line |
| A9 | Vendor stamps show who makes such parts, not which vendor is in a specific product | Parody; disclaimer |

## 4. Bird tree (trait-matched, simplified)

Clade layout follows only what Stiller 2024 and Prum 2015 agree on (B): Palaeognathae, Galloanserae and Neoaves split; inside Neoaves only Columbaves (pigeons + cuckoos) is resolved; hummingbird, albatross and parrot hang from an unresolved Neoaves polytomy (the trees disagree on where Strisores sits). Dagger = extinct. Dates from F.

| Clade | Bird | Pluggable | Trait / gag |
|---|---|---|---|
| Root | Non-avian theropod | 1x9 soldered transceiver (no date shown) | Could not leave the board |
| Stem | Archaeopteryx | GBIC (SFF-8053; year shown "2000 (spec rev)") | Early big 1G pluggable; a side branch, as in biology (B1) |
| Palaeognathae | Moa (dagger) | XENPAK (2001) | Big-body 10G, extinct |
| | Elephant bird (dagger) | CFP (2010) | Giant first-generation 100G, extinct |
| | Emu | CFP2-DCO (2018) | Big-body survivor, coherent |
| | **Ostrich** | **XPO (2026)** | Largest living bird; 12.8 Tb/s, liquid cold plate up to 400 W |
| Galloanserae | Chicken | SFP / SFP+ / SFP28 (2001 / 2006 / 2014) | About 26.6 billion chickens (2022, B3); the commodity |
| Columbaves | Passenger pigeon (dagger) | QSFP+ 40G | Once everywhere; label "legacy", not "extinct" |
| | Rock pigeon | QSFP28 100G | The classic; branch plaque "RFC 1149, 1990-04-01: IP over Avian Carriers" |
| | **Cyber pigeon** | **800G OSFP DR8 (hero)** | "Columba livia cybernetica" |
| | Cuckoo | AEC (Credo HiWire 2019) | Lays a copper egg in the optical nest; near pigeons in Stiller/Prum |
| | Dodo (dagger) | CPO | Ghost silhouette at the tree edge: a pigeon that settled on an island (the package) and lost flight (teaser) |
| Neoaves (unresolved) | Hummingbird | QSFP-DD (2016) | Your pick: 8 lanes in a QSFP-size body; highest mass-specific metabolic rate among vertebrates (B6) |
| Neoaves (unresolved) | Wandering albatross | 400ZR (2020) | Largest wingspan (B7); coherent |
| Neoaves (unresolved) | Parrot | Third-party "compatible" modules | Vocal mimic; mimics another vendor's EEPROM code (gag; standard practice, not in F) |

## 5. Storyboard v1 (60 s + 2 s; one continuous camera path)

Spec carried from the interconnect film: 1080x1350 (4:5), 30 fps, 1:1 safe square, captions in the lower third, about 2.5 words/s VO. Look: cyberpunk, dark data hall, cyan/magenta neon, haze (so beams read), wet-metal pigeons with red TX eye glow and a cold cyan RX eye. Subject = Gary v2 mesh in a hoodie with a coffee cup.

### S1 Tree of life to dovecote (0-10)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.5 | Holographic cladogram floats in the dark. Root theropod, then Archaeopteryx; branches light up in sequence, each leaf a hybrid bird/module icon | Slow push through the hologram | leaf names + years |
| 2.5-4.5 | Pass the ostrich (XPO, cold-plate pipes), hummingbird (QSFP-DD), chicken (SFP), cuckoo sitting on a copper egg (AEC), dodo ghost | Weave between branches | "CONVERGENT EVOLUTION" |
| 4.5-6.0 | Dive down the pigeon branch past the RFC 1149 plaque to the glowing cyber-pigeon node | Accelerating dive | "RFC 1149 (1990-04-01)" |
| 6.0-8.0 | The node becomes a real faceplate: a wall of OSFP pigeonholes in a neon data hall. Rows of pigeons, red right eyes | Dive becomes a dolly along the faceplate | "800G OSFP DR8" |
| 8.0-10.0 | The Subject walks down the aisle sipping coffee; every head turns in unison to track him; the camera settles on the hero | Track, then settle on the hero | none |

VO (20 words): "Birds evolved from dinosaurs. So did pluggables. This is the eight hundred gig O-S-F-P D-R-eight. And it is watching you."

### S2 Outside, beak and feet (10-20)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.0 | The Subject pulls the hero out by its leash (pull tab); it dangles and flaps its fin-feathers | Orbit as it swings out | "PULL TAB" , "HEATSINK FINS" |
| 2.0-4.5 | Beak macro: dust-cap muzzle pops off; 16 fiber teeth in one row; the angled beak tip | Push into the beak face | "MPO-16 APC: 8 TX + 8 RX FIBERS" |
| 4.5-7.0 | Under the body: two feet of gold pads; right foot glows TX-red, left foot RX-cyan; the 4 low-speed pads between them pulse | Swing under the belly | "60 CONTACTS" , "TX" / "RX" |
| 7.0-10.0 | The Subject plugs it back in; the talons land in order: ground claws, then power, then signal, each with a soft glow pulse (no sparks: the mating order exists to avoid damage) | Low macro at the cage mouth, slow-mo then snap | "GROUND FIRST" |

VO (21 words): "Feathers: heatsink. Beak: an M-P-O connector, sixteen fibers. Right foot and eye transmit. Left foot and eye receive. Ground lands first."

### S3 Right eye: transmit (20-30)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.0 | Push into the red right eye: the fiber array behind the cornea, 8 fibers meeting the chip edge | Push into the eye | "FIBER ARRAY" |
| 2.0-4.5 | Deep in the socket, 2 CW lasers glow steadily; light splits down a waveguide tree into 8 lanes | Fly backward along the waveguides to the lasers, then forward with the light | "2 x CW DFB LASER, 1311 nm" |
| 4.5-7.0 | 8 Mach-Zehnder modulators blink PAM4 patterns; nerve-like traces from the brain twitch each one; tiny monitor photodiodes wink | Glide over the modulator array | "8 MODULATORS (SILICON PHOTONICS)" , "DRIVER IN THE DSP" |
| 7.0-10.0 | Modulated light leaves the chip and shoots out of the beak down a fiber across the hall | Ride the light out of the beak along the fiber | none |

VO (22 words): "Right eye: transmit. Two D-F-B lasers glow deep in its skull. A silicon photonics chip splits the light into eight Mach-Zehnder modulators."

### S4 Left eye: receive (30-40)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.0 | The fiber ends in a second pigeon's beak (the link partner); the light dives into its cyan left eye | Continuous ride into the partner's beak | none |
| 2.0-4.5 | Light lands on 8 photodiodes (the retina), each flashing in rhythm | Over the PIN array | "8 PHOTODIODES" |
| 4.5-6.5 | Behind the retina, the 8-channel TIA amplifies tiny currents into voltages, which rush up the optic nerve to the brain | Follow signals into the TIA | "TRANSIMPEDANCE AMPLIFIER (TIA)" |
| 6.5-10.0 | Pigeon POV: a 340-degree fisheye of the data hall with a surveillance HUD; the Subject is boxed and tracked; PAM4 eye-diagram glyphs float over his face | POV fisheye, slow push on the Subject | "SUBJECT ACQUIRED" |

VO (21 words): "Left eye: receive. Eight photodiodes catch the light. A T-I-A, a transimpedance amplifier, turns tiny currents into voltage. It sees you."

### S5 Exploded view: brain, body, snitch line (40-50)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.0 | Pull out of the partner's eye, whip back along the same fiber to the hero (about 1 s, motion streaks); the hero floats out of its hole; the shell splits and the parts separate on exploded-view guide lines | Continuous pull-out, fiber whip, rise into a 3/4 hero angle | none |
| 2.0-4.5 | Brain: DSP under the thermal-pad skull cap; its stamp flicks NARROWCOM, MERVALL, MINLINEAR (brain options) | Push onto the DSP | "DSP: 8 x 100G" |
| 4.5-6.5 | Hypothalamus MCU, memory, heartbeat crystal ticking, power rails pulsing from the load switch and buck converters | Arc across the board | "MCU" , "EEPROM" , "CRYSTAL" , "POWER" |
| 6.5-8.5 | The I2C spinal cord lights up and streams telemetry into the switch: temperature, voltage, laser bias, TX power, RX power... then "SUBJECT: HEART RATE ELEVATED" | Follow the stream into the cage | DDM list |
| 8.5-10.0 | Parts snap back together; the leg band reads "HATCHED BY OUTNOLIGHT"; the hero stamps back into its hole | Snap back to the faceplate | "OUTNOLIGHT" |

VO (23 words): "The brain: a D-S-P from Narrow-com or Mer-vall. A microcontroller reports temperature, laser current and light levels to the switch. Hatched by Out-no-light."

### S6 Free-space ring (50-60)

| t | Action | Camera | Label |
|---|---|---|---|
| 0.0-2.0 | The Subject pulls the fiber bundle; the link drops; eyes flicker | Medium on the Subject | "LINK DOWN" |
| 2.0-5.0 | Three pigeons leave their holes and land on three rack tops around the Subject | Rising crane, circling | none |
| 5.0-8.0 | Each right (TX) eye fires a beam into the next pigeon's left (RX) eye; the beams close a triangle ring around the Subject through the haze | Top-down crane, then descend into the ring | "FREE-SPACE OPTICAL RING" |
| 8.0-10.0 | The Subject freezes; all three heads rotate to camera; the hero blinks (its eyelid = TX disable) | Snap push to the hero's eye, hard stop | none |

VO (17 words): "Cut the fibers? They don't need them. They talk eye to eye. In a ring. Around you."

### S7 Disclaimer (60-62)

Black card, complete from frame 1: "Pigeons are real. Pluggables are real. Parody; not affiliated with any company or bird. Architecture shown is one typical build. Modules cannot see. No free-space links; 1311 nm beams drawn visible. Real pigeons have 4 toes." Small sources list below (OSFP MSA Rev 5.2, datasheets and papers in `research/`). VO (6 words, fast): "Parody. Typical build. Modules cannot see."

### Pronunciation (`say` spellings, verified later by Whisper round trip)

| Text | say |
|---|---|
| OSFP | "O-S-F-P" |
| DR8 | "D-R-eight" |
| MPO | "M-P-O" |
| DFB | "D-F-B" |
| Mach-Zehnder | "Mock-Tsender" (check Kokoro output; fallback "Mock Zender") |
| TIA | "T-I-A" |
| DSP | "D-S-P" |
| NARROWCOM / MERVALL / OUTNOLIGHT | per DevLog-000 Section 3 |

## 6. Zoom-in and exploded-view shot list

| # | Shot | Scale | Build need |
|---|---|---|---|
| Z1 | Hologram tree dive | room | tree generator from a YAML of leaves; 15 bird/module icons |
| Z2 | Faceplate dovecote | rack | OSFP cage wall (reuse `osfp_cage_and_host_connector`), pigeon instances |
| Z3 | Beak macro (MPO-16 APC) | mm | beak = MPO receptacle (reuse `mpo_connector_and_adapter`), 16 fiber teeth |
| Z4 | Feet macro (60 pads, 3 lengths) | mm | paddle-card edge with true pad layout per MSA Fig 15-1 |
| Z5 | Right-eye fly-through: FAU, edge couplers, splitter tree, 8 MZMs, 2 CW DFBs, monitor PDs | 10-100 um | TX PIC macro (extend `pic_die`, `grating_coupler_closeup` -> edge coupler variant) |
| Z6 | Fiber ride across the hall | room | fiber spline + light pulse shader |
| Z7 | Left-eye fly-through: FAU, PIN array, octal TIA | 100 um | RX PD/TIA macro |
| Z8 | Pigeon POV fisheye HUD | room | fisheye camera, HUD overlay, eye-diagram glyphs (reuse `tex_gen.py` eye) |
| Z9 | Exploded view: shell/fins, thermal pad, DSP, MCU, EEPROM, crystal, power ICs, PCB, I2C spine | module | exploded rig with guide lines (empties + per-part offsets) |
| Z10 | FSO triangle ring | room | 3 pigeon instances, volumetric beams in haze |

## 7. Builds and risks

- Cyber pigeon rig (largest new asset): pigeon proportions over the true OSFP envelope (22.58 x 13 x 100.4 mm, scaled up for hero shots); articulated neck/head (rotation gag), eyelids, feet with 8 toes, wings as fin panels, separable parts for the exploded view, the same hook convention as the interconnect assets.
- Subject: Gary v2 with a hoodie variant; motion_v2 walk, plug-in, freeze.
- Narration: Kokoro `am_michael` as before; each parody name gets a `say` spelling; Whisper round trip must transcribe names as intended.
- Risks: the scale jumps (room to 10 um) in one continuous camera need careful depth of field and haze; at 4:5 the exploded view must fit the 1:1 safe square; beam visibility needs volumetrics (render cost).

## 8. Open questions for Wentao

1. TX = right eye/foot (true MSA orientation for a viewer-facing pigeon). OK, or flip for readability?
2. S4 uses a second pigeon as the link partner (fiber ride between two pigeons). OK, or keep it inside one pigeon with a loopback?
3. VO density: 17-23 words per 10 s, dossier style ("Right eye: transmit."), naming MPO, DFB, Mach-Zehnder, TIA, DSP, microcontroller. OK?
4. Dodo/CPO ghost in the tree: keep as a teaser for a later short, or drop?

## 9. Audit corrections (fresh-context audit, 2026-10-09)

15 findings (2 high, 7 med, 6 low); core technical mapping verified against research. Applied:
- High: S5 camera jumped from the partner pigeon to the hero; now a continuous pull-out and fiber whip back to the hero. VO word counts were wrong (S3 over budget, S7 unspeakable at 9.5 words/s); VO rewritten and recounted, S7 spoken part cut to 6 words with the rest on the card.
- Med: disclaimer now covers no-FSO, modules-cannot-see, visible beams, 4 real toes; hummingbird claim narrowed to "mass-specific metabolic rate"; albatross "longest reach" removed (unsourced); GBIC "first hot-pluggable" removed, year shown as spec revision; tree drawn as a Neoaves polytomy with only Columbaves resolved; "no lens, no isolator" softened to sourced scope; DevLog-000 D3 retina corrected to PIN photodiode array.
- Low: standard-but-unsourced facts labelled (APC 8 deg, laser bias control, EEPROM contents, TX disable via CMIS); chicken count dated 2022; orientation confidence added; plug-in sparks replaced by glow pulses; VO now names components (was partly "tiny controller"); DevLog-000 D12 phrasing and stale status fixed.

## 10. Progress log

- 2026-10-09: v1 mapping and storyboard written from the four research reports.
- 2026-10-09: fresh-context audit applied (Section 9).
