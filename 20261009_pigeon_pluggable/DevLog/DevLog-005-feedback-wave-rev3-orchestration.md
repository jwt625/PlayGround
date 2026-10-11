# DevLog-005: Feedback wave rev 3 - orchestration

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Five workstreams launched in parallel; coordinator reviews and integrates |
| Inputs | Wentao's review of crude v0.1 and the hero pigeon (below) |
| Coordinator | Claude (orchestrator only; work is delegated) |

## 1. Feedback (verbatim, 2026-10-10)

```text
feedback on the scene planning:
- do not include year in the tree. Find a balance between actual bird species tree and simple evolution tree of pluggables
- the birds in the background should be constantly and randomly moving their heads around, making contact with each other's eyes (hide the beams for now because they are not visible. When showing beams later it should have asterisk or parenthesis explaining beam is invisible and rendered for viewing purpose)
- the initial scene is also better to be outside similar to the ending scene thats outside, and then transition to inside by cam following a bird entering the datacenter to charge and exchange data with other pigeons and the landing pad. Reuse rack assets from the prev interconnect film project and make the bird racks about 1/3 of the racks. The birds should also fill the bird rack f rom top to bottom, and the rack should be built to higher details similar to the interconnect film racks, and they both probably needs further improvements after you inspect more rack refs. Use GB300 as the compute rack ref. And mimic the bird rack as a switch rack with e.g. nvidia QD-3400.
- the explaining scenes around the bird is great. Next is to polish the model and improve accuracy and complexity of those scenes and components used. Inspect and model against proper references esp. the 400G and 800G example pluggable teardown pictures. Include textures for the PIC or model more complex stuff on the layout, including electrical routings, wirebonds (low speed and high speed differential), alignment marks, dummy filling etc. Include micro optics components such as silicon lenses between the DFB and the SiPho, and include proper prism for FAU attach and optics needed to map 1x8 output facet to 2x4 fiber array.
- the Tx to Rx scene switch should zoom out of the Tx eye and bird head turn and then zoom back into the Rx eye. the Rx should have proper micro optics for mapping the 2x4 array to 1x8 PD array, proper high speed differential wirebonds to a multi-channel TIA chip, and proper inductor coil etc. on the TIA chip. Find good ref pic to build it or use as texture.
- the last scene should be more natural environment, with the birds casually on different positions (instead of an octagon which is too artificial) and objects in the scene, but all observing the guy and communicating with each other.

With these feedbacks, spin up subagents to work on various components refinement and scene/cam movement update, and start iterating similar to the interconnect film project. Delegate to subagents as much as possible and you should mainly act as the orchestrator agent. Ask them to keep track of TODOs and progress in md docs as before.
```

## 2. Workstreams and ownership

Rule: each agent writes only its own files; shared files are changed only by their owner; integration happens through the interfaces in Section 3. All Blender runs go through `scripts/bslot.sh` (machine-wide cap 3), one Blender process per agent at a time. Fast draft render settings for iteration.

| ID | Agent | Scope | Owns | DevLog |
|---|---|---|---|---|
| S | Scene/camera (resumed, built v0.1) | Crude v0.2: storyboard rev 3; outdoor opening (night street like the ending), camera follows a bird flying into the datacenter to charge and exchange data at the landing pad; tree without years (balance of real bird tree and simple pluggable evolution tree); background birds with constant random head motion making eye contact (beams hidden); TX->RX switch = zoom out of the TX eye, head turn, zoom into the RX eye; natural ending scene (birds casually perched on varied objects, all observing the Subject and communicating); explanatory beams only with an on-screen note "* beams invisible (1311 nm); rendered for viewing"; integrates racks (R), subassemblies (T, X) and the hero (H) when delivered | `scripts/build_crude_film.py`, `scenes/crude_v0_2.blend`, `outputs/crude_v0_2_*.mp4` | DevLog-006 |
| R | Datacenter racks (new) | Research rack refs; reuse and upgrade the interconnect film rack assets; GB300 NVL72-style compute racks; bird racks styled as an NVIDIA Quantum-X800 Q3400-class switch rack with pigeonholes filled top to bottom; landing pad; hall layout with bird racks about 1/3 of all racks | `scripts/assets/datacenter/`, `assets/components/datacenter/`, `outputs/assets/datacenter/`, `research/racks_refs.md` | DevLog-007 |
| T | TX optics + PIC (new) | TX subassembly at true scale: DFB on submount, silicon microlens(es) between DFB and SiPh PIC, PIC with detailed layout (MZM electrodes, RF and DC routing, heaters, monitor PDs, alignment marks, dummy fill, edge couplers; texture or geometry), low-speed and high-speed differential wirebonds, prism for FAU-style attach, optics mapping the 1x8 output facet to the 2x4 lenslet eye | `scripts/assets/optics_tx/`, `assets/components/optics_tx/`, `outputs/assets/optics_tx/`, `research/tx_optics_refs.md` | DevLog-008 |
| X | RX optics + TIA (new) | RX subassembly at true scale: 2x4 lenslet eye to 1x8 PD array micro-optics, PIN PD array die, high-speed differential wirebonds to a multi-channel TIA die with spiral inductors, bond pads, decoupling; die texture or geometry from good reference pictures | `scripts/assets/optics_rx/`, `assets/components/optics_rx/`, `outputs/assets/optics_rx/`, `research/rx_optics_refs.md` | DevLog-009 |
| H | Hero pigeon (resumed, built v1) | Polish the hero against the owner's 400G and 800G teardown pictures (accuracy and complexity of DSP, PCB, passives, flex, shell); integrate T and X subassemblies into the head when delivered; keep the explaining renders and the 15 s loop pipeline | `scripts/assets/build_hero_pigeon.py` and the other hero scripts, `assets/hero/`, `outputs/hero/` | DevLog-003 |

## 3. Interfaces

- Units: Blender meters, true scale (parts in mm). glTF export Y-up as default; .blend kept.
- T and X subassembly frame: origin at the centre of the eye's 2x4 lenslet array outer surface; +X = optical axis pointing out of the eye (TX: light leaves along +X; RX: light arrives travelling -X); +Z up. Hooks: `HOOK_eye_aperture`, `HOOK_pic_mount` (or `HOOK_pd_mount`), `HOOK_flex_rf`, `HOOK_flex_dc`, `HOOK_carrier_mount`. Every part has custom property `explode` (dx, dy, dz) in meters. Deliver: `<name>.blend`, `<name>.glb`, `<name>_parts.json`, preview renders (macro, exploded, labeled), and a scale statement. The hero (H) mounts them in the head; the scene (S) scales them up for the S3/S4 macro sets.
- R deliverables: rack blends/GLBs with hooks for pigeonhole cells (`HOOK_cell_<unit>_<row>_<col>`, as delivered; +X out of the cell), the landing pad (`HOOK_landing_pad`), and hall layout data (JSON with rack positions and types) that S reads.
- Beams: hidden in background shots; any shot that shows beams carries the note "* beams invisible (1311 nm); rendered for viewing" (exact wording can be tuned).

## 4. Status

| ID | Status |
|---|---|
| S | v0.2 and v0.2.1 delivered 2026-10-10 (DevLog-006): outputs/crude_v0_2_1_20261010.mp4 integrates R racks, T/X v1 macro sets, hero GLB (H polish in progress) |
| R | v1.1 (audit fixes applied) delivered 2026-10-10 (DevLog-007); fresh audit: 13 findings (parody badges, OSFP cage size, reuse of interconnect tray assets, overhead trays, rear detail); fixes sent |
| T | v1.1 delivered 2026-10-10 (DevLog-008), self-audited by a fresh subagent; integration notes forwarded to H |
| X | v2 delivered 2026-10-10 (DevLog-009), self-audited (19 findings, 17 fixed); switching PD->TIA input to differential per owner request |
| H | v2 delivered 2026-10-10 (DevLog-003 s15): teardown-reference polish, T and X integrated, stills + 15 s loop re-rendered; soundtrack re-muxed by coordinator |

## 5. Progress log

- 2026-10-10: feedback recorded; five workstreams launched (S and H resumed with context; R, T, X new).
- 2026-10-10: R delivered racks v1 (GB300 NVL72-style compute rack, Q3400-style bird rack, real-scale Q3400 reference, bird flap, hall_layout.json: 10/30 bird racks, 180 cells). Audit: core numbers verified; fixes sent to R. Interface note forwarded to S.
- 2026-10-10: R applied all audit fixes (parody badges NVYDIA QX-3400/QX-3400B, OSFP-size true-scale cages, interconnect cartridge/tray assets reused incl. pulled-tray variant, QD couplings + blank panels, fan/PSU rear detail, overhead busway/fibre raceway/ladder tray in hall_layout.json, matte cell walls). Coordinator checked the new previews.
- 2026-10-10: T delivered TX subassembly v1.1 (DFB -> Si lens -> isolator -> Si lens -> SSC; PIC 7.0 x 7.5 mm with layout texture; GSSG bonds; FAU-style collimator block + 45 deg staircase prism + rhomboids -> 2x4 Galilean eye plate). Declined an audit suggestion to add a driver chip (research locks DSP direct drive). Coordinator checked renders; notes forwarded to H.
- 2026-10-10: S delivered crude v0.2 / v0.2.1 (outdoor opening with hologram tree and bird flap, landing pad, head-motion eye contact, beams hidden except explanatory shots with the beam note, TX->RX head-turn switch, natural street ending with 17 casual birds). Coordinator checked 1 fps sheets. Next pass after H and X deliver: crowd birds from a simplified hero GLB, S1 colour/exposure, S4 RX exposure, S5 label overlap, S6 bird size.
- 2026-10-10: H delivered hero v2 (T 02:21 and X 02:02 subassemblies integrated, eyes moved into sockets, flexes rerouted to T/X hooks, 375k tris, pin assertion OK). Coordinator re-muxed the soundtrack onto the new loop (`_bgm.mp4`; previous kept as `_bgm_v1.mp4`). Hook-orientation bug in optics_tx.blend reported back to T.
- 2026-10-10: T fixed hook orientations (local +X = documented direction, HOOK_AXIS_OK 5 build check, no part moved).
- 2026-10-10: X delivered RX v2 (2x4 lenslets, tilted Si interleaver plates, stepped Au-facet turning block, 1x8 PIN PD row, octal TIA with 56 spiral inductors, GSSG output bonds; beam landing asserted). Coordinator decision: PD->TIA input becomes differential G-S-S-G (owner asked for differential HS bonds). H told to re-integrate final X (iris, 32 RF + 6 DC/control flex); S launched on v0.2.2 (hero-derived crowd birds, S1 grade, S4 exposure, S5 labels, S6 framing).
- 2026-10-10: X switched PD->TIA input to differential (P->IN+, N->IN-, G bonds to a submount GND strip; 64 inductors, 74 TIA pads; hooks unchanged; beam landing asserted). Coordinator checked the bond close-up. H re-integration and S v0.2.2 pick it up.
- 2026-10-10: S delivered crude v0.2.2 (hero-derived crowd proxies, night grade, RX exposure, two-column S5 labels, S6 reframing; built with final hero 02:54 and X 02:51). H delivered hero v3 (X differential integrated, RX flex 32 RF + 6 DC read from X pads; stills + 15 s loop). Coordinator re-muxed the soundtrack (`_bgm.mp4`, previous `_bgm_v2.mp4`). Pending: S v0.2.3 label fix; owner review of the wave.
- 2026-10-10: S delivered v0.2.3 (S5 label 'EXPLODED VIEW', thinner leaders). Coordinator check: S5 5-7 s right-column labels clipped at the frame edge; wing/feet leaders still cross the frame (next round). Wave rev 3 ready for owner review.
