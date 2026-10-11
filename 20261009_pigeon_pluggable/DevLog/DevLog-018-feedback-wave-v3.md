# DevLog-018: Feedback wave v3 - orchestration

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Launched; batch 1 running, batch 2 queued |
| Base | film v2.1 (`outputs/film_v2_1_20261010_av.mp4`) |
| Coordinator | Claude (max 5 concurrent subagents; Blender cap 3 via `scripts/bslot.sh`) |

## 1. Owner feedback (verbatim, 2026-10-10)

```text
interim feedbacks:
- initial tree is good. Remove actual pictures of the real pluggables, and just use bird pictures, bird names, and pluggable names.
- remove "fancy name for a pigeon". Replace it with workhorse of modern data centers.
- "feathers heat sink" replace with the more technical names of the heat sink, like Integrated heat sink (IHS)
- fix the shelves the pigeons are standing on, most of them feet are not making good contacts.
- around 0:16, the pigeon is overlapping with other pigeons on the shelf, move both the focused pigeon and gary more to the right
- dont use "real quick" etc. Directly start describe what is inside, e.g. DFB lasers and where they are from (source photonics, lumentum / coherent or their parody names)
- add fast and traveling glowing effect to the mzi to show the electrical signal coming in, they should come in from the CB contacts and thru the wirebonds and then onto the traveling wave electrodes. Use different color than the optical beam color
- the PIC circuits should also light up accordingly, e.g., DFB turns on as camera moves in, light propagates follwoing camera panning left, and light up the circuits and MMI tree accordingly, and thru and coming out of the MZI modulated.
- ~0:31 when camera move back in Rx eyes, camera should turn left similar to how it was moved away of Tx eyes but in reverse to move in.
- ~0:35, cleanup the black fixtures so that the lens and light path is more visible. And add glowing waves on the PD and TIA and wirebonds and PCBs to show electrical signals flowing.
- 0:44: no need to talk abt dumb ucontroller, talk about DFB, PD and TIA etc. (semtech, macom and their parody names). Dont zoom in and out and back in too much, just slow orbit the cam.
- 0:50: come up with a better assembling animation showing actual isolators lens getting put in place and wirebonds getting connected and enclosure moved in place etc. DYOR on reference tools and equipments images as needed. Use ounolight instead of outnolight, too weird.
- dont say they follow you outside, just say they observe you outside. Too few beams active at a time, there should be at least 3-4 active sweeping, and 2-3 fixed point to point comm thats changing/switching.
- satellite handoff is also too simple, it should be MIMO and updating config as the satellite array moves around. The satellite too close too big, 2x smaller and move them 2x faster.
- your last line is bs / too plain. Come up something that is more grand, like IoT, distributed intelligence etc.
- gary should be unaware and goes abt his day instead of getting scared.
- final outdoor scene should have san francisco signature buildings or landscapes.

Now start iteration again and delegate to subagents as much as possible. Keep requests and progress in md docs as usual. Render a better resolution for v3.
```

## 2. Decisions

| # | Decision |
|---|---|
| V1 | Module maker parody renamed OUTNOLIGHT -> OUNOLIGHT everywhere (VO, signs, leg band, hatchery, labels, credits, docs going forward); pronunciation verified by Whisper |
| V2 | VO changes: S1 last line -> "Workhorse of modern data centers."; S2 "Feathers? Heatsink." -> integrated heat sink (IHS); S3 starts directly with what is inside (DFB lasers + parody vendors, e.g. DIMENTUM / INCOHERENT); S5 drops the microcontroller line, names DFB / PD / TIA vendors (parody: DIMENTUM, INCOHERENT, SEMTREK, MACOMBO); S6 "They observe you outside."; last line grander (IoT, distributed intelligence); no "real quick"-style filler; Gary is unaware (lines must not imply fear) |
| V3 | Electrical signals use a distinct colour from the optical beams (optical stays TX red / RX cyan; electrical = warm gold/amber) |
| V4 | Satellites: 2x smaller, 2x faster, MIMO links that re-configure as the array moves |
| V5 | S6 street: San Francisco signature skyline/landmarks (generic geometry, no logos) |
| V6 | v3 render: iterate at 540x674 fast tier; final v3 at 1080x1350 (4:5) with modest samples; target full render <= 20 min wall at 1080 |

## 3. Routing (5 concurrent max)

| ID | Agent | Items | DevLog |
|---|---|---|---|
| S | Scene (resumed) | 0:16 shift hero + Gary right; RX entry camera turns left (reverse of the TX exit); S5 slow orbit only; beam counts in S6 (3-4 sweeping + 2-3 switching point-to-point); Gary unaware staging; integrate all; v3 render at 1080x1350 | DevLog-011 (new section) |
| T | TX optics (resumed) | Electrical signal glow: board contacts -> wirebonds -> traveling-wave electrodes (gold, fast, traveling); PIC light-up sequence: DFB on, light propagating through input couplers, MMI tree, MZI arms, modulated output, keyable to the camera pan | DevLog-008 |
| X | RX optics (resumed) | Hide/thin the black fixtures (keep mechanics plausible) so lenses and light path read; electrical signal glow waves on PD, TIA, wirebonds, carrier/PCB traces | DevLog-009 |
| E | Environment (resumed) | San Francisco skyline/landmarks for the S6 street; satellites 2x smaller, 2x faster; MIMO link helper that reconfigures as satellites move; more beam helpers for 3-4 sweeps + 2-3 switching links | DevLog-013 |
| W | Writer (resumed) | VO rev 4 per V2, owner style | DevLog-016 |
| M | Assembly (new) | Hatchery v3: researched assembly process (die attach, active alignment of lenses/isolators, wire bonding, lid/enclosure placement) with recognizable equipment, using the real T/X/H parts; OUNOLIGHT signage | DevLog-019 |
| Tree | Intro (resumed) | Remove the form-factor model images from the cards; keep bird photo, bird name, form-factor name | DevLog-017 |
| C | Characters (resumed) | Proxy/hero feet contact on shelves (with R's slot floor data); Gary "goes about his day" actions (coffee, phone/clipboard, casual walk, no startle) | DevLog-012 |
| A | Audio (resumed) | Voice rev 4 (OUNOLIGHT pronunciation), SFX for the new electrical/light-up and assembly events, mix, mux | DevLog-014 |

Batch 1: S, T, E, M, W. Batch 2 (as slots free): X, Tree, C, A.

## 4. Progress log

- 2026-10-10: feedback recorded; batch 1 launched.
- 2026-10-10: W drafted VO rev 4 (narration_rev4_draft.json; DevLog-016 s12): 'Workhorse of modern data centers.', 'Feathers? Integrated heat sink.', S3 'Two DFB lasers, by DIMENTUM. Steady light, zero information.', S5 'Photodiodes by INCOHERENT, TIAs by SEMTREK or MACOMBO.', 'Hatched by OUNOLIGHT.', S6 'They observe you outside.' ... 'One flock, one mind: the Internet of Birds.' Coordinator kept the main lines; A voicing (batch 2 slot 1). Queue: X, Tree, C.
- 2026-10-10: T delivered TX FX (optics_tx_fx.blend; tx_light_up / tx_signal_glow / tx_front_track; amber RF vs red optical). Coordinator checked both strips; forwarded with VO-anchored timing to S. X launched (batch 2): RX fixture cleanup + light arrival + signal glow.
- 2026-10-10: X delivered RX FX (rx_clean_fixtures, light arrival, amber signal glow; strips checked). Forwarded to S with VO-anchored timing. C launched (feet contact + Gary unaware). Queue: Tree.
- 2026-10-10: A voiced rev 4 (vo_20261010_rev4.wav; fixes: DIMENTUM phonetic, T-I-A's; OUNOLIGHT 'Oo-no-light'); nine v3 SFX kinds added. Anchors + kinds forwarded to S. Tree launched (remove model images). All batch-2 agents dispatched.
- 2026-10-10: Tree delivered intro v3 (no module images; photo + bird name + form-factor tag; API/hooks unchanged; checked mid still). S rebuild picks it up automatically.
- 2026-10-10: M delivered the assembly hatchery v3 (5 stations with real parts, 3.0 s, 3-shot camera, OUNOLIGHT sign; preview checked). Decision: replaces E's hatchery; slot ~47.9-50.0 s at speed ~1.45 keeping all shots. H asked to rename the hero leg-band text to OUNOLIGHT.
- 2026-10-10: H renamed OUTNOLIGHT -> OUNOLIGHT in the hero (leg band object renamed to leg_tx_band_text_OUNOLIGHT; board silkscreens; stills re-rendered with _v4 backups). S informed. Frozen v1/crude scripts keep the old spelling as history.
- 2026-10-10: E delivered v3 street (SF hill street + landmarks + cable car, OUNOLIGHT neon; satellites 0.5x scale, 2x speed; mimo_links; street_beam_choreo 4 sweeps + 3 switching p2p). Coordinator checked previews; forwarded to S. Remaining: C (feet, Gary), S (integration + v3 render).
- 2026-10-10: C delivered feet contact helpers (place_on_floor: 0.00 mm gap, 100% support over 239 birds; plant_on_surface / plant_on_wire) and Gary casual actions (stroll_unaware replaces startle_freeze). Checked feet sheet. All v3 assets delivered; S integrating and rendering v3.
- 2026-10-10 16:42: owner: 'start rendering sections where it is clear for v3' and 'always aggressively parallelize the pending work'. Coordinator rendering clear ranges 135-780 / 781-1430 / 1505-1860 at 1080x1350 from snapshot scenes/film_v3_snap_1634.blend; S: intro pass + hatchery 1431-1504 (clearance fix) + assembly/encode; A: pre-mixing v3 audio from the current events. Memory saved: aggressive-parallelism.
- 2026-10-10 16:52: v3 delivered: outputs/film_v3_20261010_av.mp4 (1080x1350 H.264 High yuv420p, AAC 48 kHz, 62.000 s, -16.0 LUFS, TP -4.2, faststart). Picture: coordinator chunks 135-1860 from snapshot + S's hatchery 1431-1504 (clearance 0 hits) + intro pass xfade at 4.5 s; audio: VO rev 4 + SFX from events (md5 unchanged) with overrides. Frames and snapshot deleted (disk 17 GiB). Full film render wall time ~6 min at 1080x1350 across 3 slots.
