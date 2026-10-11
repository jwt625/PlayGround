# DevLog-010: Production wave - film v1 -> v2 (models, animation, VFX, SFX, VO)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Launched; next owner review at film v2 |
| Inputs | Wentao's decisions on the rev 3 open points and the instruction to move to production (below) |
| Coordinator | Claude (orchestrator; work delegated; max 5 concurrent subagents; max 3 Blender processes machine-wide via `scripts/bslot.sh`) |

## 1. Owner input (verbatim, 2026-10-10)

```text
the pull tag pulls them backward out for maintenance. No need to show it and in the vid it can still come out forward.
bird racks are full filled from the top and can be tiered in depth to fill the gaps like a caged far, where their heads stick out for observation.
The ending flock rn is good. They should be moving and establishing links differently all the time. (similarly for the ones in the data centers).

Go on keep iterating on models, animations, VFX, and also start adding SFX and VO. I should not be still expecting to see crude version. I'm expecting at least v2 for your next feedback/inquiry. Follow similar feedbacks and workflow and concurrency limitations we had for the interconnect film. esp. try maintaining good consistency and pace between the scenes and VOs.
```

## 2. Decisions

| # | Decision |
|---|---|
| P1 | Pull tab (tail leash) is a maintenance feature (pulls the bird backward out of the cage); not shown being used. In S2 the hero comes out FORWARD (hops out of its cell). |
| P2 | Bird racks: completely filled, filled from the top; cells tiered in depth (staggered tiers like a caged farm) so every head sticks out for observation and the gaps are filled. |
| P3 | Ending flock layout stays; all birds (street and datacenter) keep moving and keep establishing different links all the time (re-pairing eye contact; links shown only where explanatory, always with the beam note). |
| P4 | No more "crude" builds: next deliverable to the owner is film v2 (v1 internal, then critic + audits, then v2). |
| P5 | Audio: VO (Kokoro TTS, local), procedural SFX, mix with sidechain ducking, -16 LUFS integrated, true peak <= -1.5 dBTP; narration text from DevLog-002 rev 2 (narrator never says "pluggable"); parody names pronounced per DevLog-000 Section 3, verified by Whisper round trip. |
| P7 | The Subject is Gary (owner, 2026-10-10: "use the same exact outfit gary has for the guy. i.e., he is gary."): exact gary_v2 outfit and materials from the interconnect film; no hoodie. |
| P6 | Pace: VO line timing is the master clock for beats; scene beats, captions and SFX cues align to it; 10 s per scene, 60 s + 2 s disclaimer. |

## 3. Workstreams (5 concurrent)

| ID | Agent | Scope | Owns | DevLog |
|---|---|---|---|---|
| S | Scene assembly (resumed) | Film v1/v2 build: integrate C/E/R/H/T/X assets and A's timing; camera; P1; VFX integration; per-scene renders and composition; pacing to VO | `scripts/build_film.py` (new, from `build_crude_film.py` v0.2.3), `scenes/film_v*.blend`, `outputs/film_v*` | DevLog-011 |
| R | Racks (resumed) | P2 tiered-depth caged-farm bird rack v2, fully filled from the top; hooks per bird with head-out pose; keep GB300 and hall layout | datacenter files as before | DevLog-007 |
| C | Characters + animation (new) | Subject on the interconnect film's Gary v2 rig + motion_v2 actions (copied in); pigeon animation rig for hero and crowd proxy (head/neck, wings fold/flap, hop, take-off, landing, legs); behaviour module: continuous random head motion and link re-pairing scheduler (seeded, deterministic) emitting link events | `scripts/anim/`, `assets/components/characters/`, `outputs/assets/characters/` | DevLog-012 |
| E | Environment + VFX library (new) | Street (opening + ending) at production detail: facades, DC facade with bird flap, props kit (lamp posts, poles/wires, railings, AC units, signs, bins, scooter, rooftops, CCTV), sky; hatchery assembly line with animation hooks; VFX node-group library: hologram, beam/link (with note), volumetric haze (cheap), compositor glare/bloom, wet ground, neon | `scripts/env/`, `scripts/vfx/`, `assets/components/environment/`, `outputs/assets/environment/` | DevLog-013 |
| A | Audio (new) | VO (Kokoro, re-download model files outside the repo), pronunciation check, timing table that S follows; SFX library and cue renderer driven by S's `scene_events.json`; music bed optional (royalty-free/procedural only); mix and loudness; mux | `scripts/audio/`, `outputs/audio/` | DevLog-014 |

H (hero), T (TX optics), X (RX optics) are complete; S uses their assets as is.

## 4. Interfaces

- Timing: A writes `scripts/audio/narration.json` (single source: per line `t0`, `t1`, `say`, `show`, measured duration) and `outputs/audio/vo_<date>.wav`; S places captions and beats from it.
- Events: S writes `scripts/audio/scene_events.json` at build time (`t`, `name`, `kind`, optional `pos`/`gain`); A renders SFX from it (kind -> sound mapping in A's scripts).
- Animation: C provides importable Python modules (`scripts/anim/pigeon_rig.py`, `scripts/anim/behaviors.py`, `scripts/anim/subject.py`) with a documented API, deterministic seeds, and a demo blend/render.
- Environment/VFX: E provides blends with hooks and `scripts/vfx/*.py` functions that create node groups/materials/compositor setups; S calls them.
- Beam note: any shot showing beams carries "* beams invisible (1311 nm); rendered for viewing".

## 5. Resource rules

- Disk: 14 GiB free at start. Render per scene, encode, delete PNG frames immediately; keep free space above 8 GiB (check with `df -h` before every full render); no duplicate large caches.
- Blender: only via `scripts/bslot.sh`, one process per agent at a time.
- Render tiers (owner, 2026-10-10: "make sure the agents are using the fast render until after v2 or until adding detailed vfx."): v1 and v2 at the fast tier: EEVEE 1 sample, 540x674, raytracing off, volumetrics off, persistent data, target: full film <= 10 min wall clock (about 0.3 s/frame effective; 3 parallel chunks); H.264 yuv420p + AAC, +faststart, 30 fps. Higher tiers only after v2 or when detailed VFX work starts.

## 6. Plan

1. Wave 1 (parallel): R, C, E, A build; S converts the crude build into `build_film.py`, integrates as assets land, applies P1, adopts A's timing.
2. Film v1: S renders; A mixes and muxes.
3. Critic pass (fresh agent) on v1: per-scene visual critique (readability, VFX, animation, pacing vs VO), plus fact/rules audit; coordinator assigns fixes.
4. Film v2: fixes applied, final render + mix; coordinator verifies and reports to the owner.

## 7. Status

| ID | Status |
|---|---|
| S | film v1 rendered 2026-10-10 (outputs/film_v1_20261010.mp4, 118 s wall in 3 chunks, 0.30-0.43 s/frame per process) |
| R | bird rack v2 delivered 2026-10-10 (DevLog-007 s10): 3 stair-step tiers, 24 birds/rack, 240 slots, heads out |
| C | delivered 2026-10-10 (DevLog-012): pigeon rig, behaviours scheduler, Subject (Gary v2 hoodie), proxy library, demo |
| E | delivered 2026-10-10 (DevLog-013): street opening/ending sets, street kit, hatchery loop, VFX library |
| A | audio v1 delivered 2026-10-10 (DevLog-014): VO am_liam, Whisper-checked; SFX from scene_events; procedural music; mix -15.9 LUFS, TP -4.0; waiting for film v1 video to mux |

## 8. Progress log

- 2026-10-10: owner decisions recorded (P1-P6); wave launched.
- 2026-10-10: R delivered bird rack v2 (caged farm: tiered depth, wire bars, fully filled from the top, 239 crowd + hero slot). Coordinator checked previews; forwarded to S.
- 2026-10-10: A delivered audio v1 (narration.json master clock, S5 lines moved earlier; 328 SFX events; mix -15.9 LUFS). Coordinator checked the mix plot; forwarded timing + event requests (pan, hall ambience split at macro sets, real-time cues for provisional kinds, no leash_pull) to S.
- 2026-10-10: E delivered environment + VFX library. Coordinator checked previews; decisions sent to S: AgX Punchy film-wide, per-scene world haze, ENV_lights windowed to street scenes, clearance on new geometry, puddles per shot.
- 2026-10-10: C delivered characters + animation (deterministic scheduler: 2,157 links over 239 rack birds with R's layout). Coordinator checked the demo; integration notes sent to S (smaller neck extension, flight windows, link events to scene_events, hero tucked legs). All asset workstreams done; S building film v1.
- 2026-10-10: owner asked for fast renders until after v2. S's film_v1 render ran at ~6 s/frame (8 spp + raytracing + volumetrics); coordinator stopped the loop and switched S to the fast tier (target <= 0.5 s/frame).
- 2026-10-10: owner: "if the full film is taking 10+ min to render, you are doing it wrong. It is too slow for iterations." Rule: full film render <= 10 min wall (3 parallel bslot chunks, per-scene collection visibility, 1 spp, no raytracing/volumetrics, light proxies/LOD, compositor off in drafts, FILM_TIER preview/draft). Sent to S.
- 2026-10-10: S rendered film v1 in 118 s wall (fast tier, per-scene collection windows). Coordinator reviewed 1 fps sheets (issues: camera through the Subject's head ~S1 9 s, tree labels vs DOVECOTE sign, S2 cutaway placeholder slab, S5 wing panels, S6/final close-up dark at fast tier). A muxing; fresh critic launched on v1.
- 2026-10-10: A muxed film v1 (outputs/film_v1_20261010_av.mp4, -16.0 LUFS, TP -3.7, Whisper 0 misses). Event/VO conflicts for v2: hud_lock to ~38.4 on 'It sees you'; stamp ticks for NARROWCOM/MERVALL at ~42.6/43.4; S6 door exit after the VO line (move exit earlier); S4 PD flashes run 0.7 s past the line (ok).

## 9. v2 work order (from the fresh critic on film v1 + audio conflicts, 2026-10-10)

Critic report: ranked top 15 + per-scene notes (contact sheets in the coordinator scratchpad `pigeon_critic_v1/`). Keep list: tiered racks, flap entry, P1 forward hop, TX macro/VO sync, red/cyan fills, RX bonds/spirals, HUD lock sync, door exit, TX blink, AV sync, disclaimer content.

| # | Time | Fix | Owner |
|---|---|---|---|
| 1 | 50-60 s | Keep the Subject in the central third 52-58.6 s; 3+ perched heads at 2-4 m turning to him; end on hero foreground + frozen Subject | S (C poses) |
| 2 | 7.9-9.8 s | Remove the 7.97 s cut and the head near-miss (>= 0.6 m clearance); one continuous move ending on the hero's eye with the Subject behind; no whip at 9.6 | S |
| 3 | 17-20.8 s | Cutaway inside R's real hero cell (lip below eye level, host pads visible), continuous pull-out into S3 (no cut at 20.8) | S, R |
| 4 | 11.8-17 s | Hero sits on the Subject's hand (Gary reaches in), hand turns to show the belly | C, S |
| 5 | 14.6-17.9 s | Pulse right pads + right eye rim red at 14.6, left pads + rim cyan at 16.4; hold on the belly to 17.9 | H, S |
| 6 | rack shots | Hero signature (rim/spot, glowing leg band, brighter neck), darker crowd, hero >= 25% frame width | H, S |
| 7 | 29.8-31 s | Orbit ~60 deg during the head turn; show the sender bird across the aisle | S |
| 8 | 41-48 s | Centre the exploded hero; Subject out of the way; wing panels at ~30% and off the camera path; NARROWCOM/MERVALL stamp on the DSP die; push in on MCU at 44.2-46; telemetry as a large mid-left HUD | S, H |
| 9 | 48-49.75 s | Leg-band close-up at the snap-back; hatchery conveyor/robots moving; bird hops off; no white blank over "Hatched by" | E, S |
| 10 | 0-4.6 s | Tree 1.5 m off the DC sign; start closer; hero in front of the trunk; bottom 18% free of world labels; Subject walks in the lower third 0.5-4.0 | S, E |
| 11 | global | Safe zones: headers top 15%, bottom 18% captions only; leader lines for contact labels; beam note ~14 px | S |
| 12 | 33-34.2 s | Camera on the PD array during "eight photodiodes"; Si material with edge highlights on the dark slabs | S |
| 13 | 55-58.6 s | Bigger/closer satellite; hold on 2-3 CCTV links; fix puddle material; hide the beam note when no beams | S, E |
| 14 | audio | Gate SFX events on camera visibility (crowd links, landings, footsteps, hand landing); boost explode/snap_back; stamp ticks at ~42.6/43.4; hud_lock onto "It sees you" (~38.4); Subject exits the door before/at "They follow you outside" | S, A |
| 15 | 6-8 s | Follow the hero at ~1.5 m to the pad; pad glow; screen-space label; DR8 label after the hero reads | S |
| n | misc | "Meet the OSFP" on the face; beak latch close-up at s2b; tail tab material; fill out 0.3 s earlier at 28.3; hold TIA to 36.5; Subject glances at the lens on "It sees you"; no ghost wing into S6; flyer wing near-miss 51.5; S7 card centred | S, H, C |

Render tier stays fast (owner); full film <= 10 min wall.
- 2026-10-10: owner: the Subject is Gary in his exact outfit (P7); sent to C (subject.py default) and S.
- 2026-10-10: R delivered hero_cell_mating (lip, 60 host pads, cutaway wall; slot wires widened). E delivered v2 env fixes (moving hatchery + hopper, fast wet ground, close satellite pass, movable tree projector). Coordinator decisions to S: tree base y18p5; compositor glare on (measured ~+13%, within the fast tier / 10 min budget); hatchery hop inside 48.3-49.6 s.
- 2026-10-10: H re-rendered the 15 s loop (v4; coordinator re-muxed soundtrack, previous _bgm_v3). C delivered exact-Gary Subject + hold_bird / look_at_lens / S6 helpers (demo checked: Gary in hard hat and overalls, bird hops onto his palm). All asset fixes for v2 delivered; S building v2.
- 2026-10-10 14:48: owner asked to deliver v2 with audio now while agents continue. Coordinator muxed the 12:21 v2 render (outputs/film_v2_20261010.mp4, built 12:14) with SFX from the matching event snapshot (outputs/audio/scene_events_film_v2_20261010_1216.json; v1 SFX kept as sfx_20261010_film_v1.wav), VO rev 3 and the procedural music: outputs/film_v2_20261010_av.mp4 (540x674 H.264 High yuv420p, AAC-LC 48 kHz stereo, 62.000 s, -16.0 LUFS, TP -3.7 dBTP, faststart). Checked: caption/VO at 15.0, 38.6, 55.5 s; Whisper spot check. Delivered to the owner. S continues the v2 fix list (v2.1).
- 2026-10-10 14:58: S delivered the fixed v2 render (14:54; 193 s wall; pops removed; clearance 0). Coordinator kept the delivered v2 AV (14:47, from the 12:21 render) and muxed the new render as v2.1: outputs/film_v2_1_20261010_av.mp4 (SFX from scene_events snapshot 14:50, VO rev 3, -16.0 LUFS, TP -3.6). Known: S4 orbit hero small, S2 hand-hold busy background, RX bench slab, S5 parts small, satellites still smallish. Credits: outputs/CREDITS_20261010.md.
