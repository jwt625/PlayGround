# DevLog-011: Film v1 -> v2 scene assembly (workstream S)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Film v1 rendered (`outputs/film_v1_20261010.mp4`, video only); waiting for the critic pass |
| Depends on | `DevLog-010-production-wave-film-v1-v2.md` (owner input, P1-P6, interfaces, resource rules), `DevLog-006-crude-v0_2.md` (rev 3 storyboard, v0.2.3 build) |
| Generator | `scripts/build_film.py` (from `build_crude_film.py` v0.2.3; crude scripts frozen: `build_crude_film_v0*.py`, v0.2.3 as `build_crude_film_v0_2_3.py`) |
| Outputs | `scenes/film_v*.blend`, `outputs/film_v*`, `scripts/audio/scene_events.json` |
| Inputs from other workstreams | R racks v2 (DevLog-007), C characters/behaviours (DevLog-012), E environment/VFX (DevLog-013), A narration timing + SFX kinds (DevLog-014) |
| Authors | Claude (workstream S), at Wentao Jiang's request |

## 1. Brief (coordinator, 2026-10-10)

1. P1: in S2 the hero comes out FORWARD (hops out of its cell); no leash pull shown (leash stays on the model, unused).
2. Integrate as other agents deliver (poll DevLogs/asset folders): R bird rack v2 + hall layout; C pigeon rig + behaviours and the Subject (Gary v2 rig + motion actions); E street/facade/props/hatchery + VFX library; A narration.json timing (VO is the master clock; 10 s scenes). Placeholders until each lands; do not block.
3. Emit `scripts/audio/scene_events.json` at build time (t, name, kind, optional pos/gain) for every SFX-worthy event; agree kinds with A (DevLog-014).
4. VFX: hologram tree look, haze (hall and street, cheap), compositor glare/bloom, neon reflections, eye glows, link beams only with the note.
5. Render tiers: iterate at 540x674 fast; film v1 at 540x674 with better samples; v2 at 1080x1350 if disk/time allow. Render per scene, encode, delete frames, keep > 8 GiB free; one Blender at a time via bslot.
6. Film v1: `outputs/film_v1_<date>.mp4` (video only; A muxes audio). Verify with stills every second, clearance check, 1 fps sheets; report.

## 2. Plan

- Start: copy v0.2.3 -> `build_film.py`; remove crude-only overlays from the film look (time code, `[FX: ...]` notes stay off in film renders; a `FILM_DEBUG` env switch keeps them for review renders).
- P1: S2 0-2 s: beak opens (latch release), the hero hops forward out of its cell onto the Subject's hand; GLB leash shown on the model, not used; stretch leash removed.
- Events: one `EV(t_film, name, kind, pos, gain)` collector; written to `scripts/audio/scene_events.json` at the end of the build, sorted by t; kinds provisional until A's DevLog-014 lists its mapping.
- VFX (own, until E delivers): compositor Glare (fog glow) for neon/eyes/beams; cheap haze = low-density volume boxes in the hall and street (EEVEE volumetrics at coarse tiles); hologram tree material with emission + transparency; existing fake wet streaks.
- Timing: read `scripts/audio/narration.json` when present; captions from its t0/t1 per line; scene beats stay inside 10 s scenes.
- Render: per-scene frame ranges (300 frames), encode per scene, concatenate, delete frames.

## 3. SFX event kinds (provisional, for workstream A)

`scripts/audio/scene_events.json`: `{"schema", "fps", "duration_s", "generator", "events": [{"t", "name", "kind", "pos"?, "gain"?, "dur"?}]}`; `t` in film seconds (scene n starts at 10 (n-1)); `pos` in world metres (hall/street frame); `dur` for sustained cues.

| kind | when |
|---|---|
| ambience_street / ambience_hall | street (S1 0-6.2, S6) / hall (S1 6.0 onwards) beds |
| hologram_hum | tree hologram S1 0-6 |
| wing_flap | every 0.18 s while a bird flies (hero S1, S2 hop, S6 flyers) |
| flap_swing | bird-flap vent curtain |
| landing, hop | landings (pad, hand, perches) and hops |
| pad_charge | landing pad charge glow |
| cell_dock / cell_undock | bird enters / leaves its cell |
| beak_latch | beak release, snap, bite on the cage lip |
| contact_mate | GND, power, signal contacts (S2 cutaway) |
| eye_glow_on | eyes power up after mating |
| head_turn / head_turn_crowd | single head turns (hero, near-hero crowd glances, gain 0.15) / many heads at once |
| macro_whoosh | colour fills into and out of the eyes |
| laser_on, pic_hum, rx_hum | TX macro (DFBs, PIC) / RX macro |
| link_blip | link established (S3/S4 bundles, S6 eye links) |
| hud_lock | POV target box |
| explode / reassemble, crystal_tick, data_stream | S5 |
| hatchery_robot | S5 hatchery stations |
| footstep | Subject walking |
| sat_ping, cctv_whir | S6 uplink, CCTV links |
| blink | hero TX blink |
| card | disclaimer card |

## 4. TODO

- [x] build_film.py created; crude v0.2.3 frozen
- [x] Film look switches (no TC / FX overlays unless FILM_DEBUG=1), P1 S2 forward hop (GLB tail tab shown, unused)
- [x] scene_events.json emitter (359 events)
- [x] VFX (own): compositor fog glow, hall/street haze volumes, hologram flicker; R overhead infrastructure added
- [x] Captions from A's narration.json (25 lines; S7 VO carried by the card)
- [x] Clearance checker: exact BVH surface/inside test after the bounding-box pre-test (vent-frame false positive gone), haze volumes ignored
- [x] Poll and integrate: R v2 racks, C rig/behaviours/Subject, E environment/VFX, A narration.json (section 6)
- [x] Verify (stills every second, clearance 0 hits every frame, 1 fps sheets); film v1 rendered (3 chunks, 118 s) and encoded `outputs/film_v1_20261010.mp4`
- [ ] Report to coordinator; v2 after the critic pass

## 6. Integration as built (film v1)

| Source | Integrated | How |
|---|---|---|
| R v2 (DevLog-007 s10) | tiered caged-farm bird racks, 240 slots, overhead infrastructure, flap vent | rack collections instanced from `hall_layout.json` v2 (+1.0 m y shift so L07 sits at y = 7.0); 239 crowd birds = C head rigs on the slots, neck extension = 0.5 x R's `neck_extension_m` (coordinator) |
| C (DevLog-012 s5) | hero rig, crowd/street/companion proxy rigs, behaviours, Subject | `PR.build_hero_rig` (H's GLB, tail tab kept, unused, P1); `PR.make_proxy_rig` head mode (rack crowd), full mode (17 street birds + 2 companions); `BH.schedule` hall (239 + 2, subject observation S1 8-S2 2 and S5 0.5-4.5, across-aisle bird blocked for the story link) and street (17, forced eye links S6 2.5 s, observation S6 4.4-8.6, blocks for uplink/CCTV/end stare); story aims with `PR.head_look_at` (hero TX/RX to the across bird, POV, uplink, CCTV, end stare), wings `flap_cycle`/`glide`/`wings_fold` on every flight, hero TX blink `tx_disable`; Subject = `SJ` (walk_path brisk 1.5 m/s in S1, turn R 90, reach to the hero, look at it, sip coffee in S4, turn + walk out S5, walk into the street S6, glance, freeze) |
| E (DevLog-013 s5) | street opening + ending, hatchery, VFX library, AgX Punchy | each child collection of `ENV_street_opening` / `ENV_street_ending` instanced (street children windowed to S1 / S5 8-10 + S6; facade always); R's vent appended separately so its curtain swings; CCTV lenses and satellite from the ending hooks; hatchery at x = 600 as a white-fill flashback in S5 8.2-9.8; `setup_eevee` (draft preset, raytracing + volumetrics, AgX Punchy), `night_sky_world`, `world_haze` keyed per scene (street 0.0035, hall 0.0012, macro 0), `setup_compositor` (bloom, vignette, grain); hologram material on the tree; beam material (`make_beam` + `key_beam`) on every explanatory link |
| A (DevLog-014) | narration.json captions; SFX kinds | captions from `narration.json` (S5 lines moved earlier by A; S5 beats re-aligned); `scene_events.json` uses A's canonical kinds, `pan` from camera space, hall ambience split around the macro sets, footsteps from the Subject's root motion, crowd / street link blips + drops from the behaviour schedules (thinned) |

Render tier (owner, 2026-10-10: fast render until after v2 or until detailed VFX; full film <= 10 min): `FILM_TIER` preview 360x450 / draft 540x674 (default) / review 1080x1350; `FILM_FAST=1` (default): EEVEE 1 sample, ray tracing off, volumetrics off (no world haze; E's haze cards stay), shadows off (`FILM_SHADOWS=1` to enable), compositor off (`FILM_COMP=1` for bloom/vignette/grain, about 15 % cost), persistent data on. Visibility windows keep each frame's depsgraph small: hall, racks, crowd rigs and hall lights only from S1 5.0 s to S6 4.0 s; street sets per layout; macro sets only in their seconds; hatchery only S5 8-10.

Measured (30 frames per scene, `bpy.ops.render.render(animation=True)`, includes the first-frame shader compile of about 5 s):

| Scene (frames) | s / frame | note |
|---|---|---|
| S1 (31-60, street + tree) | 0.33 | was 0.92 before the hall was windowed out of S1 0-5 s |
| S1 (211-240, hall) | 0.33 | |
| S2 (421-450) | 0.33 | 3 processes in parallel |
| S3 (721-750, TX macro) | 0.43 | 3 in parallel |
| S4 (1021-1050) | 0.37 | 3 in parallel |
| S5 (1321-1350) | 0.36 | 3 in parallel |
| S6 (1621-1650) | 0.36 | 3 in parallel |
| S7 (1801-1830) | 0.30 | |

Earlier tiers for reference: 8 samples + ray tracing + volumetrics + shadows about 3-6 s/frame (stopped by the coordinator); 1 sample + shadows 0.54-0.97; 1 sample, no shadows, compositor on 0.26; off 0.23 (S2, 60 frames, single process).

Film v1 render: 3 parallel chunks through `bslot.sh` (1-600, 601-1200, 1201-1860), **118 s wall clock for all 1860 frames**, 0 errors; encoded `outputs/film_v1_20261010.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 yuv420p CRF 16, +faststart, video only), frames deleted.

## 7. Known issues (film v1, for the critic pass)

- S1 8.3-8.8 s: the camera passes within a few cm of the Subject's head (his walk is now faster, 1.5 m/s); retime/offset the camera key at 8.4 s.
- S5: the exploded wing panels sweep large and dark across the frame at 4-6 s; labels partly behind them; the Subject walks behind the hero through the aisle.
- S6: street at the fast tier (no shadows, no bloom, no haze) reads dark; perched birds small in the crane frames; final hero close-up dark against the night.
- Fast tier has no shadows: rack cells and the hero read flatter than with shadows; bloom/haze are off (enable at the review tier).
- Hero GLB is used in every scene (no lighter LOD for wide shots yet); crowd/street/companions are C's proxies.
- Leg tuck in flight is not used (coordinator note); legs stay down in the hero's flights.

## 5. Progress log

- 2026-10-10 08:32: DevLog-010 read; disk 14 GiB free; no C/E/A deliverables yet (no DevLog-012/013/014); build_film.py copied from crude v0.2.3; plan written.
- 2026-10-10 08:37: P1 hop, film-look switches, event emitter, glare + haze, overhead infrastructure, narration.json captions; build OK; clearance 0 hits (checker upgraded); v1 tier (16 samples, 540x674) measured at 0.3-0.5 s/frame. Waiting for C / E / R v2 deliverables (polling).
- 2026-10-10 08:50: R v2 bird racks integrated (240 slots from hall_layout.json v2; 239 crowd + hero front slot L07 tier 0; neck extension per slot from `neck_extension_m` (0 / 0.046 / 0.126 m) as an iridescent neck segment per bird and the head instancer moved forward by it; head-motion pivot follows the extension); 421 SFX events; clearance 0 hits; hall stills checked. Waiting for C and E.
- 2026-10-10 09:05: A's requests (coordinator): S5 beats moved earlier to the new VO (explode 1.6-2.4 s, DSP labels 2.5 s, stamps from 2.3 s, reassemble 7.8-8.3 s, band label 8.2 s); captions read from narration.json (S5 lines 41.30 / 44.20 / 48.20); scene_events.json: `pan` from camera space (287 positional events; hero / flyer wing flaps follow the animated root), hall ambience split around the macro sets (6.0-21.9, 28.6-31.9, 36.4-50.0 s), A's provisional kinds emitted at real times (whoosh, whip, beam_launch, pd_flash, pam4_chatter, hud_tick, conveyor, assembly_clunk, robot_servo, door, flap_vent via alias, tia_hum), no leash_pull; 451 events.
- 2026-10-10 09:09: clearance 0 hits (every frame); stills every second (62) checked; S7 VO note hidden in film mode. Film v1 render started at the v1 tier (16 samples, 540x674), per scene with encode and frame deletion. C and E still in progress: Subject, pigeon rig/behaviours and E street/VFX are placeholders in v1. (Note: the shared scratchpad's earlier contact sheets from DevLog-006 were removed by another process.)
- 2026-10-10 09:20: E and C delivered; production integration written (C rigs/behaviours/Subject, E street/hatchery/VFX/AgX, hatchery flashback) replacing the crude bird/crowd/Subject; build 30 s; first stills: Subject arrives late (walk speed 0.92 m/s baked) -> brisk 1.5 m/s; hall too dark with shadows + AgX -> hall lights rebalanced; hatchery objects outside the root -> reparented; text bloom -> overlay emission 1.5.
- 2026-10-10 09:40: clearance checker refined (surface distance only; sky dome, haze and fake light cones ignored): 0 hits every frame. Film v1 render started per scene (about 1.6 h).
- 2026-10-10 10:32: coordinator stopped the slow render (owner: fast render until v2; <= 10 min per film). Fast tier added (FILM_TIER / FILM_FAST / FILM_SHADOWS / FILM_COMP), hall and its lights windowed out of S1 0-5 s; per-scene timing table measured.
- 2026-10-10 10:43: film v1 rendered in 3 parallel chunks (118 s wall), encoded `outputs/film_v1_20261010.mp4`; 1 fps sheets checked; partial s2 frames from the stopped render deleted.
- 2026-10-10 11:00-14:55: film v2 (section 8). v1 builder frozen as `scripts/build_film_v1.py`. Integrated R hero cell, C hold_bird / Gary outfit / neck deformation / S6 helpers, H keyable materials, E hatchery v2 / constellation / beams / data look / HUD, Tree intro, VO rev 3. Six build-render-review rounds (stills every 10 frames, NDC and Subject-camera distance reports in the build log). Final: build 3 min, render 193 s (3 chunks) + intro pass, clearance 0 hits, encode with the intro cross-dissolve.

## 8. Film v2 (DevLog-010 section 9 + DevLog-015 interim items, 2026-10-10)

### 8.1 Outputs

| Item | Path |
|---|---|
| Video (picture only; A muxes) | `outputs/film_v2_20261010.mp4` (540x674, 30 fps, 1860 frames, 62.0 s, H.264 CRF 16, yuv420p, +faststart) |
| 1 fps sheets | `outputs/film_v2_20261010_sheet_1fps_1.jpg` (0-31 s), `_2.jpg` (32-61 s) |
| Scenes | `scenes/film_v2.blend` (main), `scenes/intro_v2.blend` (intro pass, frames 1-160) |
| Builders | `scripts/build_film.py` (v2), `scripts/build_intro_pass.py` (new), `scripts/check_camera_clearance.py` (skips render-hidden child collections of instances) |
| SFX cue list | `scripts/audio/scene_events.json` (422 events after visibility gating) |
| Credits | `outputs/CREDITS_20261010.md` (photo credits, voice, music/SFX, fonts, sources); compact line on the S7 card |

Encode: intro pass frames 1-160 cross-dissolved into the main pass: `[1:v]trim=start=4.5,setpts=PTS-STARTPTS[m];[0:v][m]xfade=transition=fade:duration=0.8:offset=4.5` (the main pass is black until 4.4 s). Render wall: main 193 s in 3 parallel chunks + intro 160 frames (one process) + encode; well inside 10 min.

### 8.2 Work order status (S-owned items)

| # | Done in v2 |
|---|---|
| 1 | S6 restaged: Subject central third 52-59 s (NDC x within +-0.15); 4 perched birds 2-4 m from his mark (rail_a, comp on perch_hero, rail_b, scooter) turn to him (forced observe windows); ends on the hero perched in the foreground (scooter bar) with Gary in startle_freeze behind; TX blink 59.4 s |
| 2 | 7.97 s whip removed: one continuous arc pad -> in front of the cell -> -y side; clear hero shot on 'Fancy name for a pigeon' (hero ~30-50 % of width); ends with Gary walking in behind; camera >= 1.97 m from his head in S1 |
| 3 | R's hero cell placed; L07 rack is a real copy; 23 occluding parts + R's CUTAWAY wall dissolve (BLENDED twin materials) 17.2-17.85 s and back 20.1-20.75 s; continuous pull-out into S3; 3 crowd birds that sat between the cutaway camera and the hero removed (L07 HOOK_bird_1_1_0, HOOK_cell_1_1_0, HOOK_cell_2_0_0) |
| 4 | C hold_bird: Gary reaches in, hero hops onto the palm (12.6-13.4 s), flops on its back, hand presents belly/feet, back into the cell at 18.4 s |
| 5 | Right pads + right eye rim red at 14.6 s, left cyan at 16.3 s (VO rev 3 anchors); R host pads + hero GND (talon tips) / VCC (belly pads) / signal (toe pads) mate at 18.41 / 18.66 / 18.91 s |
| 6 | Hero spot light (140 W, 16 deg) in rack shots; H band glow 3.0, neck rim_strength 1.8; crowd materials x0.55 and no band/rim; hero >= 25 % width in the S1-S2 rack shots |
| 7 | S4 0-1 s: ~70 deg orbit (r 0.5 -> 1.3 m) during the head turn ending on a two-shot with the incoming bundle and the sender; hero is only ~16 % width there |
| 8 | Exploded hero turned (yaw 180) so the camera looks at the back wall (Gary leaves through the door at 44.2 s, out of shot); short parody call-outs in two columns inside the frame; wing panels fade to 15 % (blended); DSP lid stamp NARROWCOM at 42.6 s, MERVALL at 43.4 s (H's .blend material on a twin); MCU push 44.5-46 s; telemetry as a large mid-left HUD |
| 9 | Leg-band close-up with a band glow pulse at the snap-back; white flash 7.92-8.22 s only; E v2 hatchery loop shifted so the hopper hops at 48.55 s; 3/4 camera on the moving line + hopper |
| 10 | Superseded by F8 (tree removed from the street); Subject in the lower third while the street fades in |
| 11 | Headers top 15 % (LAB y 2.3, size 0.145), captions bottom 18 % (CAP y -1.95), beam note size 0.1 (~14 px) at the very bottom; contact labels with leader lines and dots |
| 12 | PD array shot 32.9-34.0 s (8 beams landing on the PDs, GSG bonds); Si interleaver plates grey-blue with Fresnel edge emission |
| 13 | LEO train (E generator, closer pass) with a tracking uplink beam handing off at 55.22 / 55.68 / 56.24 s; CCTV links from 57.8 s ('cameras' 57.86); beam note only while beams are on screen |
| 14 | Positional SFX gated on camera visibility (+-1 s); explode / snap_back gain 1.6; DSP stamp ticks 42.6 / 43.4 s; hud_lock 37.55 s; Subject exits the door at 44.23 s (before 50.25 s) |
| 15 | Follow cam ~1.5 m behind the hero to the pad; pad glow; landing pad + DR8 labels as screen headers after the hero reads |
| misc | 'Meet the OSFP' lands on the intro hero card; beak latch close-up on R's lip at s2b; fill out 0.3 s earlier (27.98 s); TIA held to 36.5 s; Gary glances at the lens on 'Hi, Gary'; no ghost wing into S6; flyer near-miss fixed (camera 1.8 m behind); S7 card centred |

Interim items (DevLog-015): F2 isolator label in the S3 macro; F4 E's intel HUD (lock 37.55 s); F5 parody call-outs; F6 bird-to-bird beam hops + street sweep + satellite handoffs; F7 data emitter on the hero TX lenslets (S3-S4 head turn) and PAM4 data beams in both macro sets and the hall bundles; F8 intro pass + cross-dissolve, hologram tree and projector removed from the street; F1 C's neck deformation picked up by the rebuild.

### 8.3 Checks (final build)

- Clearance (step 2, margin 3 mm): 0 hits.
- Subject head to camera (min per scene): S1 1.97 m, S2 0.44 m (beak close-up 12.2 s, Gary out of frame), S3 1.46, S4 1.48, S5 1.42, S6 4.02 m.
- Frame-difference pop check (mean abs grey diff, 108x134): the v1 pop frames now read 182: 32.2 (v1 50.0), 240: 29.1 (52.6), 518: 32.4 (~53), 625: 24.0 (57.9). Max outside fill/jump windows: 45.7 at 6.33 s (camera through the flap, continuous) and 44.3 at 40.7 s (S5 opening pan from the POV exit to the hero, continuous). Inside fills: up to 106 at 48.0 s (white flash into the hatchery, intended).

### 8.4 Known issues (v2)

1. S4 orbit end: the hero is small (~16 % width) in the two-shot; a true over-the-shoulder is impossible at the rack face.
2. S2 hand-hold views are busy (crowd birds behind the hand); the belly orbit is small because the cameras are clamped to the aisle.
3. Satellites read as small panels with a bright glint; the uplink beam is clear.
4. RX macro: the optical bench reads as a large dark/white slab in the 32.5-33 s move.
5. S5 exploded view: wide framing keeps all labels inside but the parts are small; label text ~8-9 px.
6. `outputs/film_v2_20261010_av.mp4` (14:47) was muxed from an earlier v2 render; A must re-mux from the 14:54 video.

## 9. Film v3 (DevLog-018 feedback wave, 2026-10-10)

Builder: `scripts/build_film.py` (v2 frozen as `scripts/build_film_v2.py`); scenes `scenes/film_v3.blend` (built at FILM_TIER=review, FILM_SAMPLES=4, glare) and `scenes/intro_v3.blend`.

### 9.1 S-owned items

| Item | Done |
|---|---|
| ~0:16 hero + Gary to the right | Gary's mark 0.8 m out from the cell (was 0.6); the hand-held belly views now have the aisle behind them, off the shelf pigeons |
| ~0:31 RX entry | Head-turn arc in front of the hero, then the camera enters the RX eye on the mirror of the TX exit path (e + ax*0.5 - side*0.1 -> e + ax*0.06 - side*0.03 -> eye); incoming bundle ends at 31.45 s so the camera does not fly through it |
| S5 slow orbit | One ~50 deg orbit (r 0.95 m, lens 38) 42.45-47.3 s, no push-ins; microcontroller telemetry and MCU / power / crystal / EEPROM labels removed; call-outs on the VO rev 4 vendor anchors (DSP 41.9, PDs INCOHERENT 44.2, TIA SEMTREK / MACOMBO 46.1) |
| S6 beams | 4 birds sweep continuously (12 sweep segments, beam_sweep) + 3 point-to-point links switching partners (beam_hops, ~1.3-1.5 s holds), 52.4-59.3 s; E's mimo_links (4 roof/ledge/wire birds, k = 2, inter-satellite links) under 'Satellite handoff. Again. Again.'; 0.5x train on E's doubled pass (2x speed); cable car climbs during S6; SF backdrop from E's rebuilt street |
| Gary unaware | No lens glance; S6 C stroll_unaware (coffee sips, phone) along the street; no startle / freeze |
| OUNOLIGHT | All S text; H's leg band text object renamed by H |
| VO rev 4 | Captions from narration.json; beats moved: beak bite 12.9 s; DFB on 21.05 s, light front through the MMI tree 22.7-24.4 s, RF glow from 24.95 s; PD spots 32-33.4 s, TIA signal 34-37 s; DSP stamps 42.6 / 43.4 s; assembly eyes-on 49.4 s; 'Internet of Birds' wide shot 58.4-59.05 s; TX blink 59.55 s |
| Feet | C's place_on_floor on all 234 rack birds with a BVH of the bird racks + hero cell: min sole support 1.00, max back-shift 48 mm (neck extended to keep heads in place) |
| T / X / M / Tree | T light-up + RF glow layer in the S3 macro (camera pan follows the front order); X cleaned fixtures + traced cyan light + amber signal glow in the S4 macro (my straight beams hidden); M assembly hatchery at (0, 0, -25) with its 3-shot camera at speed 1.6 (47.85-49.75 s), short white fills at the cuts, beam note on shot A, extra fill lights; Tree cards via the intro pass |

### 9.2 Fixes found while integrating

- Overlay text garbled in the 450-600 mm assembly macro shots: float32 precision at x = 600 m (camera and the tiny overlay plane). The assembly set moved near the origin; text clean.
- Clearance checker: animated curves are now tested on their evaluated geometry. The S4_tim_bead "hit" was a 5 m curve bounding box with 0 evaluated vertices. Final clearance: 0 hits.
- S6 wide shot kept on the street side of the facade (first version sat behind the DC wall).

### 9.3 Render

Review tier 1080x1350, 4 samples, glare; about 1 s per frame per process. Coordinator rendered main frames 135-780, 781-1430, 1505-1860 from a snapshot of the 16:34 build; S rendered the intro pass (1-160) and the hatchery window 1431-1504 (adds the assembly fill lights) from the 16:40 build.

### 9.4 Result and checks

- Video: `outputs/film_v3_20261010.mp4` (1080x1350, 1860 frames, 62.0 s) and `outputs/film_v3_20261010_av.mp4` (muxed with A's mix). The coordinator assembled both from the frames: intro xfade at 4.5 s for 0.8 s. S's own assembly step was skipped on the coordinator's instruction.
- Clearance on the final build (every 2nd frame, 3 mm margin): 0 hits.
- `scripts/audio/scene_events.json` was written by the final build: 434 events after gating. New kinds: dfb_on, pic_lightup, elec_glow, beam_sweep (with pan + params.pan_end), mimo_reconfig, align_stage, uv_cure, wire_bonder (params.count / step), lid_press.
- Subject head to camera, min per scene: S1 1.98, S2 0.57 (hand-hold, Gary out of frame), S3 1.47, S4 1.35, S5 1.42, S6 3.78 m.

### 9.5 Known issues (v3)

1. S3 21.0-21.6 s: the first TX-macro frames are dark. The DFB has only just come on, the GLB beams are hidden, and the camera opens on the dim carrier.
2. Assembly shot A (47.9-48.9 s) is still dim and purple even with the added fill lights; shots B and C read.
3. S5 orbit (42.5-47.3 s): the exploded parts are small chips; at lens 38 the hero sits off centre for part of the orbit.
4. S6 street birds are not planted with C's plant_on_surface / plant_on_wire yet (only the 234 rack birds use place_on_floor).
5. The RX macro cleaned-fixture glass still reads as large beige slabs in the 32.5-33 s move.
6. The main-pass frames 135-780 / 781-1430 / 1505-1860 came from the coordinator's 16:34 snapshot. The later S changes (HOLD revert, assembly at the origin, fill lights) only affect 1431-1504, which S rendered from the final build.

## 10. Film v4 (DevLog-020 feedback wave, 2026-10-10)

Builder `scripts/build_film.py` (v3 frozen as `scripts/build_film_v3.py`); scenes `scenes/film_v4d.blend` (review tier: 1080x1350, 4 samples, glare) and `scenes/intro_v4.blend`. Earlier v4 builds: v4 (17:26), v4b (assembly v4 timing), v4c (billboards, robotaxi). v4d adds R's GB300 cabling, the cutaway camera fix, and the 3.0 s card.

### 10.1 Timeline (64.0 s, 1920 frames)

| Scene | Frames | Film time | Notes |
|---|---|---|---|
| Intro pass | 1-160 | 0-5.33 s | separate pass (Standard view); main pass cross-dissolved in at 4.5 s for 0.8 s |
| S1 | 1-300 | 0-10 s | main pass used from 136 (4.5 s) |
| S2 | 301-600 | 10-20 s | |
| S3 | 601-900 | 20-30 s | TX macro 631-849 |
| S4 | 901-1200 | 30-40 s | RX macro 961-1109; POV 1110-1200 |
| S5 | 1201-1530 | 40-51 s | 11 s: exploded orbit 1201-1436, assembly slot 1437-1528 (92 frames, M speed 1.5652), white flash into S6 |
| S6 | 1531-1830 | 51-61 s | everything +30 frames vs v3; scooter-bird zoom ends at 61.0 s |
| S7 card | 1831-1920 | 61-64 s | 3.0 s; disclaimer text exactly as A re-voices it |

Implementation: `F(idx, t)` in build_film.py adds S5_EXTRA = 30 frames for scenes >= 5 (also set as blender_lib's F); S5 windows end at S5_END = 11 s; `EV()` adds +1.0 s for scenes >= 5; FILM_S7_LEN sets the card length.

### 10.2 Items

| Item | Done |
|---|---|
| EQUONOX | Street sets regenerated through bslot (backup `outputs/backups/street_sets_v3_20261010.zip`), then E's billboard rebuild (17:27) is used; the crude facade text renamed |
| No rendered captions | narr_from_json not called; intro pass has none; beam note, HUD, scene labels and the card text stay |
| ~0:36 | "OCTAL TIA" label (spiral inductor label dropped) |
| S5 orbit | Closer: r 0.62 m, lens 30, about 34 deg over 4.85 s, centred on the two label columns; labels PIN PDs: MACOMBO (44.2 s), TIA: SEMTREK (45.4 s) |
| Assembly +1.0 s | M v4: 144 build frames in a 92-frame slot from 47.85 s; beam note on shots A and C; SFX from AA.event_frame (dfb_on, align_stage, uv_cure, wire_bonder, lid_press, eye_glow_on, hop) |
| Scooter-bird zoom | 60.45 -> 61.0 s, lens 32 -> 60 into the hero on HOOK_perch_x_scooter_bar (TX blink inside the zoom), cut to the card at 61.0 s |
| X RX bits | rx_signal_glow starts at the first PD landing (frame 999, 33.27 s) |
| R GB300 | rebuilt; S2 cutaway camera moved +0.125 m in x, clear of L06's 0.12 m front cabling; hand-held cams clamped to x >= -0.93 |
| E billboards / robotaxi | picked up via append_env; robotaxi keyed along HOOK_robotaxi path over S6; CCTV shot lens 24 -> 21 so SIGMOID / AGENTS. stay in frame |
| Street birds | planted with C's plant_on_surface on a BVH of the ending street instancers |
| Intro band | camera and target lowered together (content about 9 % higher), soft dark gradient over the bottom ~19 %; hero card ending above it |

### 10.3 Checks

- Clearance v4d (every 2nd frame, 3 mm margin): 0 hits. v4c had 6 hits at S2 7.1-8.3 s against R's new L06 front cabling; fixed in v4d.
- `scripts/audio/scene_events.json` (v4d build): 429 events, duration 64.0 s.

### 10.4 Render split (review tier)

S: intro 1-160, S1 hall 169-300, card 1831-1920 (v4d). Already rendered from film_v4 before the later changes (unaffected): TX macro 632-849, RX macro 962-1109. Coordinator render agents: the remaining v4c / v4d ranges.

### 10.5 v4e / v4f patches (2026-10-10 evening)

- Owner: rendered labels "horrible" -> all 3D callouts, scene headers and the beam note left the render. A draws them in post from `outputs/film_v4_callouts.json` (per-frame anchors of 11 labels from world_to_camera_view on the parts, fixed L/R side, fade a) and `outputs/film_v4_texts.json` (per-frame header / beam-note text, 1322 frames). Macro-set labels in the S3/S4 frames rendered earlier stay as rendered.
- S5 orbit: exploded hero in the aisle centre (out x -0.05), framed on the visible exploded parts (bbox 6..79 x 0..121 mm), r 0.35 m, lens 30, slow 30 deg orbit; EEVEE DOF (f/3.2 on a focus empty) plus hall lights at 42 % during the labelled windows (S2 cutaway 541-592, S5 orbit 1259-1421); extra key light on the exploded hero.
- S2 cutaway: camera hp + (0.16, -0.48, 0.19); L06 is a real copy whose fibre / Cat6 / velcro bundles dissolve with the cutaway occluders (only in the cutaway window, so no change elsewhere). Clearance reports the faded (opacity 0) Cat6 copy at 553-603 only.
- Intro v4f: opaque black band over the bottom 20 % (card wall only above 0.80 H), opening key lowered so the root row starts above the band; no captions.
- Renders: final main pass 136-1920 rendered by the coordinator's workers from film_v4f.blend; S rendered intro 1-160 (intro_v4f). A assembles: intro xfade, callouts, headers / beam note, captions, mux.
- Open: the hero has no micro-isolator part, so "MICRO ISOLATOR (FARADAY)" has no callout anchor.

### 10.6 v4g / v4h / v4i patches (2026-10-10 night)

- v4g: TX macro empty at 631-654 ("TOSA not loaded" near 0:21). Cause: the v3 timing edit had gone into the unused `build_tx_macro` and not into `build_tx_sub`. Fixed with t0 = 0.9 in `build_tx_sub`; cam_jump puts VECTOR handles on the post-jump key. Macro set labels removed from the render and exported as callout anchors. Street ending visible only at S5_END-0.25..S5_END (this removed the floor streaks during the assembly).
- v4h: owner said the assembling and wire-bonding scenes were blurred. Cause: DOF was on for the whole film. `use_dof` is now keyed CONSTANT, True only in the S5 orbit window 1259-1421. The S2 cutaway DOF and the hall-light dimming are removed. Owner: callouts only in the exploded view, so `film_v4_callouts.json` is filtered to S5 (8 labels, frames 1258-1426).
- v4i: M's v4.1 test stand (after `assembly_v41.READY`): shots A/B/C/D, 906 objects, f0 1437, speed 144/92, 13 camera keys. Beam note on shot A (1439-1461) and shot D (1493-1521) is in `film_v4_texts.json` (28 texts, 1322 frames). scene_events.json was rewritten by the build.
- Re-renders copied into `outputs/film_v4_frames`: 517-626 (v4h, 110), 631-849 (v4g, 219), 961-1109 (v4g worker, 149), 1437-1528 (v4i, 92). Main pass 136-1920 complete (1785 frames).
- Checked by eye: f0560 / f0600 / f0700 / f0800 / f1000 / f1060 sharp, no labels; f1300 background blurred (callouts drawn in post); f1450 / f1470 sharp; f1500 / f1515 stand screens readable, no floor streaks.

### 10.7 S5 slot 2x (v4k)

Owner W17 (DevLog-020): assembly slot 92 -> 184 film frames at the same speed 144/92 = 1.5652 (M v4.2, about 288 build frames). Owned by the S5 slot agent; S's label restoration (v4j) is a separate change. Add-ons from the coordinator: W18 satellites aligned (E's `constellation_train` default jitter 0.0; S6 satellite frames re-rendered) and the original S7 disclaimer text restored verbatim.

Frame map v4k (2012 frames, 67.07 s; from `F()`):

| Scene | v4 frames | v4k frames | Notes |
|---|---|---|---|
| S1-S4 | 1-1200 | 1-1200 | unchanged |
| S5 | 1201-1530 | 1201-1622 | orbit 1201-1436 unchanged; assembly slot 1437-1620 (184); S5_END 14.0667 s, flash 1610-1622, cut to the S6 camera at 1616 |
| S6 | 1531-1830 | 1623-1922 | +92 |
| S7 card | 1831-1920 | 1923-2012 | +92, 3.0 s |

Code (`scripts/build_film.py`, top and the S5 assembly block and the S7 card only): `ASM_SLOT = 184`, `S5_EXTRA = 30 + (ASM_SLOT - 92) = 122`, `S5_END = 10 + S5_EXTRA / 30`; build prints M's build length vs the slot; S7 DISC text = the v3 / crude original. Checked by reading: every S5 tail key (white flash, cut to S6, street window, hall window, beam note end, header end, flashback_out SFX, COMP blocked window) uses S5_END; all S6 / S7 keys, windows and events use `F(5|6, t)` / `EV(5|6, t)`, so they shift by +92 exactly (rounding checked for every S5_END - d used: +92). Not in build_film.py: `scripts/post/assemble_film.py` defaults `--end 1920 --dur 64.0` (A passes the v4k values); `callout_overlay.py --callout-range 1201-1436` unchanged.

Plan / TODO:
- [x] code change (S5_EXTRA 122, S5_END, ASM_SLOT, card text)
- [x] backup of texts / callouts / scene_events JSON before the v4k build (`outputs/backups/v4j_json_before_v4k_*.zip`)
- [x] wait for M v4.2 (`assembly_hatchery.json` duration_frames > 144)
- [x] build `scenes/film_v4k.blend` (FILM_TIER=review FILM_SAMPLES=4 FILM_S7_LEN=3.0, --factory-startup), `outputs/film_v4k.SAVED`
- [x] move old card frames f1831-1920 aside (backup folder), rename f1529-1830 -> f1621-1922 top-down
- [x] render from v4k: assembly 1437-1620, satellites 1731-1868 (sats visible F(5,3.6)-F(5,8.2)), card 1923-2012
- [x] verify 136-2012 (1877 files, 1080x1350), view f1450 / f1520 / f1580 / f1610 / f1625 / satellites / f1925; texts JSON covers 1437-1620 and S6 +92
- [x] `outputs/film_v4k_frames.READY`

Progress:
- 2026-10-10 22:20: code change done; waiting for M v4.2.

### 10.7 v4j: in-place 3D labels restored (2026-10-10 night)

- Owner: only the additional post callouts were to go; the original floating in-place labels must stay. Restored the v3/v4d wlab code exactly: TX macro (7 labels, size 0.16), RX macro (5 labels, size 0.15, "OCTAL TIA" without the spiral wording), S2 cutaway contact labels with leader lines and dots (size 0.0115). Callouts JSON stays S5-only. Backup of the v4i builder: `scripts/build_film_v4i.py`.
- Rendered from film_v4j.blend into `outputs/film_v4_frames`: 517-626 and 631-849 (S); 961-1109 (coordinator worker). Markers: `film_v4j.SAVED`, `build_film_labels.DONE`, `film_v4j_frames.READY`.
- Known: with the v4e cutaway camera (closer than v4d's), the cutaway labels "1 GND: TALON TIPS (LONGEST PADS)" and "3 SIGNAL: TOE PADS (TX / RX)" run off the right edge. Some TX macro labels at the frame edge match the v4d placement.
- 2026-10-10 22:21: M v4.2 delivered (288 build frames, shots A 0-71 / B 72-121 / C 122-167 / D 168-287, per-bond events S3_bond_01..34). Wire-bond SFX now the bonds inside shot B from the sidecar (10 bonds, step 0.106 s; fallback = old 6). Build `build_film_v4k_20261010_222102.log`: slot 184 = 288 / 1.5652, f0 1437, 20 camera keys, 937 objects; subject S6 mark 1847 -> 1939 (+92); `outputs/film_v4k.SAVED`.
- 2026-10-10 22:29: old card frames f1831-1920 moved to `outputs/backups/film_v4_card_frames_v4j/` (90); f1529-1830 renamed top-down to f1621-1922 (302). Old f1529-1530 are the S5_END-relative flash tail, so they map to f1621-1622 unchanged.
- 2026-10-10 22:30-22:34: rendered from film_v4k: 1437-1620 (184), card 1923-2012 (90, original disclaimer), satellites 1731-1868 (138; sats visible F(5,3.6)-F(5,8.2); jitter 0). Logs `render_v4k_{asm,card,sats}_*.log`, 0 errors.
- Checks: 136-2012 = 1877 files, all 1080x1350. texts JSON (1414 frames): pre-1437 identical, all 200 old S6 entries at +92, header 1439-1614, beam note 1439-1482 (shot A) + 1546-1613 (shot D); callouts JSON identical; scene_events 67.067 s, 425 events. Viewed f1450 / f1520 / f1580 / f1610 (sharp, TX eye per lane, PASS stand), f1625 (S6 hall), f1778-1822 (satellites in one straight line, identical body / panel attitude), f1925 (full disclaimer, legible). Pre-existing: f1619 (v4 f1527, also in the delivered v4 mp4) has a dark band in the top third inside the white flash at the S6 camera cut; not changed.
- `outputs/film_v4k_frames.READY` written. A: `assemble_film.py` needs `--end 2012 --dur 67.067` (defaults are 1920 / 64.0).
