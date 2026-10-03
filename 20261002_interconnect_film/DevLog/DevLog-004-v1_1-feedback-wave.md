# DevLog-004: v1.1 feedback wave (physics and effects pass)

Date: 2026-10-02. Input: Wentao's timestamped review of `outputs/film_v1_0_draft_p50_20261002.mp4`. Output target: v1.1 (draft p50 first, then standard 1080x1350).

## 1. Feedback (verbatim intent) mapped to scene owner

Film time T = scene index * 10 s + scene-local t. S1 0-10, S2 10-20, S3 20-30, S4 30-40, S5 40-50, S6 50-60, S7 60-62.

| T | Scene (local t) | Request |
|---|---|---|
| 0:01 | S1 (1) | Opening: zoom OUT from inside the copper cable (pockets of electrons flying left and right) to the two-rack scene |
| 0:07 | S1 (7) | Manager walks all the way to the scope; the angry shot must show the scope traces |
| 0:10 | S1 (10) | Signal-fading line: fast zoom from Gary + scope both visible onto the scope itself |
| 0:12 | S2 (2) | More varied dummy components on the board (refs: GB300 compute tray, NVLink switch tray, AMD Helios tray); each connector gets TWO columns of retimers; NVL72 backplane cartridges visible in these shots |
| 0:16 | S2 (6) | Rack next to the bench and scope, cables between rack and scope |
| 0:18 | S2 (8) | Camera zooms in on the whiteboard more than at 0:16, Manager still in frame and turns angry |
| 0:20-0:25 | S3 (0-5) | Improve the fight animations |
| 0:26 | S3 (6) | "+3 months" accompanied by exterior with very quick sunrise/sunset/moon/stars, cycling about 3 times |
| 0:28-0:29 | S3 (8-9) | Stop frequent camera position changes; one position with everyone in frame |
| 0:31 | S4 (1) | OEs move along a fancier path: up and around the stiffener frame, then down onto the substrate |
| 0:33 | S4 (3) | Fade in/out unnatural: no black screen, fade directly into the next scene with the PIC |
| 0:34 | S4 (4) | Heat wave comes in 3 times (single pass reads as lighting) |
| 0:36 | S4 (6) | Zoom closer on the actual OEs, not only connectors; eggs land directly on the OEs |
| 0:38 | S4 (8) | Overcook eggs more: only a tiny bit of white left near the yolk |
| 0:39 | S4 (9) | Same whiteboard and workbench scene; whiteboard shows latency and energy back down from previous curve; more than 3 eggs thrown, use all 16: one on Manager head, one on the shoulder nearer the camera, one on chest, the rest fall on the ground around him |
| 0:41 | S5 (1) | Tools spaced much closer; camera flies closer to see more detail incl. interior; no linear motion: ease so more time is spent at each tool |
| 0:44 | S5 (4) | Monitor image rotated 90 deg wrong: resonance dip must point down, not right |
| 0:48 | S5 (8) | The wafer-level tester must be in this scene |
| 0:51 | S6 (1) | Use MPO connectors and receptacles |
| 0:58 | S6 (8) | Proper physics for the cable whip |
| 0:59 | S6 (9) | NVYDIA guy wears a leather jacket (like Jensen Huang; clay stylization, no real-person likeness, no logo) |

Plus: "start improving physics and various effects of the animation".

## 2. Cross-cutting physics and effects spec (all scenes)

Use Blender physics baked to keyframes or caches in the scene blend, deterministic, fixed seeds; headless rendering must reproduce them. Where Blender's solver is unstable headless, use a small deterministic Python simulation (numpy, verlet / spring) baked into F-curves.

- Eggs and thrown objects: rigid body (gravity, bounce, friction), splat decal/shape change on impact, yolk wobble (spring), fry state from the hot-chip contact.
- Cable whip (S6) and Manager's hanging cable gags: chain simulation (verlet with bending stiffness) baked to the curve/armature, whip crack travelling wave, Gary reacts on contact.
- Fights, hits (S3, S1/S2 shots): anticipation / overshoot, squash and stretch on clay bodies, hit-stop frames, camera shake on impact (noise F-curve modifier), dust/clay-crumb particle puffs, secondary motion (tie, shirt, hair as spring-driven bones or damped follow).
- Camera: eased moves (bezier / ease in-out), subtle handheld noise, motion blur on in the final render (shutter 0.5), no hard linear moves.
- Particles and glow: electron pockets (S1), heat shimmer pulses (S4), sparks, muzzle smoke and shell clay chips for shots.
- Keep render cost: draft 8 spp at 50 percent must stay under about 1.5 s per frame; flag anything slower.

## 3. Ownership for this wave (one agent per scene)

Each agent owns its scene script/blend/devlog (`scripts/film_v1/sNN_*.py`, `scenes/v1/sNN_*.blend`, `DevLog/v1/DevLog-003-scene-sNN.md`, generated textures under `assets/generated_textures/v1/sNN/`). New assets go in new files only (never overwrite existing asset blends). `asm.py` changes are allowed only as additive, backward-compatible edits by the coordinator; agents record needed changes in their devlog and work around them.

- S1: 0:01, 0:07, 0:10 + physics (electrons, fight/hit).
- S2: 0:12, 0:16, 0:18 + new dummy-component assets (GB300-type compute tray, NVLink switch tray, Helios-type tray) as new files under `assets/components/datacenter/` and `scripts/assets/datacenter/`.
- S3: 0:20-0:29 fight rework, day/night cycle, stable camera.
- S4: 0:31-0:39.
- S5: 0:41, 0:44, 0:48.
- S6: 0:51, 0:58, 0:59 (leather-jacket variant of the NVYDIA character as a new asset file; S3 and S6 both use it, so S6 publishes it as `assets/components/characters/npc_nvydia_leather.blend` and records the swap in its devlog; the coordinator patches S3).

## 4. Status

| Item | Status |
|---|---|
| Plan written | done 2026-10-02 |
| S1..S6 agents | done 2026-10-02 (self-checked by contact sheets; no fresh-context audit yet) |
| Draft p50 re-render + concat | done: `outputs/film_v1_1_draft_p50_20261002.mp4` (62 s, 540x674; S7 reused from v1.0) |
| INDEX.md regenerated | done (93 rows) |
| Wentao review | pending |

Known deviations / open points (details in per-scene devlogs):
- S3 0:26: no full exterior cutaway; roof lifts off and 3 day/night cycles run through the open wall, because the die-tower/calendar/balloon gags occupy that window. Open question: move gags and do a true exterior cutaway?
- S2: NVL72 cartridge band is on the backplane front face (real ones are at the rack rear); fans drawn at 60 percent height; tray layouts generic (no photos available).
- S4: true refraction shimmer not possible with the compositor replaced at render; heat shown as colour pulses plus a rig wobble.
- S5: oven lids do not open; hole 5 small at draft.
- S6: MPO housing/adapter dimensions are level C estimates (ferrule face level A); no MPO dressing on the pile or ribbon cable.
- S3 cast has no NVYDIA, so the leather jacket appears in S6 only.
- Framework wishes recorded by agents: eased/noisy/multi-stop `asm.shot`, NLA hit-stop helper, per-object show windows.
- Scene boundaries are hard cuts (no overlap dissolve) to keep the 10 s timing; only S4's internal black fade was removed.
- render_scene.py now preserves the scene's own motion blur through render presets (blur is on in S1, S2, S5 blends at shutter 0.5; S3/S4/S6 leave it to the hero preset).

## 5. Narration voice pass (2026-10-02)

- Voice: Kokoro-82M via kokoro-onnx (local, Apache-2.0 weights from the kokoro-onnx release `model-files-v1.0`; not stored in the repo), voice `am_michael`, base speed 1.12, chosen for the style bible's "fast, crisp, confident young male voice" (DevLog-000). No cloning of any real voice. Processing: high-pass 90 Hz, +2.5 dB presence at 3 kHz, compressor, loudnorm -16 LUFS.
- Script: `scripts/audio/make_vo.py` (segment windows follow the `asm.narr` calls of each scene; narration text per DevLog-001 Section 5; over-long lines are sped up with atempo, never slowed). Output `outputs/audio/vo_am_michael_20261002.wav`; muxed film `outputs/film_v1_1_draft_p50_vo_20261002.mp4` (video stream copied, AAC 160 kb/s).
- Check: Whisper (base.en) transcribes all 19 narration lines correctly. The S7 disclaimer (21 words in 2.0 s, about 2.3x speed-up) is deliberately ad-fast; Whisper small.en gets "results ... vary by standard ... void where copper's cheaper" only partly, which is the intended gag.
- Spoken disclaimer text is the DevLog-001 text ("...vendor and mood. Void where copper is cheaper."), which differs from the on-card text ("vendor and test condition", "Parody; not affiliated..."). Open: align them.
- Not done: SFX (crack, bang, thump markers exist in S6), music, ducking.

### 5.1 Narration rewrite and voice change (2026-10-02, after Wentao's feedback: "pretty bad, more passionate, constantly going")

- Baseline measurement of the first track (am_michael): 20 silences over 0.1 s, 22.6 s total (36 percent of the film), mainly 2-3 s of dead air at the end of each scene.
- New script (Wentao's wording for S2 retimer, S3 pluggables/balls/R&D bonus, S4 breakfast, S6 fibers/Tuesday/whip; my fill lines for the boss and the shots; "Hole number two / five" callbacks). Spoken text is in `scripts/audio/make_vo.py` (SEGS). Wentao will rewrite further.
- Voice: `am_liam` (Kokoro). Candidates measured with librosa pyin on one test sentence at speed 1.15: median F0 / F0 std / dur = michael 119 Hz / 3.2 st / 7.3 s; puck 102 / 5.0 / 6.8; liam 124 / 4.6 / 5.9; eric 159 / 3.7 / 5.6; fenrir 121 / 5.0 / 6.8; onyx 86 / 2.3 / 6.3. liam chosen as the fastest voice with large pitch variation (a proxy for liveliness, not a listening test). Lines shorter than their window are slowed down to speed floor 1.0 so the voice keeps going.
- Result: 21 silences over 0.1 s, 4.8 s total; remaining gaps over 0.25 s are the beats around the bangs (8.99-9.30, 28.43-29.10, 48.68-49.20) plus 16.55-16.90, 39.92-40.20, 58.55-58.80.
- Outputs: `outputs/audio/vo_am_liam_20261002.wav`, `outputs/film_v1_1_draft_p50_vo_20261002.mp4`. The am_michael versions were removed (my own temp outputs).
- Kokoro has no emotion control; if passion is still lacking the next options are a cloud voice with delivery control or Wentao's own recording.


## 6. SFX pass (2026-10-02)

Files: `scripts/audio/sfx_cues.json` (239 cues; time, type, duration, linear peak gain, pan, params, note with source), `scripts/audio/make_sfx.py` (numpy synthesis + mix, deterministic), output `outputs/audio/sfx_20261002.wav` (48 kHz stereo, 62.000 s, 24-bit PCM). Run from the repo root: `uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx.py`. No samples, no downloads; the voice track is not read or modified. Cue `t` = film seconds of the hit instant (continuous effects: start); gain = linear peak of the normalised effect.

### Cue table (full list with notes in the JSON; repeated cues are grouped)
Gains in this table are the design values; the JSON holds the final gains (non-impact types raised x1.4 after the loudness check, so e.g. chip_click 0.13 appears as 0.182). Scene index = film time // 10. Cue counts: S1 25, S2 32, S3 53, S4 75, S5 34, S6 19, S7 1.

| Film time (s) | Type | Gain | Source of time |
|---|---|---|---|
| 0.35-1.95 (7 cues) | electron_zip, hum 0.3 | 0.05 / 0.035 | s01_copper.py D.add(0,2.2), s01_cam_fx electrons (rhythmic placement in window) |
| 0.75 | whoosh (zoom-out) | 0.12 | s01 shot_zoom_out, fastest near 2.0 |
| 2.2 / 3.0 / 3.6 | blip (5 m / 2 m / 1 m) | 0.06 | s01 GAP_KEYS |
| 4.4 | cable_creak; strand_pop 4.95, 5.35, 5.65, 5.95 | 0.10 / 0.10 | asm.fxn(4.4, 6.0); pop times placed in window (not frame-checked) |
| 5.9, 9.45 | whoosh (scope zooms) | 0.09 / 0.07 | s01 D.simple(6.0..), T_ZOOM0 = 9.45 |
| 7.55, 8.2 | steam_hiss; angry_pop 8.15 | 0.05-0.09 / 0.12 | ear_steam 7.5; p_anger 8.2 |
| 9.2 | shotgun (+pump), clay_thud, hole_pop 9.45 | 0.50 / 0.15 / 0.10 | T_BANG = 9.2 |
| 11.17 ... 14.87 (17 cues) | chip_click x4 per row | 0.10-0.13 | s02 T_ARR, t_row + 0.22 (landing) |
| 12.0-14.65 (7 cues) | whoosh (camera hops) | 0.05 | s02 HOP = 0.20 before T_ARR |
| 15.4 | whoosh (crane out) | 0.12 | s02 cam_rig.shot(5.4, 6.4) |
| 16.4 (2.2 s) | marker_squeak, curve_rise | 0.07 / 0.045 | asm.fxn(6.4, 8.6), GP ramp 6.4-8.6 |
| 18.1 / 18.2 | steam_hiss, angry_pop | 0.07 / 0.10 | ear_steam 8.1; p_anger 8.2 |
| 19.2 | shotgun, clay_thud, hole_pop 19.45 | 0.50 | T_SHOT1 = 9.2 |
| 21.8-22.5 | cloth_rustle x2 (arguing) | 0.05 | brawl argue 1.78-2.5 |
| 22.52-27.88 (about 30 cues) | clay_thud / clay_slap per hit, cloth_rustle tugs, paper_flutter x4, cap_whoosh 25.72 / 25.98, whiff 26.64 | 0.08-0.40 | s03_brawl.compose EVENTS (bump 2.52, headbutt 2.72, shoves 3.18 / 3.46, hooks 4.84 / 4.90 / 5.36 / 6.22, uppercut 5.98, punches), s03_npo PAPER_BURSTS, fx.hat |
| 23.05-23.25 | pkg_flip x3 | 0.12 | s03_npo PKG t0 = 3.0 / 3.1 / 3.2 (clack at t0 + 0.4) |
| 25.4 (1.6 s) | timelapse, balloon_inflate; tick 25.9, 26.4, 26.9 | 0.06 / 0.10 / 0.09 | T_SKY0..T_SKY1, p_inflate_scale, p_flip |
| 26.2, 26.3 | chip_click (die tower), boing | 0.12 / 0.07 | s03_npo tower keys 6.2 |
| 27.514, 28.0 | whoosh + paper_flutter (envelope), clay_slap (catch) | 0.06 / 0.08 | ENV_REL, env_c shown 8.0 |
| 28.3, 28.5, 28.95 (+28.96) | shotgun (no pump / no pump / pump), hole_pop 28.55, 28.75, 29.2 | 0.42 / 0.42 / 0.40 x2 | BANG_M, BANG_N, 8.95 muzzle flashes |
| 29.24, 29.44, 29.65 | clay_thud (body falls) | 0.22 | estimated (shot + 0.94 s) |
| 31.2-31.46 (16 cues) | chip_click (OE touchdown, pan by column) | 0.09 | s04 oe_path t_in + OE_FLY |
| 31.4, 32.6, 34.5 | whoosh (dolly-zoom, OE zoom, pull back) | 0.09-0.12 | s04 asm.shot(1.4/2.6/4.5) |
| 31.5-31.9 | steam_hiss x5 (smoke puffs) | 0.035 | asm.fx smoke_puff 1.5 + 0.1 q |
| 33.37, 33.73, 34.09 | hum with shimmer (heat pulses) | 0.12 | s04 PULSES 3.52 / 3.88 / 4.24 (swell peaks mid-pulse) |
| 35.84-37.77 (16 cues) | egg_splat + sizzle | 0.10 / 0.035 | s04 t_splat (computed from t_land, th) |
| 38.45 | curve_fall | 0.05 | GPROG 8.45-9.85 |
| 38.9 | shotgun, clay_thud, hole_pop 39.15; whoosh (plate) 39.144 | 0.50 | T_BANG = 8.9, T_REL |
| 39.56, 39.64, 39.72 + 8 floor splats 39.62-39.90 | egg_splat | 0.28 / 0.08 | s04 T_HIT; floor times estimated |
| 40.3-43.05 (5) | whoosh (dolly between tools) | 0.10 | s05 DW dwell gaps |
| 42.12, 42.78, 43.0 | sparks | 0.10-0.12 | s05 asm.fx sparks_burst |
| 43.4 | laser_zap | 0.08 | s05 beams 3.4 |
| 43.3 / 43.8 / 44.3 (+0.17, +0.43) | wafer_slide; probe_tick x8 (1/30 s); stage_clunk | 0.14 / 0.08 / 0.12 | s05 t0 = 3.3 + 0.5 w; hop_frames |
| 44.8-46.6 (10) | trace_ping (rising pitch) | 0.06 | tex_gen.spectrum_sequence, 4.8 + k/30 |
| 44.9, 47.1 | whoosh (screen, pull-back); ding 47.2 | 0.07-0.08 | camera_stops; asm.big(7.2) |
| 49.1 | shotgun, clay_thud, hole_pop 49.35 | 0.50 | s05 muzzle_flash 9.1 |
| 50.0 / 50.95 / 51.0 | whoosh (plug glide), plug_jam, rattle x8 | 0.06 / 0.26 / 0.10 | s06 plug_pose keys |
| 51.8 / 52.05 | tray_slide, noodle_spill | 0.12 / 0.10 | asm.fxn(1.5, 3.2); frames 51.4-53.15 checked |
| 54.32, 55.52, 56.72 | tie_zip | 0.10 | s06 BUNDLES tt 4.6 / 5.8 / 7.0 (ending at appearance) |
| 54.5, 55.45 | slither | 0.04 / 0.06 | asm.fxn(4.4, 7.4), thread_fiber 5.4-6.6 |
| 55.6, 57.62 | whoosh (push-in, whip swing) | 0.06 / 0.20 | s06 shots |
| 58.0667 | whip_crack | 0.75 | blend marker SFX_whip_crack frame 243 |
| 58.3 | flesh_thump | 0.55 | marker SFX_whip_hit_thump frame 250 |
| 58.42 | boing | 0.22 | HITSTOP_end 253 (8.40) |
| 58.6 | whoosh (customers shot) | 0.05 | asm.shot(8.6, 10) |
| 59.147 | clay_slap (slap), boing 59.2, squash 59.3 | 0.50 / 0.16 / 0.16 | T_SLAP_START + 11/30 (SLAP flash frame 59.15 checked) |
| 60.0 | ding | 0.08 | disclaimer card |

### Synthesis recipes (all numpy, deterministic seeds)
- shotgun: noise crack (250 Hz-7 kHz, tau 16 ms) + sine boom 36+120 exp(-t/70 ms) Hz (tau 260 ms) + low noise body, tanh drive, 115 ms and 260 ms echoes, convolution reverb from decaying noise IR (1.3 s, per-channel); pump = two resonant clacks (1.25 / 1.9 kHz) at +0.55 and +0.69 s, shell tinks 4.3 kHz with two bounces.
- hole_pop: rising sine 280 to about 1.2 kHz, tau 35 ms, plus click.
- chip_click: band-passed noise click + 2.3 kHz ring (tau 4 ms) + 650 Hz tick, then a 1.7 to 2.5 kHz sine-plus-3rd-harmonic blip (30 ms); n clicks at fixed step, pitch per tray.
- sizzle: Poisson impulse crackle (rate 500/s decaying, tau 0.45 s) band-passed 1.8-9 kHz, rare 500-1400 Hz pops, faint hiss; egg_splat: swept-noise squelch 1300 to 260 Hz + falling sine slap, yolk blip 430 to 260 Hz, 330 Hz wobble with 11 Hz AM.
- curve_rise: sine with FM and 17 Hz tremolo, exponential glide 1.2 kHz x 2^(1.6 u) following the graph ramp; curve_fall: 2.4 kHz to 450 Hz glide with vibrato.
- marker_squeak: sine strokes 2.6-3.6 kHz with 40-60 Hz stick-slip FM and AM, noise band 2-6 kHz.
- whoosh/cap_whoosh/whip swish: filter-bank swept noise (22 log bands, Gaussian centre moving f0 to f1) under an asymmetric sin^2 envelope; cap adds 13 Hz spin AM.
- electron_zip: FM sine glide x4 in 140 ms, moving pan; hum: harmonic stack with 6 Hz warble and swell, optional 2.5-6.5 kHz shimmer.
- cable_creak: stick-slip resonant pulse train (rate 30 to 90 Hz, resonance 450 to 1050 Hz) plus 80-150 Hz groan; strand_pop: falling sine + click.
- steam_hiss: 2.5-7 (9) kHz noise with 12 Hz puff modulation; angry_pop: 180 to 700 Hz sine rise with vibrato then a pop.
- clay_thud: sine 170 to 55 Hz drop + low-passed noise body + click; clay_slap: 0.6-4.2 kHz noise slap + falling sine (+ optional thud); flesh_thump: 120 to 42 Hz drop + band-passed wet noise + squelch.
- cloth_rustle: 1.5-6.5 kHz noise with 4-22 Hz random AM; paper_flutter: decaying random band-passed ticks 20-80 ms apart plus swish.
- timelapse: 3 triangular sine glides 400 Hz to 1.8 kHz with shimmer; tick: resonant click; balloon_inflate: rising squeal + 1.2-5 kHz hiss gated at 5.5 Hz (pumping).
- sparks: dense fast-decaying crackle + chirp; laser_zap: 2.6 to 6.2 kHz chirp; probe_tick: 3.6-5 kHz metallic ring bursts; wafer_slide: swept-noise scrape + 2.25 / 3.45 kHz clink + drawer thunk; stage_clunk: 120 Hz + 720 Hz resonances; trace_ping: bell partials 1 / 2.76 / 5.4.
- tray_slide: 120-1400 Hz scrape with 21 Hz roughness + roll, end stop clunk; noodle_spill: dense decaying 0.9-5.2 kHz tick rain + slither band; slither: 0.5-2.4 kHz noise with 6-8 Hz AM.
- plug_jam: crunch + 1.1 / 2.6 kHz ring + 90 Hz thud; rattle: n resonant plastic clacks; tie_zip: 14 accelerating ratchet ticks + snap.
- whip_crack: 1.5-15 kHz noise tau 6 ms + 5.2 kHz chirp snap + low thump, tanh, 85 and 190 ms slap echoes; boing: 240 Hz carrier with 8 Hz decaying FM and 2nd/3rd harmonics; squash: falling sine 810 to 160 Hz with 38 Hz wobble + squelch; pkg_flip: swept-noise swish + 1.9 kHz clack at +0.35 s; ding: partials 1 / 2 / 3.01, tau 450 ms.
- Mix: each effect normalised, scaled by gain, constant-power panned (params.pan_end moves it), 5 ms lead + 3 ms fade-in, 8 ms fade-out, summed at t - 5 ms, final guard limit 0.70 (not triggered; 0.704 would scale by 0.995 only).

### Checks (2026-10-02)
- Length 62.000 s (ffprobe), 48 kHz, 2 ch, 24-bit; peak -3.2 dBFS (ebur128 true peak -3.2); no NaN, DC -35 uV.
- Loudness: voice -16.1 LUFS integrated (as given); SFX full track -24.2 LUFS (dominated by the bangs); SFX without bangs, whip, flesh thump, slaps, splats, pops, thuds, boings, squash and angry_pop: -29.8 LUFS, i.e. 13.7 dB below the voice. In per-cue isolation the median cue is 23 dB below the local voice RMS (400 ms windows); the only cues above the voice level are the shotguns (+0.1 to +2 dB re local voice RMS at 9.2 / 19.2 / 38.9 / 49.1; 28.3 is 8.8 dB above a quiet gap), as intended.
- Overlay test (scratch, not saved): voice (-1.5 dBFS peak) + SFX summed at unity: peak -0.1 dBFS, no samples at or above 0.999, integrated -15.7 LUFS. Headroom with the voice is only 0.1 dB, so in the final mux put the voice at about -1 dB or run a limiter.
- Onset of the transient matches the cue time within 1 ms for shotgun, whip crack, flesh thump and slap (peak sample of the composite lags 9-40 ms because the boom/tonal part peaks after the crack).
- Spectrogram viewed: 17 percent of 0.1 s windows are below -60 dBFS (silence between effects, no room tone), effects are short and band-limited, bangs show as broadband vertical lines.
- Time checks against the draft video (frames at 0.1-0.25 s step): S6 tray/spill 51.4-53.15, S6 slap 58.95-59.9 (SLAP flash at 59.15-59.25), S3 muzzle flashes at 28.3-29.0 and nubiss on the floor by about 29.6. Brawl hit times were obtained by running s03_brawl.compose with stubbed bpy/Fighter and printing EVENTS (scratch only). Whip crack/hit times are the blend markers (frames 243 / 250).
- Not auditioned by ear (no audio playback here): spectrogram, levels, and timings only.

### Open items
- Uncertain times (+-0.1 to 0.3 s): strand pops 4.95-5.95 (placed inside the FX window), tray slide start 51.8, S3 body-fall thuds 29.24 / 29.44 (shot + estimated fall duration), S4 floor egg splats 39.62-39.90 (egg order is random in s04_cpo.py, only the range 9.60-9.90 is known), S6 hug squash 59.3, S1 scope zoom whoosh at 5.9, vein pop and steam times (derived from p_anger keys, not frames), S1 and S2 body falls after the bang not cued (topple off screen or timing unknown).
- No cue for the S4 whiteboard rising curve: in s04_cpo.py the graph only runs down (GPROG 8.45-9.85); the rising sweep exists for S2 only.
- Hole pops are placed 0.25 s after each bang (the decal appears at the bang frame).
- Needs an ear pass by Wentao: relative levels of the shotguns vs voice around 9.2-9.3, 28.3-29.1 and 49.1-49.3 where the voice resumes inside the boom tail; chip clicks under the S2 and S4 narration (gain 0.10-0.18) may need to go lower; the 2.2 s curve_rise / marker_squeak under the S2 narration is the likeliest masking risk.
- Mux step (voice + SFX + video) not done; video files untouched.
