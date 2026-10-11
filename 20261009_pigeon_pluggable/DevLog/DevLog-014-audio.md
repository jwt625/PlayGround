# DevLog-014: Audio (VO, SFX, music bed, mix, mux) - workstream A

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Film v1 with sound delivered: `outputs/film_v1_20261010_av.mp4` (2026-10-10 10:45) |
| Depends on | DevLog-010 (P5, P6, interfaces), DevLog-002 rev 2 narration, DevLog-000 Section 3 (parody names), DevLog-001 Section 5 (pronunciation), DevLog-006 rev 3 timing |
| Owns | `scripts/audio/` (except `scene_events.json`, written by S), `outputs/audio/`, this file |
| Reference | Sibling film `../20261002_interconnect_film/scripts/audio/` (make_vo.py, make_sfx.py; read-only; helpers adapted) and its DevLog-004 section 5 / DevLog-005 |

## 1. Plan

1. VO: Kokoro-82M (kokoro-onnx, Apache-2.0), voice `am_liam`, one read speed for all lines; `scripts/audio/narration.json` is the single source and master clock (P6).
2. Whisper round trip on every line and on the full track (pronunciation + caption alignment).
3. SFX: procedural numpy synthesis only (no samples), cue renderer reads S's `scripts/audio/scene_events.json`; provisional cue list from DevLog-006 rev 3 until S publishes.
4. Music bed: procedural (numpy), low under the VO.
5. Mix: sidechain ducking of SFX + music under the VO, -16 LUFS integrated, true peak <= -1.5 dBTP, 48 kHz; mux script for S's `outputs/film_v1_*.mp4`.

## 2. Model files (outside the repo)

`$KOKORO_DIR` = local model directory (scripts default to `~/.cache/kokoro-onnx`); not in the repo. Downloaded once each on 2026-10-10 (curl, `--limit-rate 20M`, sequential):

| File | URL | Bytes | SHA-256 |
|---|---|---|---|
| kokoro-v1.0.onnx | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx | 325,532,387 | 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 |
| voices-v1.0.bin | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin | 28,214,398 | bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d |

## 3. Voice

`am_liam`, speed 1.10 for every line (disclaimer 1.45). Why: the sibling film started on `am_michael` and switched to `am_liam` after Wentao's "more passionate, constantly going" feedback; liam measured the fastest of six Kokoro male voices with large pitch variation (sibling DevLog-004 section 5.1, pyin: median F0 124 Hz, 4.6 semitone std). Same voice keeps the two films consistent. Kokoro has no emotion control.

Processing (sibling chain): high-pass 90 Hz, +2.5 dB at 3 kHz, compressor 3.5:1, then gain to -16 LUFS, 4x-oversampled limiter 0.6 dB under the TP target, linear loudnorm trim (`audio_util.master`). Plain two-pass linear loudnorm fell back to dynamic mode (peak-to-loudness ratio about 18 dB), hence the limiter stage.

## 4. Scripts and outputs

Run from the project root; all scripts write `outputs/logs/audio_<name>_<YYYYmmdd_HHMMSS>.log`.

| Step | Command | Output |
|---|---|---|
| VO | `KOKORO_DIR=... uv run --no-project --with kokoro-onnx --with soundfile --with numpy python scripts/audio/make_vo.py [--dry-run]` | `outputs/audio/vo_<date>.wav` (48 kHz mono), `outputs/audio/vo_lines/<id>.wav`, t0/t1/dur written into `narration.json` |
| Whisper check | `uv run --no-project python scripts/audio/check_vo.py [--model small.en\|large-v3-turbo] [--vo FILE]` | log only |
| Provisional cues | `uv run --no-project python scripts/audio/make_provisional_events.py` | `scripts/audio/sfx_events_provisional.json` (anchored to narration line ids) |
| SFX | `uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx.py [--events F] [--supplement] [--list-kinds]` | `outputs/audio/sfx_<date>.wav` (48 kHz stereo) |
| Music | `uv run --no-project --with numpy --with soundfile python scripts/audio/make_music.py` | `outputs/audio/music_<date>.wav` |
| Mix | `uv run --no-project python scripts/audio/mix.py [--no-music]` | `outputs/audio/mix_v1_<date>.wav` + `.json` (stem gains, loudness) |
| Check image | `uv run --no-project --with numpy --with soundfile --with matplotlib python scripts/audio/plot_audio.py` | `outputs/audio/mix_v1_<date>_check.png` |
| Mux | `uv run --no-project python scripts/audio/mux.py [--video outputs/film_v1_<date>.mp4]` | `outputs/film_v1_<date>_av.mp4` (video copied, AAC 192 kb/s 48 kHz, +faststart) |

Rebuild order after S changes timing or events: make_vo (only if narration text/anchors change) -> make_provisional_events -> make_sfx -> mix -> plot_audio -> mux.

## 5. Interfaces

### 5.1 narration.json (A writes; S reads)

Per line: `id`, `scene`, `anchor` (earliest scene-relative start, hand-edited), `say` (TTS text), `show` (caption), `t0`/`t1` (film seconds of the spoken line, measured), `dur` (s); optional `tail_fix` (shorten a final plosive release), `fast` + `window` (S7). `make_vo.py` places lines at `max(scene start + anchor, previous end + 0.22 s breath)`, pulls a scene's lines earlier if the last one would end later than 0.12 s before the scene end, and never time-stretches (except the S7 disclaimer).

Rev 2 timing (film v1; superseded by rev 3 in section 11; kept in narration_rev2.json), 2026-10-10 08:59:

| id | t0-t1 | dur | say | show |
|---|---|---|---|---|
| s1a | 0.25-2.04 | 1.79 | Birds evolved from dinosaurs. | same |
| s1b | 2.26-3.62 | 1.36 | Or so they told you. | same |
| s1c | 3.84-7.24 | 3.40 | Meet the O-S-F-P: Overhead Surveillance Feathered Platform. | Meet the OSFP: ... |
| s1d | 8.10-9.35 | 1.25 | It is watching you. | same |
| s2a | 10.25-11.28 | 1.03 | Feathers: heatsink. | same |
| s2b | 11.80-13.62 | 1.82 | Beak: a latch that bites the cage. | same |
| s2c | 14.60-16.14 | 1.53 | Right foot and eye transmit. | same |
| s2d | 16.36-17.90 | 1.54 | Left foot and eye receive. | same |
| s2e | 18.12-19.43 | 1.31 | Ground lands first. | same |
| s3a | 20.25-21.38 | 1.12 | Right eye: transmit. | same |
| s3b | 22.00-24.31 | 2.31 | Two D-F-B lasers glow deep in its skull. | Two DFB lasers ... |
| s3c | 24.60-26.43 | 1.83 | Eight Mach Zehnder modulators. | Eight Mach-Zehnder modulators. |
| s3d | 27.00-27.92 | 0.92 | Eight lenses. | same |
| s3e | 28.30-29.19 | 0.89 | Eight beams. | same |
| s4a | 30.25-31.56 | 1.31 | Left eye: receive. | same |
| s4b | 32.00-33.88 | 1.89 | Eight lenses, eight photodiodes. | same |
| s4c | 34.30-38.26 | 3.96 | A T-I-A, a transimpedance amplifier, turns tiny currents into voltage. | A TIA, ... |
| s4d | 38.48-39.41 | 0.94 | It sees you. | same |
| s5a | 41.30-43.91 | 2.61 | The brain: a D. S. P. from Narrow-com or Mer-vall. | The brain: a DSP from NARROWCOM or MERVALL. |
| s5b | 44.20-47.98 | 3.78 | A microcontroller reports temperature, laser current and light levels. | same |
| s5c | 48.20-49.56 | 1.36 | Hatched by Out no light. | Hatched by OUTNOLIGHT. |
| s6a | 50.25-51.68 | 1.43 | They follow you outside. | same |
| s6b | 52.60-54.17 | 1.57 | Eye to eye, bird to bird. | same |
| s6c | 55.20-56.40 | 1.20 | Up to the satellites. | same |
| s6d | 56.90-58.62 | 1.72 | Down to every camera on the street. | same |
| s7a | 60.10-61.57 | 1.47 | Parodee. Typical build. Not real birds. (fast, speed 1.45) | Parody. Typical build. Not real birds. |

43.9 s of speech in 60 s (73 percent); read speed identical across scenes; air after the last line of each scene: S1 0.65 s, S2 0.57, S3 0.81, S4 0.59, S5 0.44, S6 1.38.

### 5.2 scene_events.json (S writes; A reads)

```json
{"events": [{"t": 7.0, "name": "hero_lands_pad", "kind": "landing_thump", "gain": 1.0, "pan": 0.0, "dur": 0.4, "pos": [0, 7, 2.2], "params": {}}]}
```

`t` film seconds of the hit (or start for continuous kinds); `kind` from 5.3 (canonical name or accepted alias); optional `gain` (linear multiplier on the kind default, default 1.0), `pan` (-1 left .. 1 right, default 0), `dur` (s, continuous kinds), `pos` (accepted, unused: the renderer has no camera), `params` (kind-specific: `color` red/cyan for `fill_whoosh`, `rate` for `flight`, `f` for blips/ticks, `fade` for beds, `count` for `head_servo_crowd`). A bare list is also accepted. Unknown kinds are logged and skipped. Events of one kind closer than the kind's min gap are thinned (first kept). Name rules: `contact_mate` with `gnd`/`pwr`/`sig` in the name -> `contact_gnd`/`contact_pwr`/`contact_sig`; `beak_latch` with `release` in the name -> `beak_open`; `fill_whoosh`/`macro_whoosh` with `rx` in the name -> cyan.

Supplement (opt-in since 2026-10-10 10:45, `--supplement`): adds A's provisional cues whose kind S's file does not emit at all (`flight`/`takeoff` skipped when S emits `wing_flap`). It was on by default for S's first event file (08:43); S's film v1 file (10:40) emits every kind it wants and dropped `leash_pull` on purpose (P1: no pull shown), so the default is now off. One-shot kinds are capped in length (`MAX_DUR`: disclaimer_whoosh 0.6 s, whoosh 1.2, whip 0.6, fill_whoosh 1.4, door 1.0, hop 0.5) so an emitter's scene-length `dur` (S's 2 s `card`) does not stretch them.

### 5.3 Supported SFX kinds

All synthesised in numpy (no samples). Default dur (s), default peak gain (linear), min gap (s).

| Kind | Aliases | Dur | Gain | Gap | Sound |
|---|---|---|---|---|---|
| `wing_flap` | `flap` | 0.18 | 0.30 | 0.08 | one wingbeat: air push + feather flutter |
| `flight` | | 1.5 | 0.26 | 0 | wingbeats at `rate` Hz (default 7) for `dur` |
| `takeoff` | | 0.9 | 0.40 | 0.2 | wing claps then flurry |
| `hop` | | 0.4 | 0.35 | 0.15 | one flap + light landing |
| `landing_thump` | `landing` | 0.35 | 0.45 | 0.1 | body thump + talon taps + feather settle |
| `head_servo` | `servo`, `head_turn` | 0.22 | 0.07 | 0.12 | small servo whir, random pitch |
| `head_servo_crowd` | `head_turn_crowd` | 1.5 | 0.12 | 0.3 | 6-24 servos spread over `dur` |
| `elec_glow` | `electrical_glow`, `elec_signal` | 1.5 | 0.18 | 0 | gold electrical signal glow: fast random ticks + crackle + faint buzz (`params.rate` ticks/s, default 90) |
| `pic_lightup` | `pic_light_up`, `light_up` | 2.0 | 0.20 | 0.3 | PIC light-up front: rising sweep, partials enter as the tree splits |
| `dfb_on` | `dfb_switch_on` | 0.5 | 0.25 | 0.1 | DFB switch-on (laser_on sound) |
| `wire_bonder` | `wirebond`, `wire_bond` | 0.25 | 0.22 | 0.06 | per bond: ultrasonic chirp (stylised 7-9 kHz) + capillary tap; `params.count`, `params.step` (s) for a run in one event (set `dur` >= count x step) |
| `align_stage` | `active_alignment` | 1.0 | 0.14 | 0.2 | active alignment micro servo: settling high whir steps |
| `uv_cure` | `uv_flash` | 0.6 | 0.22 | 0.3 | UV cure flash: zap + exposure buzz |
| `lid_press` | `lid` | 0.4 | 0.38 | 0.2 | lid press: hiss then metal thunk |
| `mimo_reconfig` | `mimo`, `sat_reconfig` | 0.35 | 0.12 | 0.15 | satellite MIMO reconfigure: soft 3-4 note chord blip (`params.f` root, `params.notes`) |
| `beam_sweep` | `sweep` | 1.2 | 0.18 | 0 | one sweeping beam (whoosh layer + faint tonal edge); one event per beam, layer freely; `pan` -> `params.pan_end` moves it across the stereo field; `params.tone` varies the edge |
| `blink` | | 0.18 | 0.30 | 0.1 | eyelid shutter ticks (TX disable) |
| `eye_glow_on` / `eye_glow_off` | `glow_on` | 0.6 / 0.45 | 0.20 / 0.18 | 0.1 | rising / falling power tone |
| `pad_charge_hum` | `pad_charge` | 1.2 | 0.16 | 0 | hum + rising charge whine |
| `cell_dock` / `cell_undock` | `dock` / `undock` | 0.45 / 0.4 | 0.42 / 0.40 | 0.1 | slide + latch / release + slide |
| `beak_latch` / `beak_open` | `latch` | 0.15 / 0.25 | 0.45 / 0.35 | 0.08 | metal snap / release twang |
| `contact_gnd` / `contact_pwr` / `contact_sig` | `contact_mate` (+name rule) | 0.12-0.14 | 0.32-0.35 | 0.05 | ticks at 1.25 / 1.75 (+zap) / 2.5 kHz (triple) |
| `leash_pull` | | 0.45 | 0.30 | 0.1 | strap zip + creak |
| `whoosh` / `whip` | | 0.7 / 0.35 | 0.28 / 0.40 | 0.15 | camera move / fast whip pan |
| `fill_whoosh` | `fill`, `macro_whoosh` | 1.0 | 0.40 | 0.3 | swell + tonal shimmer (red lower, cyan higher) |
| `elec_hum` / `pic_hum` / `tia_hum` | `rx_hum` | 2.5 | 0.12 | 0 | buzz + coil whine + hiss bed (TIA variant higher) |
| `laser_on` | | 0.5 | 0.25 | 0.1 | zap + rising whine |
| `pam4_chatter` | | 1.5 | 0.10 | 0 | 4-level stepped tones (stylised PAM4) |
| `pd_flash` | | 0.08 | 0.14 | 0.05 | short high blip (`params.f`) |
| `crystal_tick` | | 0.05 | 0.10 | 0.1 | tiny glassy tick |
| `beam_launch` | | 0.7 | 0.26 | 0.2 | 8-partial shing + air |
| `i2c_stream` | `data_stream` | 1.5 | 0.10 | 0 | clocked telemetry chirps |
| `hud_lock` / `hud_tick` | | 0.6 / 0.06 | 0.20 / 0.15 | 0.2 / 0.05 | target-lock beeps / UI tick |
| `explode` / `snap_back` | `reassemble` | 0.9 / 0.6 | 0.35 / 0.38 | 0.3 | parts apart / reverse swell + clunk |
| `robot_servo` | `hatchery_robot` | 0.6 | 0.18 | 0.1 | industrial arm servo |
| `conveyor` | | 1.5 | 0.14 | 0 | belt rumble + roller clacks |
| `assembly_clunk` | | 0.3 | 0.30 | 0.1 | pneumatic hiss + clunk |
| `street_amb` | `ambience_street` | 10 | 0.16 | 0 | traffic rumble, light rain, city hum (stereo) |
| `hall_amb` | `ambience_hall` | 10 | 0.10 | 0 | fan roar, blade tones, mains hum (stereo) |
| `hologram` | `hologram_hum` | 4 | 0.10 | 0 | glassy rising shimmer |
| `footstep` | | 0.15 | 0.22 | 0.2 | shoe on concrete |
| `door` | | 0.9 | 0.30 | 0.3 | push-bar clank, swoosh, close |
| `flap_vent` | `flap_swing` | 0.7 | 0.30 | 0.2 | rubber bird-flap slaps + creak |
| `sat_ping` | `ping` | 1.3 | 0.24 | 0.3 | sonar ping + echoes |
| `cctv_whir` | `cctv` | 0.8 | 0.16 | 0.2 | PTZ pan motor + focus tick |
| `link_blip` / `link_drop` | `link` | 0.11 | 0.14 / 0.10 | 0.09 | ascending / descending double blip |
| `disclaimer_whoosh` | `card` | 0.5 | 0.38 | 0.3 | fast whoosh + tape-stop |

## 6. Whisper round trip (pronunciation)

Per-line takes, `small.en` and `large-v3-turbo` (local Whisper CLI, temperature 0). Spelling iterations (all tested through Whisper, scratch takes deleted):

| Term | First `say` | Problem heard | Final `say` / fix | small.en | large-v3-turbo |
|---|---|---|---|---|---|
| OSFP | O-S-F-P | - | O-S-F-P | "OSFP" | "OSFP" |
| heatsink | Feathers: heatsink. | "Feather is a heat sinker" (Kokoro adds a schwa after a final k) | same text + `tail_fix` (cut 139 ms release) | "Feathers heatsink" | "Feathers Heatsink" |
| transmit | Right eye: transmit. | "transmits" (long final t aspiration) | `tail_fix` 0.015 s | "transmit" | "Transmit" (s3a); s2c still "transmits" |
| DFB | D-F-B | - | D-F-B | "DFB" | "DFB" |
| Mach-Zehnder | Mach-Zehnder (`mɑːkzeɪndɚ`, glued) | "Moxander" | "Mach Zehnder" (space): `mˈɑːk zˈeɪndɚ` | "Mach-Zehnder" | "Mach Xander" (same sounds, different spelling) |
| TIA | T-I-A | - | T-I-A | "TIA" | "TIA" |
| DSP | D-S-P | turbo "DSB" | "D. S. P." | "DSP" | "DSP" |
| Narrow-com | Narrow-com | - | Narrow-com (`nˈæɹoʊkˈɑːm`) | "Narrowcom" | "Narocom" (homophone) |
| Mer-vall | Mer-vall | - | Mer-vall (`mˈɜːvˈɔːl`) | "Mervol" | "Merval" (both = intended "mer-vawl", not "Marvell") |
| Out-no-light | Out-no-light | "out no lights"; glued "Outnolight" -> "Altenolite" | "Out no light" + `tail_fix` | "out no light" | "OutNoLight" |
| Parody (S7) | Parody. | "Parity" (fast read flaps the d) | "Parodee." | "Parody" | "Parody" |

Result: small.en 0 key-term misses on 26 lines; large-v3-turbo 2 spelling-only misses (Mach Xander, Narocom) plus "transmits" on s2c. Long-form transcription of the full VO and of the final mix (small.en, word timestamps): all 132 words fall inside their caption windows (0 outside); the fast S7 line is heard as "Harrody Tickle Build, not real birds" in long-form passes (intended ad-fast gag, as in the sibling film; per-line pass gets it right).

## 7. Mix

- Stems (ebur128): VO -16.0 LUFS (TP -2.0 dBTP, LRA 2.6); SFX -24.7 LUFS raw; music -18.2 LUFS raw. Gains by measured loudness: SFX 7 LU under the VO (+1.7 dB), music 13 LU under (-10.8 dB), before ducking.
- Ducking: `sidechaincompress` keyed by the VO: SFX threshold 0.04, ratio 2.5, attack 15 ms, release 250 ms; music threshold 0.02, ratio 6, attack 40 ms, release 500 ms.
- Master: -16.0 LUFS target, 4x-oversampled limiter at -2.1 dBFS, linear trim. Result `outputs/audio/mix_v1_20261010.wav`: **-15.9 LUFS integrated, TP -4.0 dBTP, LRA 2.2 LU, 48 kHz stereo, 62.000 s**.
- Music bed (`make_music.py`): procedural only (no third-party audio; no licence needed beyond this repo): D minor, two chords per scene on the 5 s grid, detuned additive pad with slow filter sweep, 8th-note pluck arpeggio at 100 BPM, sub pulse on beats 1 and 3; cuts at 60.0 s (0.25 s tail), silent under S7.
- Check image `outputs/audio/mix_v1_20261010_check.png`: stem envelopes at mix gain with the caption windows shaded, short-term loudness (about -16 LUFS through the film; dip to -19 LUFS at 40-41.6 s where S5 has no VO), spectrogram. Looked at: VO sits 10-15 dB above the beds in speech; beds about -30 to -35 dBFS RMS in the gaps; no clipping; music gone after 60.25 s.
- Mux test: `mux.py` on `outputs/crude_v0_2_3_20261010.mp4` (to a scratch file, deleted): h264 copied, AAC 48 kHz stereo 198 kb/s, 62.000 s both streams, -16.0 LUFS after AAC, TP -4.1 dBTP, moov before mdat.

## 8. TODO

- [x] Download model files to `$KOKORO_DIR`, record SHA-256
- [x] narration.json from DevLog-002 rev 2 + make_vo.py (layout, measured timing)
- [x] Whisper check (per line + full track + final mix), spellings iterated
- [x] SFX synth library + cue renderer (`make_sfx.py`), provisional cues, S's kinds mapped
- [x] Procedural music bed
- [x] Mix (sidechain ducking, -16 LUFS, TP <= -1.5 dBTP) + check image
- [x] Mux script (tested on crude v0.2.3)
- [x] Switch to S's scene_events.json (in use since 08:53)
- [x] Mux S's film v1: `outputs/film_v1_20261010_av.mp4`
- [x] Narration rev 3 voiced and Whisper-checked (section 11)
- [x] Narration rev 4 voiced and Whisper-checked; v3 SFX kinds added (section 12)
- [ ] SFX, mix, mux for film v3 (after S publishes the v3 video and events)
- [ ] Listening pass by Wentao (no human listening done here; checks are Whisper, loudness and plots)

## 9. Open issues

- Link texture: S's film v1 file has 376 link_blip + 366 link_drop; thinned to 243 + 206 (min gap 0.09 s per kind), peak gain 0.028 / 0.015 linear: a faint continuous chirp bed under S1-S6, not individual readable events.
- Event vs VO conflicts noticed in S's film v1 events (10:40); none blocks, S/coordinator decide:
  1. S4 `hud_lock` at 36.60 lands 1.9 s before "It sees you" (s4d 38.48), during the TIA line; moving it to about 38.4 would land the lock on the line.
  2. S5 has no stamp ticks for NARROWCOM / MERVALL (spoken at about 42.6 / 43.4); A's provisional had `hud_tick` there.
  3. S6 `door` (subject exits) at 51.40 comes after "They follow you outside" (s6a 50.25-51.68); the line describes the exit before it happens on screen.
  4. S2 `hop` back in at 17.20 then `contact_gnd` at 18.10, 0.02 s before "Ground lands first" (s2e 18.12): good sync (no change).
  5. S4 PD flashes 32.60-34.56 run 0.7 s past "Eight lenses, eight photodiodes" (ends 33.88) into the TIA line; acceptable.
- large-v3-turbo still hears s2c as "transmits"; small.en hears "transmit".
- Kokoro delivery is neutral (no emotion control); if more drama is wanted, options are a cloud TTS with style control or Wentao's own read.

## 11. Narration rev 3 (2026-10-10, DevLog-015 F3, writer DevLog-016)

Source `scripts/audio/narration_rev3_draft.json` with the coordinator's picks: s4d "Target acquired. Hi, Gary.", s6d "Smile. The cameras are in on it.", s3a "Inside, real quick." (isolator question not used); all other lines as drafted. Rev 2 kept as `scripts/audio/narration_rev2.json`; `scripts/audio/narration.json` is now rev 3. Same voice and processing (am_liam, speed 1.10, breath 0.22 s, no time-stretch except S7). Every line fits its window at anchor without shortening (the writer's fallback of shortening s3b, then s5b, was not needed). Output `outputs/audio/vo_20261010_rev3.wav` (-16.0 LUFS, TP -2.0 dBTP, LRA 2.7 LU); `make_vo.py --tag rev3` names it; `mix.py --vo` selects it. 45.0 s of speech in 60 s (75 percent). Air after each scene's last line: S1 0.50 s, S2 0.85, S3 0.39, S4 1.04, S5 0.44, S6 1.02.

| id | t0-t1 | dur | say | show |
|---|---|---|---|---|
| s1a | 0.25-2.04 | 1.79 | Birds evolved from dinosaurs. | Birds evolved from dinosaurs. |
| s1b | 2.26-3.62 | 1.36 | Or so they told you. | Or so they told you. |
| s1c | 3.84-7.24 | 3.40 | Meet the O-S-F-P: Overhead Surveillance Feathered Platform. | Meet the OSFP: Overhead Surveillance Feathered Platform. |
| s1d | 7.90-9.50 | 1.60 | Fancy name for a pigeon. | Fancy name for a pigeon. |
| s2a | 10.25-11.31 | 1.06 | Feathers? Heatsink. | Feathers? Heatsink. |
| s2b | 11.80-13.62 | 1.82 | Beak: a latch that bites the cage. | Beak: a latch that bites the cage. |
| s2c | 14.60-15.98 | 1.38 | Right side transmits. | Right side transmits. |
| s2d | 16.30-17.62 | 1.32 | Left side receives. | Left side receives. |
| s2e | 17.84-19.15 | 1.31 | Ground lands first. | Ground lands first. |
| s3a | 20.25-21.63 | 1.38 | Inside, real quick. | Inside, real quick. |
| s3b | 22.00-25.00 | 3.00 | Two D-F-B lasers: steady light, zero information. | Two DFB lasers: steady light, zero information. |
| s3c | 25.22-27.67 | 2.45 | Eight Mach Zender modulators: fix that. | Eight Mach-Zehnder modulators fix that. |
| s3d | 27.89-29.61 | 1.72 | Four levels, two bits a symbol. | Four levels, two bits a symbol. |
| s4a | 30.25-31.02 | 0.77 | Incoming... | Incoming. |
| s4b | 32.00-33.96 | 1.96 | Photodiodes: light in, current out. | Photodiodes: light in, current out. |
| s4c | 34.20-36.74 | 2.54 | A T-I-A, turns tiny current into voltage. | A TIA turns tiny current into voltage. |
| s4d | 37.40-38.96 | 1.56 | Target acquired. Hi, Gary. | Target acquired. Hi, Gary. |
| s5a | 41.30-43.91 | 2.61 | The brain: a D. S. P. from Narrow-com or Mer-vall. | The brain: a DSP from NARROWCOM or MERVALL. |
| s5b | 44.20-47.88 | 3.68 | A microcontroller snitches: temperature, laser current, light levels. | A microcontroller snitches: temperature, laser current, light levels. |
| s5c | 48.20-49.56 | 1.36 | Hatched by Out no light. | Hatched by OUTNOLIGHT. |
| s6a | 50.25-51.68 | 1.43 | They follow you outside. | They follow you outside. |
| s6b | 52.40-54.38 | 1.98 | Bird to bird. No fiber required. | Bird to bird. No fiber required. |
| s6c | 54.80-56.65 | 1.85 | Satellite handoff. Again. Again. | Satellite handoff. Again. Again. |
| s6d | 57.30-58.98 | 1.69 | Smile. The cameras are in on it. | Smile. The cameras are in on it. |
| s7a | 60.10-61.57 | 1.47 | Parodee. Typical build. Not real birds. | Parody. Typical build. Not real birds. |

Beat word onsets (large-v3-turbo word timestamps on the takes, film s, about +-0.1 s): s2c "Right" 14.60 (red pulse), s2d "Left" 16.30 (cyan pulse 16.3-16.4); s2e Ground 17.84 / lands 18.18 / first 18.50; s4d Target 37.40 / acquired 37.68 / Hi 38.08 / Gary 38.40 (hud_lock on 37.40-37.7); s5a DSP 41.88 / Narrow-com 42.60 / Mer-vall 43.38 (stamps); s6c Satellite 54.80 / handoff 55.22 / again 55.68 / again 56.24 (three satellite handoffs); s6d Smile 57.30 / cameras 57.86; s6a starts 50.25 (door exit at or before).

Whisper round trip, rev 3 (key-term checks now keyed on the caption text, so new terms are checked wherever they appear):

| Line | First take heard as | Fix | small.en | large-v3-turbo |
|---|---|---|---|---|
| s2a Feathers? Heatsink. | - (tail_fix carried) | - | "Feathers heatsink" | "Feathers Heatsink" |
| s3c Mach Zehnder ... fix that | "Mach-Zander ... fixed that" (both models) | say "Eight Mach Zender modulators: fix that." (common English "Mock Zender", DevLog-001 fallback; colon keeps "fix" apart from "that") | "mock Zender ... fix that" | "Mach Zender ... fix that" |
| s3d two bits a symbol | small.en "cymbal" (homophone) | - | "cymbal" | "symbol" |
| s4a Incoming. | "In comings" / "In Cummings" (both) | say "Incoming..." | "incoming" | "Incoming" |
| s4c A T-I-A turns ... | small.en "A TI turns" | say "A T-I-A, turns ..." (caption unchanged) | "A TIA turns" | "A TIA turns" |
| s4d Target acquired. Hi, Gary. | small.en "high, Gary" (homophone) | - | "high, Gary" | "Hi Gary" |
| s5b ... snitches: ... | - | - | "snitches" | "snitches" |
| s6c Satellite handoff. Again. Again. | - (not flat to Whisper: two separate "again" with pauses; the "Again, and again." fallback was not needed) | - | "handoff again, again" | "handoff again, again" |
| s5a parody names | - | - | "Narrowcom or Mervol" | "Narocom or Merval" (homophone) |
| s7a | - | - | "Parody" | "Parody" |

Final: small.en 1 miss (s4d "high, Gary", homophone), large-v3-turbo 1 miss (s5a "Narocom", homophone); full-track word timestamps: all words inside caption windows (one zero-length Whisper word at 37.68 flagged by the overlap test, an artefact). Not a listening test: delivery (for example whether "Again. Again." sounds flat) still needs Wentao's ear.

Pending: S re-aligns beats and captions to these times; when S publishes the v2 video and `scene_events.json`, A re-runs make_sfx, `mix.py --vo outputs/audio/vo_20261010_rev3.wav`, plot_audio, and `mux.py --video outputs/film_v2_<date>.mp4 --out outputs/film_v2_<date>_av.mp4`.

## 12. Narration rev 4 and v3 SFX kinds (2026-10-10, DevLog-018 V1/V2, writer DevLog-016 section 12)

Source `scripts/audio/narration_rev4_draft.json`, coordinator picks: main draft lines (s5b "Photodiodes by INCOHERENT, TIAs by SEMTREK or MACOMBO.", s6d "One flock, one mind: the Internet of Birds."). s5b fits without alt A (3.81 s, ends 48.01; s5c pushed 0.03 s to 48.23). Rev 3 kept as `scripts/audio/narration_rev3.json`; `scripts/audio/narration.json` is now rev 4. Same voice and processing. Output `outputs/audio/vo_20261010_rev4.wav` (-15.9 LUFS, TP -2.0 dBTP, LRA 1.8 LU). 46.2 s of speech in 60 s (77 percent). Air after each scene's last line: S1 0.49 s, S2 0.85, S3 0.61, S4 1.04, S5 0.55, S6 0.53.

| id | t0-t1 | dur | say | show |
|---|---|---|---|---|
| s1a | 0.25-2.04 | 1.79 | Birds evolved from dinosaurs. | Birds evolved from dinosaurs. |
| s1b | 2.26-3.62 | 1.36 | Or so they told you. | Or so they told you. |
| s1c | 3.84-7.24 | 3.40 | Meet the O-S-F-P: Overhead Surveillance Feathered Platform. | Meet the OSFP: Overhead Surveillance Feathered Platform. |
| s1d | 7.46-9.51 | 2.05 | Workhorse of modern data centers. | Workhorse of modern data centers. |
| s2a | 10.25-11.80 | 1.55 | Feathers? Integrated heat sink. | Feathers? Integrated heat sink. |
| s2b | 12.02-13.84 | 1.82 | Beak: a latch that bites the cage. | Beak: a latch that bites the cage. |
| s2c | 14.60-15.98 | 1.38 | Right side transmits. | Right side transmits. |
| s2d | 16.30-17.62 | 1.32 | Left side receives. | Left side receives. |
| s2e | 17.84-19.15 | 1.31 | Ground lands first. | Ground lands first. |
| s3a | 20.25-22.38 | 2.13 | Two D-F-B lasers, by Dim-en-tum. (phonemes override) | Two DFB lasers, by DIMENTUM. |
| s3b | 22.70-24.44 | 1.74 | Steady light, zero information. | Steady light, zero information. |
| s3c | 25.00-27.45 | 2.45 | Eight Mach Zender modulators: fix that. | Eight Mach-Zehnder modulators fix that. |
| s3d | 27.67-29.39 | 1.72 | Four levels, two bits a symbol. | Four levels, two bits a symbol. |
| s4a | 30.25-31.02 | 0.77 | Incoming... | Incoming. |
| s4b | 32.00-33.96 | 1.96 | Photodiodes: light in, current out. | Photodiodes: light in, current out. |
| s4c | 34.20-36.74 | 2.54 | A T-I-A, turns tiny current into voltage. | A TIA turns tiny current into voltage. |
| s4d | 37.40-38.96 | 1.56 | Target acquired. Hi, Gary. | Target acquired. Hi, Gary. |
| s5a | 41.30-43.91 | 2.61 | The brain: a D. S. P. from Narrow-com or Mer-vall. | The brain: a DSP from NARROWCOM or MERVALL. |
| s5b | 44.20-48.01 | 3.81 | Photodiodes by In-co-here-ent, T-I-A's by Sem-trek or May-combo. | Photodiodes by INCOHERENT, TIAs by SEMTREK or MACOMBO. |
| s5c | 48.23-49.45 | 1.22 | Hatched by Oo-no-light. | Hatched by OUNOLIGHT. |
| s6a | 50.25-51.71 | 1.46 | They observe you outside. | They observe you outside. |
| s6b | 52.40-54.38 | 1.98 | Bird to bird. No fiber required. | Bird to bird. No fiber required. |
| s6c | 54.80-56.65 | 1.85 | Satellite handoff. Again. Again. | Satellite handoff. Again. Again. |
| s6d | 57.10-59.47 | 2.37 | One flock, one mind: the Internet of Birds. | One flock, one mind: the Internet of Birds. |
| s7a | 60.10-61.57 | 1.47 | Parodee. Typical build. Not real birds. | Parody. Typical build. Not real birds. |

Word anchors (large-v3-turbo word timestamps on the takes, film s, about +-0.1 s):
- s1d Workhorse 7.46 / data centers 8.60-8.88; s2a Integrated 10.59 / heat sink 11.23-11.41
- s2b (beak close-up) Beak 12.02 / latch 12.50 / bites 12.90 / cage 13.32, line ends 13.84 (target window was 12.3-14.2; now 0.3 s earlier because "Integrated heat sink" measured 1.55 s, not the estimated 1.86 s)
- s3a Two 20.25 / DFB 20.45 / lasers 21.01 / DIMENTUM 21.67, ends 22.38 (DFB switch-on on "lasers" about 21.0; the coordinator's 20.9 matches)
- s3b Steady 22.70 / light 23.00 / zero 23.24 / information 23.60, ends 24.44 (target 22.7-24.7)
- s3c Mach Zender 25.18 / modulators 25.84 / fix that 26.54-27.06
- s5a DSP 41.88 / Narrow-com 42.60 / Mer-vall 43.38
- s5b Photodiodes 44.20 / INCOHERENT 45.10 / TIAs 46.12 / SEMTREK 46.70 / MACOMBO 47.36, ends 48.01 (target 44.2-47.6; 0.4 s longer than estimated)
- s5c Hatched 48.23 / OUNOLIGHT 48.75; s6a They observe 50.25-50.41 (door exit at or before 50.25)
- s6c handoff 55.22 / again 55.68 / again 56.24
- s6d One flock 57.10-57.30 / one mind 57.74-57.88 / the Internet of Birds 58.42-58.94, ends 59.47 (target 57.1-59.3; hero TX blink after 59.5)

Whisper round trip, rev 4 (small.en / large-v3-turbo):

| Term | First take | Fix | small.en | large-v3-turbo |
|---|---|---|---|---|
| Integrated heat sink (tail_fix) | - | - | "integrated heat sink" | "Integrated Heat Sink" |
| DIMENTUM | "Dementum" (both; also with Dimm-en-tum, Dim-entum, Dimentum) | phonemes override `baɪ dˈɪm ɛntəm` (stress on "Dim", no glide into "en") | "dimentum" | "Dim Entem" |
| INCOHERENT (In-co-here-ent) | - | - | "Incoherent" | "Incoherent" |
| TIAs | "TIS" / "TI is" | say "T-I-A's" | "TIAs" | "TIAs" |
| SEMTREK (Sem-trek) | - | - | "Semtrec" (homophone) | "Semtrek" |
| MACOMBO (May-combo) | - | - | "Maycombo" | "Macombo" |
| OUNOLIGHT (Oo-no-light) | "Unolite" (both; the fallbacks Ooh-no-light and Ou-no-light give the same phonemes `ˈuːnˈoʊlˈaɪt` and the same "Unolite"; "Ooh no light" -> turbo "UNO lights") | kept primary "Oo-no-light" + tail_fix: "Unolite" is read "oo-no-lite" (as in "Uno"), i.e. the intended sound | "Unolite" | "Unolite" |
| One flock, one mind: the Internet of Birds | - | - | exact | exact |
| They observe you outside | - | - | exact | exact |

Remaining misses: small.en s4d "high, Gary" (homophone); turbo s5a "Narocom" (homophone). Full-track word timestamps: all words inside caption windows except two zero-length Whisper words at segment joins (Ground 18.12, A 34.32; overlap-test artefacts). Not a listening test.

v3 SFX kinds (for S to emit; full rows in 5.3): `elec_glow`, `pic_lightup`, `dfb_on`, `wire_bonder`, `align_stage`, `uv_cure`, `lid_press`, `mimo_reconfig`, `beam_sweep`. Each rendered once in a test cue list (no NaN; per-kind RMS -46 to -23 dBFS before mix; a pan -0.8 -> 0.8 sweep starts 3.6 dB left); `elec_glow` default gain raised 0.10 -> 0.18 after the test (its spiky ticks read 10 dB under the other kinds). Recommendations for S: one `elec_glow` event per glowing stretch with `dur` (not per spark); one `beam_sweep` per sweeping beam with `pan`/`params.pan_end` from its screen motion; one `mimo_reconfig` per link re-configuration; `wire_bonder` either one event per bond or one event with `params.count`.

Pending: S's v3 video and events -> make_sfx, `mix.py --vo outputs/audio/vo_20261010_rev4.wav`, plot_audio, `mux.py --video outputs/film_v3_<date>.mp4 --out outputs/film_v3_<date>_av.mp4` (when the coordinator says so).

## 10. Progress log

- 2026-10-10 08:32: model files downloaded (section 2).
- 2026-10-10 08:35: VO v1: 27 lines, all inside their scenes with natural pacing, no time-stretch; linear loudnorm fell back to dynamic -> oversampled limiter stage added.
- 2026-10-10 08:37: Whisper small.en: 3 key-term misses (Mach-Zehnder, Mer-vall regex, Out-no-light), plus "heat sinker", "Parity".
- 2026-10-10 08:38-08:47: three rounds of spelling variants through small.en / base.en / large-v3-turbo; `tail_fix` added for over-long final plosive releases; fixes in section 6.
- 2026-10-10 08:48: VO final (`outputs/audio/vo_20261010.wav`, -16.0 LUFS, TP -2.0 dBTP); small.en 0 misses, turbo 2 spelling-only.
- 2026-10-10 08:52: SFX renderer + provisional cues; S's `scene_events.json` (421 events, written 08:43) already present and picked up; S's kind names mapped as aliases + name rules; A's cues of kinds S does not emit added as a supplement (30 events). SFX: 328 events rendered, peak -9.0 dBFS.
- 2026-10-10 08:54: procedural music bed; mix v1 -15.9 LUFS, TP -4.0 dBTP; check image looked at; Whisper on the final mix: all words inside caption windows.
- 2026-10-10 08:56: mux script tested on crude v0.2.3 (scratch output, deleted).
- 2026-10-10 08:59: S5 was tight (0.12 s air before the S6 cut): anchors s5a 1.6 -> 1.3, s5b 4.6 -> 4.2; S5 lines now 41.30 / 44.20 / 48.20 (air 0.44 s). VO, SFX, mix, check image rebuilt: mix -15.9 LUFS, TP -4.0 dBTP; Whisper small.en on the final mix: 0 key-term misses, 131 words, 0 outside caption windows. S must re-read narration.json for S5 captions/beats.
- 2026-10-10 10:45: S's film v1 (`outputs/film_v1_20261010.mp4`, 540x674, 30 fps, 62.000 s) and rewritten `scene_events.json` (995 events, canonical kinds, 935 with camera-space pan, written 10:40). make_sfx: 654 events after thinning (341 thinned, mostly link blips/drops); provisional cues dropped out except `leash_pull` (S removed it on purpose) -> supplement made opt-in, re-rendered with 0 provisional events; S's 2 s `card` dur capped to 0.6 s. Pan verified in the SFX stem: beak_bites_lip (pan -0.82) L/R +8.9 dB, beams_cross_aisle (-0.58) +1.8 dB, lands_hand (0) -0.1 dB. Mix: -15.9 LUFS, TP -3.6 dBTP, LRA 2.1 LU. Mux: `outputs/film_v1_20261010_av.mp4` (h264 copied 540x674 30 fps, AAC 48 kHz stereo 197 kb/s, 62.000 s both streams, -16.0 LUFS / TP -3.7 dBTP after AAC, moov before mdat). Whisper small.en on the muxed film: 0 key-term misses, 132 words, 0 outside caption windows. Captions in the video checked on 8 frames at line midpoints (5.5, 18.8, 25.5, 36.3, 42.6, 48.9, 57.8, 60.8 s): each shows the line being spoken. Check image looked at: VO 10-15 dB over beds, hall bed gap at 28.3-28.6 s (S's split), short-term about -16 LUFS, dip to -18.5 at 40-41.3 s (no VO).
- 2026-10-10 11:47: narration rev 3 voiced (section 11): `outputs/audio/vo_20261010_rev3.wav`; rev 2 json kept as `narration_rev2.json`; three spelling iterations (Incoming..., T-I-A comma, Mach Zender + colon); timings sent to the coordinator for S. scene_events.json was rewritten by S at 11:39; no v2 video yet.
- 2026-10-10 15:35: narration rev 4 voiced (section 12): `outputs/audio/vo_20261010_rev4.wav`; rev 3 json kept as `narration_rev3.json`; fixes: DIMENTUM phoneme override, "T-I-A's"; OUNOLIGHT kept as "Oo-no-light" (Whisper "Unolite"). Nine v3 SFX kinds added to make_sfx.py and published in 5.3.
- 2026-10-10 16:41: v3 audio started in parallel with S's v3 render: `scripts/audio/scene_events.json` (written 16:39, md5 2248fee004e32d33ef313c990c4c161d) snapshotted to `outputs/audio/scene_events_film_v3_164119.json`; make_sfx on the snapshot: 341 events after thinning (93 thinned), all v3 kinds rendered (elec_glow 2, pic_lightup 1, dfb_on 1, wire_bonder 1, align_stage 1, uv_cure 1, lid_press 1, mimo_reconfig 6 of 16, beam_sweep 12), 0 unknown kinds, peak -5.4 dBFS; `mix.py --vo outputs/audio/vo_20261010_rev4.wav --out outputs/audio/mix_v3_20261010.wav`: -15.9 LUFS, TP -3.8 dBTP, LRA 1.9 LU; `plot_audio.py` gained `--mix/--vo`; check image `outputs/audio/mix_v3_20261010_check.png` looked at (VO over beds throughout; SFX nearly silent 0-4.4 s during the intro cards, street bed starts 4.4 s; short-term dips to -19 LUFS at 34 s and -21 LUFS at 41.5 s where VO pauses). Event vs VO (rev 4) conflicts in section 13. Mux waits for `outputs/film_v3_<date>.mp4`.

## 13. Film v3 event vs VO (rev 4) check (events 16:39)

| # | Event (film s) | VO (rev 4) | Status |
|---|---|---|---|
| 1 | `door` subject_exits_door 44.23 (S5, during the vendor line s5b) | s6a "They observe you outside" 50.25; target: exit at or before 50.25 in S6 | CONFLICT: the exit sound lands in the exploded-view scene, 6 s early; no door event in S6 (only camera_through_door whoosh 52.20). Likely a stale time in the build |
| 2 | contacts gnd 18.43 / pwr 18.68 / sig 18.93 | s2e Ground 17.84 / lands 18.18 / first 18.50 | GND clicks on "first", 0.6 s after "Ground"; moving the three to about 18.15 / 18.45 / 18.75 would land GND on "lands" |
| 3 | sat_ping handoffs 55.23, 55.47, 55.67, 55.97, 56.23 | handoff 55.22 / again 55.68 / again 56.24 | 55.23, 55.67, 56.23 sit on the words; the extra pings at 55.47 and 55.97 blur the "Again. Again." beats (keep or drop: S's call) |
| 4 | snap_back 47.60, flashback_in whoosh 47.80, wire_bonder 47.91 | s5b ends 48.01 (MACOMBO 47.36); s5c Hatched 48.23 | hatchery starts 0.2-0.4 s before the vendor line ends; acceptable |
| 5 | mimo_reconfig: 6 events at the same instant 55.07, 16 in 55.07-56.40 | s6c | thinned to 6 by the 0.15 s min gap; fine as a texture |
| 6 | blink 59.40 | s6d ends 59.47 (target: blink after the line) | 0.07 s early; negligible |
| ok | beak_bite 12.90 = "bites" 12.90; tx/rx glow 14.60 / 16.30; dfb_on 21.05 on "lasers" 21.01; elec_glow 24.95 with s3c 25.00 and 34.00 with s4c 34.20; pd_flash 32.90-33.32 in s4b; hud_lock 37.55 on "acquired" 37.68; stamps 42.60 / 43.40 = Narrow-com 42.60 / Mer-vall 43.38 | | aligned |

Coordinator decisions (2026-10-10 16:42) and how they are applied: (1) drop `subject_exits_door` 44.23 (off screen; the 52.20 camera-through-door whoosh stays); (2) keep the contact clicks 18.43 / 18.68 / 18.93 (synced to the rendered pad glow pulses); (3) drop the satellite pings at 55.47 and 55.97; (4) minor items unchanged. S5 frames were already rendering, so the fix is audio-side: `scripts/audio/event_overrides_v3.json` (drop/shift lists matched by exact event name and time +-tol) applied by `make_sfx.py --overrides FILE` after reading the events and before per-kind thinning.

Found while applying (3): the first v3 render (16:41) had thinned the sat_ping events with the 0.3 s min gap, which kept the uplink 54.95 and the two "extra" pings 55.47 / 55.97 and dropped the three on the words. Fixes: overrides run before thinning; sat_ping min gap 0.3 -> 0.2 s. Rendered pings now: uplink 54.95, handoffs 55.23 / 55.67 / 56.23 (on handoff 55.22 / again 55.68 / again 56.24).

- 2026-10-10 16:44: v3 SFX re-rendered with the overrides (events snapshot 16:39, md5 2248fee004e32d33ef313c990c4c161d unchanged): 434 -> 431 events after overrides, 341 after thinning, 0 door events, 4 sat_pings; mix `outputs/audio/mix_v3_20261010.wav` -16.0 LUFS, TP -4.1 dBTP, LRA 1.9 LU; check image re-plotted and looked at. Mux waits for the coordinator's go (rule: if S's final events change only in the hatchery window 47.9-50.0, re-run SFX + mix with the same overrides, then mux).

## 14. v4 wave (2026-10-10, DevLog-020 W2-W7): VO rev 5, 63.0 s timeline, post captions, Plague Blossom bed, M's assembly SFX

### 14.1 VO rev 5 (W3, W5)

- `scripts/audio/narration.json` is rev 5; rev 4 kept as `narration_rev4.json`. New header keys `total_s` (63.0) and `scene_bounds` ([0,10] ... [40,51] [51,61] [61,63]: S5 is 11 s because the assembly slot grew +1.0 s); `audio_util.TOTAL` / `SCENE_BOUNDS` now come from narration.json, so make_vo, make_sfx, mix and plots all run at 63.0 s.
- s5b "Photodiodes by May-combo, T-I-A's by Sem-trek." / "Photodiodes by MACOMBO, TIAs by SEMTREK." (2.91 s, 44.20-47.11); s6c "Satellite handoff. Again. And again." (1.96 s); S6 lines pre-placed +1.0 s (s6a 51.25, s6b 53.40, s6c 55.80, s6d 58.10-60.47); s5c stays 48.20 (inside the slot). To be re-synced to S's final table.
- Output `outputs/audio/vo_20261010_rev5.wav` (-15.9 LUFS, TP -2.0 dBTP). Whisper: s5b small.en "Maycombo ... Semtrek", turbo "Maycombo ... Semtrak"; s6c both "handoff again and again". Other lines unchanged from rev 4.
- Disclaimer s7a (W5 text, say = show except "Parodee"): 30 words. Measured: natural fast read (Kokoro speed 1.5) 6.21 s. Squeezed with pitch-kept atempo (sibling-film method) and Whisper-checked:

| Read length | Squeeze | small.en | large-v3-turbo |
|---|---|---|---|
| 6.2 s (none) | x1.0 | near-verbatim ("Brody, Gary's fictional ...") | near-verbatim |
| 3.5 s | x1.78 | mostly ("Brody, Gary's sexual, not real birds. Typical show vendors ...") | near-verbatim except the first word ("R.D. Gary's fictional not real birds. Typical build shown ... Void where fiber is cheaper.") |
| 3.0 s | x2.07 | partial | mostly |
| 2.5 s | x2.49 | poor | partial |
| 2.0 s (card as is) | x3.11-3.22 | nothing usable ("Okay.") | nothing usable ("I'll see you next time.") |

  The first word is the weakest at any squeeze ("Brody", "R.D.", "Erty"); reading "Parodee." as a separate unsqueezed take helps that word but Whisper then stops after it. Current rev 5 VO has the 2.0 s placeholder (61.00-62.93, x3.22): NOT intelligible. Decision needed (coordinator / owner): card >= 3.5 s (film 64.5 s), or a shorter text (for example without "Beams invisible, rendered for viewing", which is already the on-screen beam note).

### 14.2 Post captions (W2): `scripts/post/burn_captions.py`

- Local ffmpeg 8.1.2 has no `drawtext` / `subtitles` / `ass` filters (checked `ffmpeg -filters`; `overlay` present), so captions are Pillow PNGs + overlay. One narration line at a time ("show" text) from narration.json t0/t1: on at t0 - 0.05, off at min(t1 + 0.35, next t0 - 0.13), 0.12 s alpha fades, overlap asserted impossible; lines wider than 88 percent of 1080 wrap into two balanced rows; s7a (fast disclaimer, text on the card) skipped by default (`--skip`).
- Glyphs (after the owner's 2026-10-10 complaint about broken black edges): Arial Bold 54 px at 1080 wide, white fill, solid black stroke max(6, 0.09 x font px) = 6 px all round, drawn at 2x and downsampled (premultiplied Lanczos, straight alpha out); soft shadow (offset 3,4 px, blur 4 px, 55 percent) drawn UNDER the stroke; strip = full width x bottom 18 percent band (243 px at 1350), text block asserted to fit inside with stroke + shadow padding (no clipping). Compositing: `[0:v]format=gbrp`, `overlay=format=gbrp` (no chroma subsampling of the caption or its alpha during the composite), `format=yuv420p` only at the end; H.264 CRF 18 preset medium, same fps, `+faststart`; optional `--audio` muxes AAC 192 kb/s 48 kHz in the same pass.
- Test on `outputs/film_v3_20261010.mp4` (pipeline test only; v3 still has its old Blender captions underneath, which show up as grey ghost letters with offset edges in the crops): 24 captions, 14 s encode. Crops checked over the white intro cards and yellow tags (1.0 s), the cyan fill (30.7 s), the pale assembly shot (48.9 / 49.5 s), a dark street (54.0 s) and a two-row caption over the TX macro (26.0 s): the black outline is complete and even on every glyph; the beam note (bottom left) stays clear of two-row captions. The old-caption ghosts are the "partial, offset black edge" the owner saw in the v3 render.
- Open: during the intro (0-3.6 s) captions sit over the photo-card labels / yellow form-factor tags in the bottom band; the intro layout should keep the bottom 18 percent clear (tree agent / S).

### 14.3 Plague Blossom bed (W6) - NOT CLEARED FOR PUBLIC POSTING

- `scripts/audio/make_music_bed.py --src <mp3 outside the repo>` -> `outputs/audio/music_plague_20261010.wav`. Track 390.1 s, about 161.5 BPM (librosa). Start 0.000 s (first beat 0.116 s): the track's opening (0-50 s) is a sparse pulse with hits about every 3 s and builds into the full theme at about 50-52 s, which lands near the S6 street finale (51.0 s); a continuous 63.0 s excerpt, no loop or edit. High-pass 40 Hz, voice pocket -5 dB at 1.6 kHz and -4 dB at 3.2 kHz (1 octave), fade-in 0.3 s, fade-out 60.8-63.0 s (under the disclaimer). Bed -23.8 LUFS before gain.
- In the mix (`mix.py --music ... --music-lu 14`): 14 LU under the VO stem before ducking (gain -6.1 dB), then the existing VO sidechain (ratio 6, threshold 0.02, attack 40 ms, release 500 ms). On the check plot the intro bed sits about 25 dB under the VO and the finale build 10-15 dB under it before ducking. Third-party music: personal / preview use only (owner, W6).

### 14.4 SFX for M's assembly and the scooter zoom (W4, W7)

- New kinds: `coupler_in`, `coupler_out` (glide + seat / release click), `scooter_zoom` (motor whine rising into the cut + rush), `pass_chime` (two-note chime), `place_click` (precise part placement), `pneumatic` (press stroke; `dir` up/down, name rule "up"). New aliases: `bond`, `lid_close`, `eye_on`, `eye_off`, `coupler_insert`, `coupler_remove`, `zoom_transition`, `pass`, `test_pass`, `part_place`, `press`, `press_down`, `press_up`, `eye_diagram_on`/`off` (-> hud_tick). All rendered once in a test list (no NaN; RMS -40 to -22 dBFS before mix).
- `scripts/audio/make_provisional_events_v4.py` -> `outputs/audio/scene_events_v4_provisional.json` (449 events): v3 snapshot + v3 overrides, v3 assembly events removed (8), v3 events at or after 49.70 shifted +1.0 s (137), M's events from `assets/components/assembly/assembly_hatchery.json` at film t = 47.9 + build_frame / 1.5652 / 30 (slot 92 film frames = 144 build frames): isolator 48.18, lens 2 48.37, tool leaves 48.52, DFB on 48.54 (soft), UV 48.62, press down 49.48, lid closed 49.73 (clunk, gain 1.2), press up 49.82, couplers in 49.99, eyes on 50.12, eye diagram on 50.18, PASS 50.46, eye diagram off 50.54, couplers out 50.56, hop 50.73; wire bonds every 3rd of the 26 (10 events, gain 0.6, a texture); scooter zoom 59.95-61.05 peaking on the 61.0 cut.

### 14.5 Pre-mix at 63.0 s

`make_sfx.py --events outputs/audio/scene_events_v4_provisional.json --out outputs/audio/sfx_v4_20261010.wav` (360 events after thinning); `mix.py --vo outputs/audio/vo_20261010_rev5.wav --sfx outputs/audio/sfx_v4_20261010.wav --music outputs/audio/music_plague_20261010.wav --music-lu 14 --out outputs/audio/mix_v4_20261010.wav`: -15.9 LUFS, TP -3.8 dBTP, LRA 2.1 LU, 63.000 s; check image `outputs/audio/mix_v4_20261010_check.png` looked at (VO on top throughout; bed fades out 61-63 s; short-term about -16 LUFS with the usual dips in VO pauses).

### 14.6 Final steps when S delivers `outputs/film_v4_20261010.mp4` + scene_events.json

1. Re-sync narration rev 5 to S's timing table (anchors / scene_bounds), re-voice if anything moved (about 1 min), settle the disclaimer length (14.1).
2. Snapshot `scripts/audio/scene_events.json` to `outputs/audio/scene_events_film_v4_<time>.json`; `make_sfx.py --events <snapshot> [--overrides ...] --out outputs/audio/sfx_v4_20261010.wav`.
3. `mix.py --vo outputs/audio/vo_20261010_rev5.wav --sfx outputs/audio/sfx_v4_20261010.wav --music outputs/audio/music_plague_20261010.wav --music-lu 14 --out outputs/audio/mix_v4_20261010.wav`; plot_audio.
4. `uv run --no-project --with pillow python scripts/post/burn_captions.py --video outputs/film_v4_20261010.mp4 --audio outputs/audio/mix_v4_20261010.wav --out outputs/film_v4_20261010_av.mp4` (captions + H.264 yuv420p + AAC 192k + faststart in one pass).
5. Verify: ebur128 on the output, Whisper on the final mix (check_vo --vo), caption frames at several times (light and dark backgrounds).

- 2026-10-10 17:45: v4 wave prepared (section 14): VO rev 5, 63.0 s timeline in the scripts, caption tool rebuilt after the owner's stroke complaint and tested on v3 (crops checked), Plague Blossom bed, M's assembly SFX + scooter zoom, pre-mix `mix_v4_20261010.wav`. Waiting on: disclaimer length decision, S's v4 picture + events.

### 14.7 Disclaimer decision and 64.0 s pre-mix (coordinator, 2026-10-10)

- Card 3.0 s from 61.0 s, film 64.0 s (`total_s` 64.0, S7 window [61, 64]); shorter text without the beam line (on screen already). show: "Parody. Gary is fictional. Not real birds. Vendors and specs vary by module, standard and mood. Void where fiber is cheaper." (21 words).
- First word: variants fitted to 2.8 s and checked with small.en / large-v3-turbo: "Parody." -> "Harry" / "Erty"; "Parodee." -> "Brody"; "Pair-uh-dee." -> "Ready?"; "It's a parody." -> correct in both; "This is a parody." -> correct in both. say = "This is a parody. Gary is fictional. ..." (say only; show keeps "Parody.").
- Read: Kokoro speed 1.5, natural 4.60 s, atempo x1.66 (pitch kept) into 61.05-63.83 (2.78 s). large-v3-turbo on the take: "This is a parody Gary's fictional Not Real Birds. Vendors and specs vary by module standard and mood. Voidware fiber is cheaper." (all words; "Voidware" = "void where"); in the full mix it is heard at 60.84-63.80 with the same words.
- Music bed re-cut to 64.0 s with the fade-out 60.9-64.0 s (under the disclaimer); provisional events and SFX re-rendered at 64.0 s; pre-mix `outputs/audio/mix_v4_20261010.wav`: -16.0 LUFS, TP -3.8 dBTP, LRA 2.1 LU, 64.000 s; turbo on the pre-mix: 141 words, 0 outside caption windows; check image looked at.
- burn_captions.py skips s7a by default (card text in render).
- 2026-10-10 19:47: disclaimer rev (3.0 s card, 64.0 s film) voiced and checked; pre-mix at 64.0 s. Waiting on S's film_v4 picture + final scene_events.json.

## 15. Film v4 final audio, captions, mux (2026-10-10, v4d events)

- Re-sync: S's v4 timeline (DevLog-011 10.1: S5 40-51 s, S6 51-61 s, card 61-64 s, assembly slot 47.85 s + 92 frames) matches narration rev 5 `scene_bounds` / `total_s`; no line moved, no re-voice. Word anchors vs S's labels and events: MACOMBO spoken 45.04 (label from 44.2), SEMTREK 46.50 (label from 45.4) - labels lead the words by 0.8-1.1 s; satellite pings 56.23 / 56.67 / 57.23 on handoff 56.24 / again 56.70 / again 57.32.
- Events: `scripts/audio/scene_events.json` (v4d, 429 events, 64.0 s, md5 ed5d9e54f2374b53943f295f38fddfb7) snapshotted to `outputs/audio/scene_events_film_v4d_200201.json`; all kinds known.
- Overrides `scripts/audio/event_overrides_v4.json` (make_sfx.py now also supports `rekind` and `add`): drop subject_exits_door 44.23 and the extra pings 56.47 / 56.93 (v3 decisions carried over); rekind press_down -> pneumatic, heads_in/out -> coupler_in/out, the single wire-bond event -> a 6-bond run 0.3 s apart; add press up 49.75, eye diagram on 50.13 / off 50.49, PASS chime 50.406 (build frame 120), scooter zoom 60.35 (0.7 s, peak on the 61.0 cut; S emits no event for it). 429 -> 431 events, 342 after thinning; SFX peak -5.2 dBFS.
- Mix `outputs/audio/mix_v4_20261010.wav` (VO rev 5, SFX v4, Plague Blossom bed at 14 LU under the VO, ducked): -16.0 LUFS, TP -3.4 dBTP, LRA 2.2 LU, 64.000 s; check image looked at.
- v4e: S rewrote `scene_events.json` during the v4e re-render (425 events; only crowd link blips/drops changed vs v4d) -> snapshot `outputs/audio/scene_events_film_v4e_204707.json`; SFX + mix re-run with the same overrides: mix -16.0 LUFS, TP -3.4 dBTP.
- Callouts (owner rejected the in-render 3D labels): `scripts/post/callout_overlay.py`, hero-loop style from `scripts/assets/anim_label_overlay.py` (box (8,12,18) at 85 percent, 1 px outline (150,210,230), text (235,240,245) Helvetica Neue 27 px, box 39 px, 2 px leader from a 4 px dot through a knee; columns 14 px from the edges; column and in-column order fixed from each label's first frame; heights follow anchors with an 8 px gap, rate-limited to 6 px per frame; boxes above 0.82 H). Tested on a fake JSON over S5 frames and compared 1:1 with the hero loop at 7.0 s (same box, outline, leader, font; 1.29x scale). Run on S's `outputs/film_v4_callouts.json` (233 frames: 553-616 S2 cutaway, 1258-1426 S5; 11 labels, L 5 / R 6): 0 boxes outside; one fading anchor (DSP, frame 1425, a = 0.17) is 7 px above the frame edge.
- Picture: intro `outputs/film_v4_intro_frames` 1-160 cross-dissolved (xfade fade 0.8 s at 4.5 s) into the main pass 136-1920 (symlink list: labelled frame where it exists, else the clean frame) -> `outputs/film_v4_20261010.mp4` (H.264 yuv420p CRF 18, 30 fps, 1920 frames, 64.000 s, +faststart; log `outputs/logs/assemble_film_v4_*.log`).
- Final: `burn_captions.py --video outputs/film_v4_20261010.mp4 --audio outputs/audio/mix_v4_20261010.wav --out outputs/film_v4_20261010_av.mp4`: 24 captions (s7a skipped); H.264 yuv420p 1080x1350 30 fps 1920 frames 64.000 s, AAC 48 kHz stereo 199 kb/s 64.000 s, moov before mdat; ebur128 on the file: -16.0 LUFS, TP -3.5 dBTP, LRA 2.2 LU. large-v3-turbo on the final film audio: 141 words, 0 outside caption windows; transcript complete incl. the disclaimer ("This is a parody, Gary's fictional, not real birds. Vendors and specs vary by module standard and mood. Voidware fiber is cheaper."); per-line misses are the known homophones (Narocom, high Gary, Semtrak).
- Crops looked at (scratchpad `crops_v4/`): captions at 1.0 / 1.6 s (intro), 48.6 / 49.0 s (assembly), 54.0 s (street), 5.5 / 59.5 s (two rows); callouts at S5 f1280 / f1340 / f1410 and S2 f0570 / f0600. Outline complete and even everywhere; callout boxes stay above the caption band. Findings for S: (1) at 1.6 s the intro photo cards (THEROPOD / ARCHAEOPTERYX) sit inside the caption band and the caption crosses the photos (at 1.0 s the cards are above it); (2) a blurred teal rectangle at the top left of every S5 and S2-cutaway frame (looks like a defocused in-render label plane left after removing the 3D labels); (3) the assembly beam note at 48.6 s renders doubled (offset ghost copy) in S's frame; (4) the v4 assembly is dark at 48-50 s, the only white frames are the flash into S6 (about 50.9 s, no caption there), so the light-background check used the intro white cards.
- Cleanup: the symlink list and `outputs/film_v4_frames_lab/` removed after the final mp4 was verified; clean frames left in place (coordinator clears them after the owner review).
- 2026-10-10 20:55: film v4 delivered: `outputs/film_v4_20261010_av.mp4`.

## 16. Headers and beam note in post (2026-10-10, coordinator: DOF blurred the in-render headers into a teal smudge; the beam note doubled in assembly shot C)

- `scripts/post/callout_overlay.py` now also reads S's `outputs/film_v4_texts.json` (`--texts`; `{"W","H","frames":{"<frame>":[{"kind":"header"|"beamnote","text","a"}]}}`) and writes one labelled copy per frame listed in either JSON (callouts + texts in one pass). Same palette and box style as the callouts. Header: top left, 22 px from the top and left edges, Helvetica Neue 30 px, wrapped to <= 60 percent of the width (multi-row box). Beam note: bottom left, 20 px, box top at 0.955 H (1289 px), 22 px from the left edge.
- `scripts/post/burn_captions.py`: caption blocks are now kept above 0.945 H (1275 px) instead of being centred in the band: two-row captions sit higher in the band (ink 1128-1274 px), one-row captions stay centred (ink about 1194-1269 px); a pixel check on every caption PNG asserts the ink (alpha > 8, incl. stroke and shadow) ends above 0.945 H. Gap to the beam-note box >= 14 px.
- Test on a fake texts JSON over S5 / assembly frames with the real callouts and two-row captions composited: header, wrapped long header (3 rows, translucent at a = 0.6), beam note under "Photodiodes by MACOMBO, / TIAs by SEMTREK." - no overlap, 0 boxes outside the frame (scratchpad `texts_test_bottom.png`, `texts_test_header_wrap.png`, `texts_test_full.png`).
- Next (on the coordinator's go after S's v4f frames, intro and both JSONs): callouts + texts -> labelled copies, intro xfade, silent encode, captions + mux to `outputs/film_v4_20261010_av.mp4` with the unchanged `mix_v4_20261010.wav`; verify as in section 15 plus crops of a header frame (S5), the beam note (assembly C) and the intro at 1.6 s and 3.0 s.

## 17. Film v4 final (v4f/v4i frames, M's v4.1 test stand; 2026-10-10 22:06)

- Re-runnable: `scripts/post/finalize_v4.sh [--audio]` (events snapshot -> make_sfx with `event_overrides_v4b.json` -> mix -> callout_overlay (callouts guarded to 1201-1436, owner: S5 only) + headers + beam note into a fresh labelled folder -> `scripts/post/assemble_film.py` (intro xfade 0.8 s at 4.5 s, labelled frames preferred) -> burn_captions + mux -> stream / loudness check), log `outputs/logs/finalize_v4_*.log`.
- Assembly SFX: `scripts/audio/make_overrides_v4b.py` builds the overrides from M's rebuilt `assembly_hatchery.json` events at S's slot start 47.867 s (frame 1437; S's assembly_* events dropped by window and replaced): isolator / lens place clicks, tool leaves, DFB on, UV, 10 wire bonds, press down, lid clunk 49.40, press up, hop into socket, feet seated click, heads in, eyes on, heads docked clunk, eye diagram tick, full link blip + 8 rising lane ticks (new kind `lane_ticks`), PASS chime 50.42, screens off + heads out, hop off; plus the v4 drops (door 44.23, pings 56.47 / 56.93) and the scooter zoom. make_sfx gained `drop_window`.
- Final run 22:03 on events md5 d66415290f8062f950c0e5958cb8704d (v4i, 424 events) and frames after `film_v4_frames.READY` (22:01:46): `outputs/film_v4_20261010_av.mp4` 64.000 s, H.264 yuv420p 1920 frames + AAC 48 kHz 199 kb/s, faststart, -16.0 LUFS, TP -3.5 dBTP, LRA 2.1 LU. Turbo Whisper on the previous identical-audio build: 141 words, 0 outside caption windows.
- Crops looked at (scratchpad `crops_v4i/`, `crops_v4f/`): f560 cutaway sharp, header only, no callouts; f640 TX subassembly with header + beam note; f1300 8 S5 callouts + header; f1460 / f1510 test stand sharp with screens, header + beam note clear of the caption; intro 1.0 / 1.6 / 3.0 s cards above the caption band; beam note under a two-row caption at 59.5 s, no overlap. The teal smudge and the doubled beam note are gone.
- `outputs/film_v4_frames_lab/` and temp lists removed after verification; clean frames untouched.
