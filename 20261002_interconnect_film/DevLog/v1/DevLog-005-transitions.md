# DevLog-005 transitions (agent T): oscilloscope scene-to-scene transitions

Date: 2026-10-02. Plan owner: `DevLog/DevLog-005-v1_2-polish-plan.md` phase 1c. Scene timing is unchanged (S1 0-10 s ... S7 60-62 s); film length stays 62.0 s, 30 fps.

## 1. Design

### Principle
Every boundary b (10, 20, ..., 60 s) is a "scope moment" that REPLACES film frames [30b-15, 30b+15) (1.0 s, 30 frames) with a rendered clip. Clip frames 0..14 show the last 0.5 s of scene N, frames 15..29 the first 0.5 s of scene N+1, both in order and unretimed, so the story timing and every VO/SFX time are untouched.

The scene pictures are the TEXTURE of the 3D oscilloscope screen (asset `lab_office/bench_oscilloscope`, its own `MAT_lab_office_screen`, 16:10, 256 x 160 mm, emission). The camera starts exactly framing the 4:5 picture rectangle on the screen (so clip frame 0 equals scene frame 285 within 33-39 dB PSNR, see section 5), pulls back out of the screen with perspective, the trace sweep wipes the picture out and the trace "morphs" into the next scene's signature, then a second sweep wipes the next picture in while the camera pushes back into the screen until it fills the frame exactly. The clips are opaque; this is a deviation from the "overlay with alpha" idea in the task, taken because a picture inside a moving 3D screen cannot be done by an alpha overlay.

### Shared family (all six)
- Same scope, bench mat, wall, 3 area lights; camera path = log-space dolly out and back with a plateau of about 6 frames at the apex (u = 0.4..0.6), small lateral arc, elevation and Dutch roll (all zero at both clip ends), handheld shake that is zero at the ends.
- Same screen language: 16 x 10 graticule, phosphor traces with core + halo, alpha channel of the screen texture = glow boost (emission strength 1 + 3.5 alpha; the picture itself is alpha 0), compositor bloom only above 1.05 (the picture never blooms), scanlines, noise, vignette, chroma split, horizontal-sync roll with dark sync bar, horizontal tear on glitch frames, trigger flicker, motion blur. All CRT artefacts are weighted by an envelope that is exactly 0 at clip frames 0 and 29.
- Glass reflection of the screen is keyed 0 at the clip ends and 0.5 at the apex (a constant 0.5 cost about 9-12 dB of PSNR at the ends in the A/B test).
- Sweep line: a bright vertical line moving left to right; left of it is the new content. Sweep 1 wipes picture N out (trace world appears), sweep 2 wipes picture N+1 in.
- Bezel micro-gags (all keyed in `transitions.blend`): knob twists, button presses, scope jolt, probe cable whip, power LED off.

### Storyboard (u = clip frame / 29; clip frame i is at film time b - 0.5 + i/30)

| T | Boundary | Trace world (signature morph) | Sweeps (u) | Camera | Micro-gag | Pun |
|---|---|---|---|---|---|---|
| T1 | S1 to S2 (10.0) | Closed noisy eye (tex_gen.eye_frame, same simulation as the S1/S2 scope screens) opens in three retimer steps; label RETIMER 1/2/3 and the simulated BER readout changes | 0.14-0.40, 0.56-0.84 | Continues S1's closing zoom on the scope: the picture (already showing a scope) shrinks into the screen (scope inside scope), apex 0.62 m, swing right | Horizontal scale knob clicks three times with the eye steps (u about 0.32, 0.46, 0.61); screen flash per step (trigger flicker) | Eye re-opens through retimers |
| T2 | S2 to S3 (20.0) | Whiteboard curves (energy per bit and latency, tex_gen._curves shapes, CH1 yellow / CH2 cyan) collapse to a dot, then a label XY appears and the beam plots a package outline in XY mode: substrate, ASIC, 2 x 4 optical engine blocks, laser diode symbol and beam | 0.14-0.40, 0.64-0.88 | Apex 0.80 m, swing left | AUTOSET button press at the XY switch (frames 10-17) | Curve turns into the laser-and-package outline |
| T3 | S3 to S4 (30.0) | TEMP baseline rises, nine thermal spikes (fast rise, slow decay) one after another, trace turns green to red, dashed limit line is crossed; picture N gets heat shimmer before it is swept away | 0.14-0.38, 0.60-0.84 | Apex 0.70 m, strongest shake of the six | Whole scope jolts with the spikes (key frames 8-23); long glitch burst | Heat curve spiking |
| T4 | S4 to S5 (40.0) | Flat baseline then six Lorentzian ring-resonance dips appear one after another with a counter n/6; centre dip lands in a green pass window (as the S5 spectrum screen) | 0.14-0.40, 0.56-0.84 | Apex 0.78 m, largest Dutch roll (0.22 rad at the apex) | Horizontal position knob turns two full turns (frames 13-17) | Ring dips |
| T5 | S5 to S6 (50.0) | The six dips deepen and narrow into vertical lines, rotate to horizontal, turn fibre cyan and wiggle: six parallel fibres (S6's cyan fibre) | 0.12-0.36, 0.60-0.84 | Whip: extra lateral swing (+-0.35 d) through the apex, heavy sync roll twice | Probe cable whips (frames 12-24) | Dips collapse into fibre lines |
| T6 | S6 to S7 (60.0) | No trace: CRT power-off. Picture squashes to a bright horizontal line (u 0.04-0.26), line shrinks to a dot (0.28-0.40), dot fades, black phosphor with faint NO SIGNAL, then the disclaimer card is swept in | none, 0.52-0.78 | Slowest and widest (apex 0.95 m) | Power button press, power LED turns off (frame 12) | Flatline into the black card |

Notes: the eye BER numbers, dip counts and package outline are illustrative (same standing as the film's other simulated scope screens): the 2 x 4 optical engine blocks are schematic, not tied to the S3/S4 "8 + 8 OEs" caption; no measured data used.

### Cue proposal (transition SFX)
`scripts/audio/sfx_cues_transitions.json` (70 cues, schema as `sfx_cues.json`; types defined in `scripts/audio/make_sfx_transitions.py`, not in `make_sfx.py`). Per boundary: CRT degauss thump at u 0.04, two rising sweep tones, horizontal-sync roll buzz, static bursts on glitch frames, plus: T1 3 knob clicks with pitch-rising blips; T2 XY button, falling glide at the curve collapse, sample-and-hold plot tone while the outline is drawn; T3 rising heat sizzle and 9 zaps (pitch and level rise); T4 six dip pings (the pass dip higher and louder) and a knob ratchet; T5 shimmer of six detuned partials and three probe flicks; T6 power-off pitch drop (59.54), button click, FLATLINE 1 kHz beep 59.75-59.98, dot pip 59.89. The existing `ding` at 60.0 is not moved. Rendered: `outputs/audio/sfx_transitions_20261002.wav` (48 kHz stereo, 62.0 s, peak 0.18 before limiting, rms 0.006 overall vs 0.044 for sfx_20261002.wav; window rms 0.016-0.024). Not mixed into the VO mix by me.

## 2. Files

| File | Role |
|---|---|
| `scripts/film_v1/transitions.py` | texture generator (numpy), Blender build and render modes |
| `scenes/v1/transitions.blend` | scope + bench + lights + 6 cameras bound to timeline markers T1..T6 (frames 1-30, 31-60, ... 151-180) + bezel gags + glare node; the screen image is plugged in at render time, so opened standalone the screen is black |
| `scripts/film_v1/compose_film.py` | driver: textures, Blender render, clip encode, concat, audio mux |
| `scripts/audio/make_sfx_transitions.py`, `scripts/audio/sfx_cues_transitions.json`, `outputs/audio/sfx_transitions_20261002.wav` | SFX |
| `outputs/film_v1_2_transitions_test_draft_p50_20261002.mp4` | test film (no audio, 540 x 674, 1860 frames) |
| `outputs/v1/transitions/film_v1_2_transitions_test_draft_p50_20261002/t1..t6.mp4` | encoded draft clips (CRF 10, 4.9 MB total), reused by `--reuse-clips` |

## 3. Build and usage

Rebuild the blend (only needed after editing camera paths or gags): `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/transitions.py -- build scenes/v1/transitions.blend`

Draft p50 test film (took 86 s end to end for all six):
```
python3 scripts/film_v1/compose_film.py --scene-glob 'outputs/v1/s{n:02d}_*_v1_1_draft_p50.mp4' --s7-glob 'outputs/v1/s07_*_v1_0_draft_p50.mp4' --out outputs/film_v1_2_transitions_test_draft_p50_20261002.mp4
```
Final p100 with audio (pattern must select exactly one file per scene, last sorted wins; S1-S6 300 frames, S7 60 frames, all one size):
```
python3 scripts/film_v1/compose_film.py --scene-glob 'outputs/v1/s{n:02d}_*_v1_2_p100.mp4' --s7-glob 'outputs/v1/s07_*_p100.mp4' --out outputs/film_v1_2_p100_20261002.mp4 --audio outputs/audio/mix_vo_sfx_20261002.wav
```
Options: `--only 3,4` re-render only those boundaries (others must exist in `--clips-dir`), `--reuse-clips` re-concat only, `--xfade N` cross-fades N frames between scene and clip at both clip ends (default 0; tested with 2 and audio, 1860 frames, AAC 48 kHz stereo, 62 s), `--crf`, `--work`, `--keep-work`. The Blender render is told the scene width and height, so the transitions are rendered at exactly the scene resolution (not rendered at 100 percent and downscaled: the picture texture must be 1:1 with the scene pixels; draft cost is lower). Audio is muxed with `-c:a aac -b:a 192k -ar 48000`. Run uv for the texture step is automatic (`uv run --no-project --with numpy`). SFX: `uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx_transitions.py [--write-cues]`.

Environment overrides for experiments (render step): `TRANS_FRAMES=1,14,30`, `TRANS_MB`, `TRANS_DOF`, `TRANS_FILTER`, `TRANS_INTERP`, `TRANS_SPEC`; build step: `TRANS_INSET`.

## 4. Measured cost (this machine, one Blender process)
- Draft 540 x 674: Blender 52.6 s for 180 frames (0.29 s/frame including process start); first boundary 0.62 s/frame in an earlier run (shader compile), the rest 0.26-0.31 s/frame. Texture generation 31.8 s for 180 frames (0.18 s/frame). Whole draft pipeline 86 s.
- p100 1080 x 1350, boundary 1 only (the only boundary with p100 scenes available): render 0.86 s/frame (25.8 s for 30), texture generation 15.2 s for 30 frames (0.5 s/frame, includes decoding). Estimate for all six at p100: about (0.86 + 0.5) x 180 = 4 min (extrapolated, not measured for S3-S6).
- Disk: draft textures 39 MB per boundary (deleted after encode); p100 about 145 MB per boundary of PNGs; the work dir is deleted unless `--keep-work`.

## 5. Verification (2026-10-02)
- Test film: 1860 frames, 30 fps, 540 x 674, yuv420p, 62.0 s. Contact sheets (every 3rd frame over the 1 s window, all six) inspected: no black frames (mean luma minimum inside a window 17.2 at T6, the designed dark phase; 73-101 elsewhere), no jumps.
- Frame-to-frame mean abs difference at window entry/exit (film frames 30b-16 to 30b-15 and 30b+14 to 30b+15) equals the old hard-cut film's own values within about 1 (e.g. b=50 entry 14.44 vs 14.32; b=20 entry 9.45 vs 9.60).
- Fidelity of the clip ends against the scenes (PSNR of the test film vs the v1.1 film at the same index, includes one extra lossy generation): entry frame (30b-15) 33.1-38.9 dB; exit frame (30b+14) 28.4 (S7 thin text), 33.1-37.1 dB for the others. Frame 30b-14 drops to 22-26 dB because the camera has started to move, as designed.
- Checked: knob rotates about its own centre (matrix_world of the mark), scope jolt keys. Full-resolution frames of T1 viewed (sharp, bloom confined to phosphor).
- Not verified: S3-S6 at p100 (those files do not exist yet); audio sync of the SFX against the transitions by ear.

## 6. Requested scene-side tweaks (nobody edited; scene scripts untouched)
1. All scenes: in the last 0.5 s (t in [b-0.5, b)) and first 0.5 s of the next scene keep camera motion slow and avoid hard cuts or full-frame flashes; the picture is shown shrunken and moving inside the screen, so fast whips read as noise. Captions and text are fine (they shrink with the picture).
2. S1: finish the closing zoom with the scope on the screen centre and the eye visible; the pull-out then reads as scope-in-scope. No change needed if the framing stays as in v1.1 (the scope is at about 45 percent of the frame width at 9.9 s).
3. S2 opening (10.0-10.5): optionally frame the bench scope larger or more central, so the "pulled out of the scope" gag lands; no new fast motion in the first 0.5 s.
4. S5/S6: the S5 opening shuffle (motion-blur ramp) and the S5 end (Gary flying) are the busiest windows (frame difference 14 at b=50 entry). Acceptable as is; calmer is better.
5. S6 end: the SLAP hit and text should be done by about 59.6 s (the squash starts at 59.54). In v1.1 it is visible at 59.5.
6. S7: the card must be the full card from its first frame (no own fade-in), and the first 0.3 s of S7 is covered by the sweep; if the disclaimer VO starts at 60.0 the text will lag it by about 0.2-0.3 s (see open questions).
7. Optional hook, not required now: none of the scene blends has to contain a scope; the pictures are taken from the rendered mp4s.

## 7. Open questions for the coordinator / Wentao
1. Is "clip replaces frames, scene pictures baked as screen texture" acceptable instead of an alpha overlay? It forces transitions to be re-rendered whenever a scene mp4 changes (about 4 min estimated at p100).
2. T6 covers the first 0.3 s of the disclaimer card; the existing `ding` at 60.0 and the disclaimer VO start were not moved. Does the VO start at 60.0 or later? If at 60.0, sweep 2 of T6 can be moved earlier (SPEC T6 sweep2 and the matching cue u values).
3. The transitions use their own bench set (dark teal mat, wall) and do not run through the scene compositor group (`NG_comp_post`: vignette, grain, bloom); only the picture inside the screen carries the scene grade. OK, or should the group be added?
4. DOF is implemented but off (it made the screen soft); motion blur is on (shutter 0.4).
5. Eye and BER values on screen come from the film's own illustrative simulation; if any must carry source tags like the cards, say so.

## 8. Progress log
- 2026-10-02: design, textures (all six), Blender scope rig, six cameras, gags, compose script, test film built and checked; SFX proposal written and rendered.
- Done: everything in the deliverables list. Partial: p100 only measured on boundary 1; audio sync not auditioned. Not started: fresh-context audit.
