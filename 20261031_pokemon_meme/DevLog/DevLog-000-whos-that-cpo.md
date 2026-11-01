# DevLog-000: Who's That Pokemon - CPO edition

Started 2026-10-31.

## Goal
Interconnect meme on the "Who's That Pokemon?" template: the hidden Pokemon is a co-packaged optics (CPO) package, cut out of `marvell-CPO.png`. Silhouette during the question, color reveal on the answer sting. Narrator line ("co-packaged optics") is WIP and will be mixed in later.

## Template facts (inspected 2026-10-31)
- `Who's That Pokémon？ Template V2.mp4`: 1280x720, 30 fps, 521 frames, 17.44 s, H.264 yuv420p + AAC 44.1 kHz stereo.
- Video: Poke Ball close-up f0-15; zoom-out f16-19; ball becomes starburst f20-22; burst zoom-overshoot f23-25; steady state f26+ (burst at ~(365,304), white core ~400x400, wobbles about +-8 px); "?" fades f274-283; empty burst f284-520. Background streaks are animated.
- Audio: sting 1 0.10-5.9 s (question voice ~1.2-2.7 s); silence 5.9-8.5 s; sting 2 onset ~8.6 s; low-level gap 9.8-11.0 s (likely name slot, inferred from spectrogram); loud tail 12.0-13.0 s; silence 14.4-17.44 s.

## Plan
1. Segment the CPO from `marvell-CPO.png` with an established matting model (rembg / isnet), crop with margin, inspect the mask, clean up.
2. Assets: `outputs/cpo_cutout.png` (RGBA color), `outputs/cpo_silhouette.png` (same alpha, flat dark fill).
3. Composite over the template frames (Python + numpy/Pillow, frames piped to ffmpeg):
   - Silhouette appears with the burst (f22), scale follows the measured burst size through the zoom overshoot (f22-29), then holds.
   - Reveal: short white flash, then color cutout at the sting-2 onset (~8.64 s, f259).
   - Template audio passed through unchanged (narrator added later).
4. Encode H.264 yuv420p, AAC, +faststart, 30 fps -> `outputs/whos_that_cpo.mp4`. Verify by frame sheet.

## Assumptions
- Silhouette fill: flat dark navy, like the anime silhouettes.
- No name text on screen (not requested); the narrator will carry the answer.

## TODO
- [x] Cutout + silhouette assets
- [x] Composite script `scripts/make_video.py`
- [x] Render + verify frames
- [x] Mix in narrator

## Progress
- 2026-10-31: template inspected; plan written.
- 2026-10-31: segmentation. rembg isnet-general-use left blobby edges on the dark right-side connector housings; switched to rembg birefnet-general (clean straight edges, picks up a thin strip of contact shadow, invisible at video scale). `scripts/cutout.py` -> `outputs/cpo_cutout.png` (1035x299 RGBA), `outputs/cpo_silhouette.png` (fill #141834).
- 2026-10-31: composite. CPO width 600 px centered (365,312). Appears f20 with burst; zoom overshoot f20-29 from measured burst core width (steady ~355 px), damped to 40% because the CPO is wider than the burst (full follow pushed it off-frame and over the "?"). Reveal f259 (8.64 s, sting-2 onset): white blend 1.0/0.6/0.25 over f259-261, then color. Template audio stream-copied.
- 2026-10-31: rendered `outputs/whos_that_cpo.mp4`: H.264 yuv420p 1280x720 30 fps, 521 frames, 17.44 s, AAC copy, +faststart. Verified frames 21/23/25/150/258/259/261/450 from the encoded file. f23 (peak zoom) is clipped ~13 px at the left edge for one frame.
- 2026-10-31: user request: cut both silences to 0.5 s. Template digital silence (< -120 dBFS) was 2.701 s pre-reveal (5.872-8.573 s) and 3.068 s tail (14.370-17.438 s). Cut on frame boundaries: drop template f179-244 (2.200 s; f179 = least-visible jump in the silent window, frame diff ~1.9x a normal frame step) and end at f445. Audio trimmed with the same times (atrim+concat), re-encoded AAC 192k. Result: 380 frames, 12.667 s; measured silence 0.488 s pre-reveal (5.885-6.373 s) and 0.494 s tail (AAC quantization noise trims ~12 ms vs the 0.501 s / 0.497 s cut). Reveal now at output f193 (6.433 s), sting-2 onset 6.373 s. Spectrograms: `outputs/template_spectrogram.png` (original), `outputs/whos_that_cpo_spectrogram.png` (trimmed).
- 2026-10-31: name slot in template sting 2: "It's" voice ends ~9.80 s; deep dip (>1 kHz gone) 9.80-11.12 s (~1.3 s); reduced music 11.15-11.95 s; full music back 12.00 s -> usable name window ~2.2 s (output timing 7.60-9.80 s).
- 2026-10-31: narrator clips from `Inside Co-packaged Optics - Marvell - full.mp3` via `scripts/transcribe.py` (faster-whisper small.en on short windows, located with the user's caption transcript; a full-file large-v3 run was too slow and was killed). End points refined with 20 ms band-energy envelopes (final "s" fricative found >4 kHz). Clips (15/20 ms edge fades) in `outputs/narrator_clips/`, each re-transcribed to confirm content: cpo_0m47 47.64-48.60 (0.96 s), cpo_3m01 181.92-182.81 (0.89 s), marvell_1m29 89.74-90.30 (0.56 s), marvell_1m51 111.38-111.70 (0.32 s), marvell_4m38 278.40-278.82 (0.42 s). Speaker says "co-package optics" (no -d).
- 2026-10-31: user picked marvell_1m29 + cpo_0m47. Mixed at unity gain (clip RMS -15.5 / -20.2 dBFS vs template voice -19.3 dBFS; slot bed -30..-40 dBFS): "Marvell", 0.10 s gap, "co-package optics", starting 7.60 s (output), ending ~9.22 s; full music returns 9.80 s. Output peak -1.6 dBFS. Re-transcribed the output 6.9-10.3 s: "It's Marvell co-package optics" (Marvell 7.66-8.20, optics ends 9.08 per whisper). `outputs/whos_that_cpo_spectrogram.png` regenerated.
- 2026-10-31: distilled the pipeline into project skill `.claude/skills/whos-that-meme/` (SKILL.md + copies of cutout.py, make_video.py, transcribe.py).
