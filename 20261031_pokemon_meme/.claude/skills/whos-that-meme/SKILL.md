---
name: whos-that-meme
description: Make a "Who's That Pokemon?" meme video from the template mp4 - cut a subject out of a photo (rembg BiRefNet), composite a silhouette over the starburst, white-flash color reveal on the answer sting, trim the template silences, and mix a narrator name line cut from a source recording (faster-whisper word timestamps). Use when asked for a who's-that-pokemon / silhouette-reveal meme, or to reuse this pipeline with a new subject, image, or narrator.
---

# Who's That Pokemon meme pipeline

Built 2026-10-31 for the CPO (co-packaged optics) edition in `PlayGround/20261031_pokemon_meme`. Reference result: `outputs/whos_that_cpo.mp4` there; full decision log in its `DevLog/DevLog-000-whos-that-cpo.md`.

## Inputs
- Template: `Who's That Pokémon？ Template V2.mp4` (scripts glob `Who*Template V2.mp4`).
- Subject photo (any background; the subject should be obvious).
- Narrator source (mp3/m4a) plus, ideally, the user's caption transcript to locate words.

## Layout
Copy `scripts/` from this skill into the new project folder. Scripts resolve `ROOT` as the parent of `scripts/`, read inputs from `ROOT`, write to `ROOT/outputs/`. Keep a `DevLog/DevLog-000-<title>.md` with plan, TODOs, timestamped progress. Use the scratchpad for preview frames and sheets; outputs/ holds only final assets.

## Template facts (V2, measured; re-verify if the template changes)
- 1280x720, 30 fps, 521 frames, 17.44 s, H.264 yuv420p, AAC 44.1 kHz stereo. No subject in it: both silhouette and reveal must be composited.
- Video: Poke Ball close-up f0-15; zoom-out f16-19; ball becomes starburst f20-22; burst zoom overshoot f22-29; steady f30+; burst center ~(365,304-312), white core ~355-400 px wide, wobbles +-8 px; "?" fades f274-283; empty burst to the end. Background streaks scroll, so no color keying - always composite over the decoded original frames.
- Audio (digital zero < -120 dBFS): lead-in 0-0.075 s; sting 1 0.075-5.872 s (question voice ~1.2-2.7 s); silence 5.872-8.573 s (2.70 s); sting 2 from 8.573 s; "It's" voice ends ~9.80 s; name slot 9.80-12.00 s (~2.2 s: deep dip with >1 kHz gone 9.80-11.12 s, reduced music 11.15-11.95 s, full music back 12.00 s); silence 14.370-17.438 s (3.07 s).
- Silence length depends on threshold (-60 dBFS / -40 dBFS give longer gaps); report the threshold.

## Workflow
1. **Inspect** (only if the template is new): ffprobe; 0.5 s contact sheet and per-frame sheets via `ffmpeg -vf "select=...,scale,tile"`; `showspectrumpic` + `silencedetect`; 100 ms RMS and band energy with numpy over raw PCM. This ffmpeg build has NO `drawtext` - render any text with Pillow.
2. **Cutout** (`scripts/cutout.py`): set `CROP_BOX` (source px around the subject with margin) and the source filename. Uses `rembg` model `birefnet-general` (~1 GB first download to `~/.rembg`). `isnet-general-use` was tried first and left blobby edges on dark parts against a dark background; BiRefNet gives straight, clean edges (it may keep a thin contact-shadow strip, invisible at video scale). Inspect the mask over magenta plus a zoomed edge crop before moving on. Writes `cpo_cutout.png` (RGBA, cropped to alpha bbox) and `cpo_silhouette.png` (same alpha, fill #141834 dark navy). Rename for the new subject.
3. **Composite** (`scripts/make_video.py`, constants at top): `WIDTH` 600 px for a wide 3.5:1 subject (scale so it roughly fills the burst; wider subjects overflow into the rays, which is fine), `CENTER` (365,312). Silhouette appears at f20 with the burst; zoom follows the measured burst overshoot table `ZOOM` damped by `ZOOM_DAMP`=0.4 (full follow pushed a wide subject off-frame and over the "?"). Reveal at `REVEAL_F`=259 (template sting-2 onset 8.64 s): CPO blended toward white 1.0/0.6/0.25 for 3 frames, then color. Run `--preview f1,f2,...` first, tile the PNGs into a sheet, look at it, then encode.
4. **Silence trims** (user asked for 0.5 s each): cut on frame boundaries with identical audio times so A/V stays synced. `GAP_CUT`=(179,245) drops 2.200 s inside the 5.872-8.573 s silence; the start frame was picked by minimizing mean |frame(A-1) - frame(A+66)| over the allowed window (A in 177..191; all ~2x a normal frame step, f179 best). `END_F`=446 leaves 0.497 s tail. Audio: `atrim`+`concat`, AAC 192k re-encode; AAC noise shortens measured silence by ~12 ms. Reveal moves to output f193 (6.433 s). Name slot moves to 7.60-9.80 s.
5. **Narrator clips** (`scripts/transcribe.py <audio> <start> <dur> ...`): do NOT transcribe the full file with large-v3 on CPU (too slow; killed). Locate candidates from the user's caption transcript, then run faster-whisper `small.en` with word timestamps on 5-7 s windows. Refine ends with a 20 ms envelope split into >4 kHz vs 100-1500 Hz: a final "s" is quiet in RMS but shows as a high-band burst (e.g. "optics" needed +0.1 s to keep the s before "concept"). Cut with `-ss/-to`, 15 ms fade-in / 20 ms fade-out, 48 kHz wav to `outputs/narrator_clips/<word>_<XmYY>.wav`. Verify each clip by re-transcribing it (pad 0.5 s silence before/after). Offer every occurrence with length and context; let the user pick. If the name is much shorter than the ~2.2 s slot, prepend a word (here "Marvell" + "co-package optics" = ~1.6 s).
6. **Mix**: `NARRATOR` list concatenated with `NAME_GAP`=0.10 s, delayed to `NAME_T`=7.60 s (right after "It's"), `amix normalize=0` at unity gain. Check levels first with `astats`: clips at -15..-20 dBFS RMS vs template voice -19 dBFS and slot bed -30..-40 dBFS needed no gain change; final peak -1.6 dBFS.
7. **Encode**: H.264 `-crf 18 -preset slow -pix_fmt yuv420p -r 30 -movflags +faststart`, AAC. Verify: ffprobe frame count/duration; frame sheet pulled from the encoded mp4 (appear, zoom peak, steady, pre-reveal, flash, color, end); re-transcribe the output name window (expect "It's <name>"); regenerate the output spectrogram.

## Tooling
- Python via `uv run --with ... python -I`; numpy + Pillow for compositing (frames piped rgb24 from/to ffmpeg); `uvx ruff check --no-cache` (avoid leaving `.ruff_cache` in the folder).
- Other processes may drop files into the project folder (user's narrator WIP); never touch files you did not create.

## Possible extensions (not built)
- Name text on screen where the "?" was (Pillow-rendered PNG overlay, f284+ in template timing).
- Soft crossfade over the gap-cut jump (silent region, video only).
