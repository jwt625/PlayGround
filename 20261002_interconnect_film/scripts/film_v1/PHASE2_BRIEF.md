# Phase 2 brief (v1.2 scene pass)

Audience: the agent updating ONE scene of the film for v1.2. Date: 2026-10-03. Read `SCENE_BRIEF.md` first (framework, process rules); this file adds the v1.2 requirements.

## Read

1. `DevLog/DevLog-005-v1_2-polish-plan.md` (plan, progress log).
2. `DevLog/v1/DevLog-005-critique-v1_1.md`: your scene's table, the cross-cutting items X1-X6, the top-10 list and the SFX-sync notes. Fix every high and medium item for your scene unless it conflicts with Wentao's feedback.
3. `DevLog/DevLog-004-v1_1-feedback-wave.md` Section 1: Wentao's feedback; everything there must stay satisfied.
4. `DevLog/v1/DevLog-005-transitions.md`: scene-side tweak requests and constraints (slow camera, no cuts or flashes in the first and last 0.5 s of the scene, specific requests per boundary). The transitions bake the first and last 0.5 s of your scene into a scope screen, so those frames must be clean.
5. Characters: `DevLog/v1/DevLog-005-characters-v2.md` and `DevLog/v1/DevLog-005-npc-v2.md` (switch instructions). Motion: `DevLog/v1/DevLog-005-motion-v2.md` (usage guide, API `scripts/assets/characters/motion_v2/motion_v2.py`; libraries already baked for every v2 rig in `assets/components/characters/motion_v2/<asset_id>_motion_v2.blend` with manifests listing frame counts and impact/hit-stop frames).
6. Your previous scene devlog `DevLog/v1/DevLog-003-scene-sNN.md` and your scene script(s).

## Requirements

- Characters: switch to the v2 assets (`gary_v2` or `gary_v2_holes30`, `manager_v2`, `npc_*_v2`); fix hard-coded v1 names. Replace stiff v1 actions with motion_v2 actions (walk/stomp_walk with locked feet, gun_raise_aim_fire, shot_hit_fall, reactions, fights, work loops); use their impact frames for hit-stop, camera shake and FX timing; layer face actions or keep `p_expr_*` keys.
- Human readability: at the key human beats (Manager anger, aiming, the shot, Gary's reaction and fall, fights, the whip) frame people at about 35-50 percent of frame height (4:5 portrait), faces readable; Gary visible at every shot with the hole readable; the shooter and target both readable or cut as a clear two-shot. No bodies floating without a floor in frame.
- Subtitles: replace your `asm.narr([...])` call with `asm.narr_vo(N)` (N = scene number). Text and timing come from `scripts/audio/narration.json`, the same source as the voice track. Do not edit narration.json; if a line fights the picture, note it in your devlog.
- Keep visual event times where the SFX are cued (`scripts/audio/sfx_cues.json`, film time = 10*(N-1) + scene time): shotgun bangs, whip crack/hit, splats, impacts. If you must move an event, list old and new film times in your devlog under a heading "SFX retime".
- HUD: the draft HUD (FX notes via `asm.fxn`, timecode via `asm.timecode`) is gated by the env var `FILM_HUD`; build with `FILM_HUD=0`. If your script writes FX notes or timecodes with `L.ovt("FX"...)`/`L.ovt("TC"...)` directly, switch to `asm.fxn`/`asm.timecode`. Clean renders (Wentao, 2026-10-03): with `FILM_HUD=0` the FX notes, timecodes and source footnote cards (`asm.card`) are NOT rendered; `asm.finalize` writes them to `scenes/v1/sNN_<name>.blend.overlays.json` and `scripts/film_v1/post_overlays.py` burns them in during post for an annotated cut. Always use `asm.card`/`asm.fxn`/`asm.timecode` (never `L.ovt` for these kinds) so nothing is lost. Subtitles, BIG words and LAB labels are still rendered. Captions now use a bold outlined font (blender_lib OVL), check they read on your backgrounds and do not cover the action.
- Look: tame blown-out ceiling lamps and speckle (lower emission, larger soft area lights, no dithered textures), lower bloom triggers (emission strength), keep a consistent warm clay look; avoid photosensitive flicker (no luma swings > 30 percent faster than 3 per second).
- Motion: eased cameras, no 1-frame luma flashes at camera moves, no hard cuts that are not motivated; motion blur on (shutter 0.5) where it helps.
- Render cost: draft preset at 50 percent under about 2 s per frame.

## Process

- Back up your current scene blend and scripts to your scratchpad first. Build with `FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/sNN_<name>.py -- scenes/v1/sNN_<name>.blend`.
- Verify: 12-still contact sheet at 540x675 across the 10 s plus 6-frame strips (0.1 s spacing) over each key beat (shots, falls, fights, whip, eggs); view the PNGs; fix the biggest problems before reporting. Then render your scene at draft quality yourself with `RENDER_PRESET=draft python3 scripts/film_v1/render_all.py <your scratchpad> 50 N v1_2_draft` (writes `outputs/v1/sNN_<name>_v1_2_draft_p50.mp4`; deletes frames after encoding) and look at frames extracted from that mp4 around the key beats.
- You own only your scene's script(s), blend, devlog `DevLog/v1/DevLog-003-scene-sNN.md` (append a v1.2 section) and `assets/generated_textures/v1/sNN/`. Do not edit asm.py, blender_lib.py, render scripts, assets, narration/SFX files or other scenes; request framework changes in your devlog and final report.
- No git, no emoji, ISO dates, one Blender process at a time (quit when done; never kill processes you did not start). Disk is tight (about 7 GiB free, shared by six agents): keep your scratchpad under 400 MB, delete your own stills when done, never leave PNG sequences behind.
- Be token-economical: switch characters and motion and fix the high-severity items first (checkpoint the devlog), then medium items and polish. If you run low on budget, leave an exact done/partial/not-started list in your devlog.
- Final report (under 450 words): per critique item and per requirement what changed, measured per-frame cost, SFX retimes (if any), framework requests, remaining problems, path of your draft mp4.
