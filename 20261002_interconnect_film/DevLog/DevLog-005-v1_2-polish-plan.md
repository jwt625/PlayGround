# DevLog-005: v1.2 polish plan (characters, motion, VFX, scope transitions)

Date: 2026-10-02. Input: Wentao, "soundtrack is decent; next iteration of polishing, VFX etc.; the humans need better modeling, actions and animations; improve transitions between scenes, maybe use the scope as the transition item/theme; think, inspect existing rendering, plan and fan out".

## 1. Inspection of v1.1 draft (coordinator, 2026-10-02)

- Characters (previews `assets/components/characters/previews/gary_three_quarter.png`, `manager_front.png`): tall and thin, small head (about 1/7 of height), thin fingers, stiff A-pose, flat suit (no lapels or collar shape, sharp white shirt wedge), overalls without cloth folds, diaper-like hem seams on NPCs (known from the v1.0 audit). The reference style (Zack D. Films, clay, chunky, big expressive heads and faces) wants larger heads, thick limbs, mitten hands, readable faces at 4:5 portrait distance.
- Motion: walk is stiff (audit), gun and arm clip through the scope in S1 (8.2-9.0 s), clinches interpenetrate in S3, hit-stop only freezes one strip, falls are single baked topples.
- Transitions: all six scene boundaries are hard cuts; the only designed transition is S1's closing zoom onto the scope screen. Boundary frames (9.9 s vs 10.0 s etc.) share no motif.
- VFX: bloom over-glow at draft (S4/S5), true refraction shimmer missing (compositor replaced by render_scene.py), oven lids static, hit FX bright at draft.

## 2. Plan

| Phase | Work | Owner (agent) | Status |
|---|---|---|---|
| 1a | Character v2 models (Gary, Manager): bigger heads, chunky clay, mitten hands, better cloth shapes; same skeleton, hooks, `p_` props, shape keys, hole system; new files `gary_v2`, `manager_v2` | C1 | pending |
| 1b | Motion library v2: better walk/run/idle/point/shout/punch/shove/hit reactions/fall/get-up/hug/whip reaction as actions on the existing skeleton (works on v1 and v2 rigs), overlap, anticipation, weight | C3 | pending |
| 1c | Scope transition system: 6 transitions using the oscilloscope screen as the motif, with compositing script | T | pending |
| 1d | Visual critique of v1.1 per scene (VFX, readability, human issues) feeding Phase 2 | Critic | done: `DevLog/v1/DevLog-005-critique-v1_1.md` |
| 2a | NPC family v2 (vendors, customers, NVYDIA leather, OPENAY hugger) on the v2 kit | C2 | after 1a |
| 2b | Scene passes S1..S6: swap to v2 characters, apply motion library, fix critic list, VFX polish, integrate transitions' scene-side hooks | six scene agents | after 1a-1d (+2a for S3, S6) |
| 3 | Re-render draft p50 per scene, review, then standard 1080x1350 renders, transitions composite, audio re-sync (VO and SFX times if scene timing shifts), final mux | coordinator and render agent | after 2b |

Principles: new asset files only (never overwrite v1 assets), same hook and property names so scenes can switch by an id constant, scene time stays 10 s per scene so VO and SFX cue times remain valid, deterministic baked animation, no emoji, no git.

## 3. Progress log

- 2026-10-02: plan written.
- 2026-10-02: critic report received. Cross-cutting: burned-in subtitles still carry the old DevLog-001 narration (S6 lags VO by about 2 s), draft HUD (FX line, timecode, data cards, S7 debug line) burned in, thin captions low contrast on pale frames, blown-out ceiling lamps with dither speckle, humans only 10-19 percent of frame height (faces/holes/guns unreadable), unrelated colour worlds per scene. Top 10 ranking is in the critique file; Phase 2 scene agents get their scene tables plus: subtitles must follow `scripts/audio/make_vo.py` SEGS, a HUD gate (off for final), human framing 35-50 percent of frame height at key beats, S3 day/night strobe (photosensitivity) must be softened, S3 28-30 shooting gag rebuilt as readable mediums, S2 hop flashes removed.
