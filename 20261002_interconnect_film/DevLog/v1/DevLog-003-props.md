# DevLog-003-props: gag and hand props

Owner: props agent. Spec: `../../assets/ASSET_SPEC.md`. Scripts: `../../scripts/assets/props/` (shared category helper `props_lib.py`). Outputs: `../../assets/components/props/`.

## Plan

Build order: shotgun (+ shells), aoc_whip, eggs_plate, office_gags, cable_ties_tools. Each .blend saved as soon as usable; previews 900x675 EEVEE 16 samples; one process at a time. Build scripts re-open the saved blend before rendering previews (proves standalone load, refreshes drivers).

## Checklist

| Item | Status |
|---|---|
| props_lib.py helper | done |
| shotgun (clay + wood variants, break rig) | done 2026-10-02 |
| shotgun_shells (live + spent) | done 2026-10-02 |
| aoc_whip | done 2026-10-02 |
| eggs_plate | done 2026-10-02 |
| office_gags | done 2026-10-02 |
| cable_ties_tools | done 2026-10-02 |

## Sources (access date 2026-10-02)

- QSFP-DD MSA Hardware Specification (Rev 4.0 PDF, qsfp-dd.com mirrors; Rev 5.0/5.1 at qsfp-dd.com): module width 18.35 mm, height 8.5 mm, module length about 89.4 mm (found as drawing dimension strings 89.4 / 92.4 in the Rev 4.0 PDF; which figure each belongs to was NOT verified because figure text is vector art). Accuracy B.
- 12 gauge: nominal bore 18.5 mm (Wikipedia 12-gauge shotgun); 2-3/4 in (70 mm) shell; SAAMI hull base diameter 0.809 in (about 20.55 mm) (shotgunworld thread, secondary source).
- Cable tie 200 x 4.8 mm: width 4.8 +-0.2 mm, length 200 +-5 mm, PA66, max bundle 50-52 mm (hwlok GT-200ST page; accu-components datasheet listing). Head dimensions not found for this exact size: estimates (about 7 x 6 x 4 mm, +-1 mm) flagged in metadata.
- Estimates (flagged in metadata): shotgun barrel OD profile, stock curves, receiver size; egg 57 x 44 mm (from task brief); plate 270 mm (task brief).

## Log

- 2026-10-02: props_lib.py written; shotgun script builds both variants, rig with drivers (note: driver changes headless need `root.update_tag()`; previews are rendered after re-opening the saved blend). Break-open verified (muzzle drops about 304 mm at p_break=1).
- 2026-10-02 (resume after usage limit): economical mode; contact-sheet previews only.

- 2026-10-02: shotgun.blend (30k tris, 2 variants) and shotgun_shells.blend (4.3k tris) saved with JSON and previews in assets/components/props/. Break-open, hooks and drivers verified after reload.
- 2026-10-02: aoc_whip.blend saved (87k tris, 24-bone rig, 12 plugs, 8 ties, tape; actions POSE_straight, POSE_coiled, ACT_whip_crack_demo) with JSON and previews. QSFP-DD length 89.4 mm not figure-verified.
- 2026-10-02: eggs_plate.blend saved (8 ASSET_ collections: egg, egg_cracked, fried_egg, plate, spatula, frying_pan, plate_with_eggs, pan_with_egg; 70k tris); fried egg shape keys spread/edge_crisp/yolk_dome/bubbles, vertex groups, UV_topdown + UV_radial documented in JSON.
- 2026-10-02: office_gags.blend saved (11 ASSET_ collections, 245k tris incl. calendar day text); preview review next.
- 2026-10-02: cable_ties_tools.blend saved (7 ASSET_ collections, 77k tris). All five required assets now have a usable .blend; remaining: review contact sheets, fix defects, final audit.

## Decisions / deviations

- Origin of shotgun = stock wrist (HOOK_grip_R), barrels along -Y, muzzle hooks have local +Z along the firing direction.
- Both shotgun variants live in `shotgun.blend` as `VARIANT_clay` (visible) and `VARIANT_wood` (hidden) under `ASSET_shotgun`, sharing one rig.

- 2026-10-02: all five required assets saved and reviewed on contact sheets. Fixes after review: egg_cracked units (mm vs m) corrected; preview PNGs quantised to 256 colours (scripts/assets/props/shrink_previews.py, previews 13 MB, category 74 MB); shotgun lighting/framing tuned. Not done: fresh-context audit subagent (budget), checkering on the wood shotgun, bending pages on the calendar.

## Final state (exact)

Done: shotgun (clay + wood, break rig, 7 hooks, drivers), shotgun_shells (live, spent), aoc_whip (24 bones, 12 plugs, 8 ties, tape, 3 actions), eggs_plate (8 items; fried egg keys/groups/UVs), office_gags (11 items), cable_ties_tools (7 items).
Partial: hard-hat peak is a flat tile (reads weak), shotgun hammers and trigger guard are thin on the wood variant, calendar pages are rigid.
Not started: audit subagent pass, LOD variants.

## Open questions for Wentao

- Wood variant has no checkering (bump-only simplification).
- QSFP-DD plug length 89.4 mm: please confirm against the MSA figure (the drawing text could not be machine-extracted).
- Price balloon nominal radius 250 mm (v0.2 value): keep, or choose a real party-balloon size?
- Shotgun origin is the stock wrist (hand mount); say if you prefer a floor origin.

## Audit corrections

(none yet)
