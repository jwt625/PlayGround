# DevLog-005-npc-v2: NPC family v2 (vendors and customers)

| Field | Value |
|---|---|
| Date started | 2026-10-03 |
| Owner | agent C2 (plan: `DevLog/DevLog-005-v1_2-polish-plan.md`, phase 2a) |
| Inputs | v2 kit `scripts/assets/characters/v2/` (imported, not edited), v1 NPC assets `assets/components/characters/npc*.{blend,json}` (not edited), critique `DevLog/v1/DevLog-005-critique-v1_1.md` (S3/S6 readability) |
| Outputs | `assets/components/characters/npc{,_molexx,_nubiss,_terahop,_ayarr,_nvydia,_openay}_v2.{blend,json}`, code `scripts/assets/characters/v2/npc/`, previews `assets/components/characters/previews_v2/npc/` |
| Status | done (2026-10-03); audit corrections applied |

## Plan

1. Own module `scripts/assets/characters/v2/npc/npc_kit_v2.py` on top of the v2 kit: untucked tee with flared rolled hem, tapered trousers (diaper-seam fix), raised wordmark with depth/offset control, hair shells (short, flat-top, buzz, bald fringe, bob, ponytail, curly blobs), face hair that carries the 14 expression keys (beard/goatee shell generated from the head deformation, moustache tube anchored to head-grid vertices), glasses, accessories (lanyard badge, headset, cap, backpack) toggled by `p_accessory`, leather moto jacket.
2. `build_npc_v2.py <variant>`: one parameter table, same H and build as v1 (skeleton compatibility), v2 silhouette from wide/arm_r/leg_r/belly/head scale/hair/accessory.
3. Build the base and MOLEXX first, verify (kit `verify_v2.py`, asm append test), then the other variants.
4. Previews (540x675, EEVEE 16): front, three-quarter, lineup of 7 (daylight and dark floor), faces sheet (NUBISS, delivered; MOLEXX checked in the session scratchpad only), v1 vs v2 compare, asm test sheets.
5. Fresh-context audit, final report.

## Design values (not measured; flagged as design choices)

Stature (skull crown) = v1 H for every variant (H and build define the v1 joint table, so keeping them keeps every non-head bone at the v1 rest position): npc 1.75, MOLEXX 1.80, NUBISS 1.70, TERAHOP 1.88, AYARR 1.62, NVYDIA 1.82, OPENAY 1.40 (0.8 x 1.75). The skull crown stays at the v2 baseline 1.749 (x K) for every head shape (chin height = 1.749 - 0.233 x Sz).

| asset | body | head scale (x, y, z) | hair / face | accessory (p_accessory) | colours |
|---|---|---|---|---|---|
| npc_v2 | average (wide 1.0, belly 0.10) | 1.90, 1.42, 1.68 | short | none | grey tee (v1) |
| npc_molexx_v2 | broad: wide 1.12, arm_r 1.12, leg_r 1.06, belly 0.18 | 1.98, 1.40, 1.62, boxy (sup 2.9, jaw 0.02) | flat-top, moustache, bushy brows | lanyard badge | red tee, white MOLEXX (v1) |
| npc_nubiss_v2 | round: belly 0.62, wide 1.10, arm_r 1.12, leg_r 1.10 | 2.04, 1.50, 1.60, round (sup 2.0), big nose | bald with side fringe, full beard, rectangular glasses | headset | green tee (v1) |
| npc_terahop_v2 | lean: wide 0.88, arm_r 0.86, leg_r 0.86 | 1.70, 1.36, 1.72, narrow tapered chin | ponytail, goatee | white cap (bone hat) | purple tee (v1) |
| npc_ayarr_v2 | short, slight: wide 0.98, arm_r 0.96 | 1.98, 1.46, 1.70, big eyes | large curly hair, round white glasses | backpack | yellow tee, black AYARR (v1) |
| npc_nvydia_v2 | solid: wide 1.04, belly 0.15 | 1.88, 1.42, 1.66 | silver buzz cut (lighter than v1 grey) | none | black clay leather moto jacket (sheen, coat) over the v1 green NVYDIA tee, open V |
| npc_openay_v2 | 0.8 scale | 1.96, 1.46, 1.72 | brown bob (v1 near-black) | none | cream tee with dark OPENAY / ANTHROPY (v1: black tee, white text), slate trousers |

Customer colour changes vs v1 are deliberate (critic S6: black-clad customers vanish on the dark floor).

## Checklist

- [x] M0 module + build script + preview/test script
- [x] M1 MOLEXX full build with actions, `verify_v2.py`, asm test (walk, idle, punch_loop, topple_back)
- [x] M2 all variants with actions, verify each
- [x] M3 previews: views, lineup (day and dark), faces (NUBISS: beard + glasses + headset), compare, asm sheets
- [x] M4 audit (fresh-context subagent), corrections, full rebuild with `run_npc_v2.sh`, final report

## Results (2026-10-03)

| asset | stature (skull crown, m) | top incl. hair/cap (m) | head height (m) | stature / head | evaluated tris (render subdiv, all modifiers) | blend MB | v1 action on v2 rig, worst (m) |
|---|---|---|---|---|---|---|---|
| npc_v2 | 1.75 | 1.776 | 0.391 | 4.47 | 59.6k | 7.9 | 0.029 |
| npc_molexx_v2 | 1.80 | 1.829 | 0.388 | 4.64 | 62.1k | 8.2 | 0.030 |
| npc_nubiss_v2 | 1.70 | 1.730 | 0.362 | 4.69 | 71.3k | 8.6 | 0.028 |
| npc_terahop_v2 | 1.88 | 1.931 (cap) | 0.430 | 4.37 | 66.3k (verify; the JSON field says 63.7k, see audit 2) | 8.3 | 0.031 |
| npc_ayarr_v2 | 1.62 | 1.681 (curls) | 0.367 | 4.42 | 75.2k | 9.2 | 0.027 |
| npc_nvydia_v2 | 1.82 | 1.827 | 0.402 | 4.53 | 74.4k | 8.6 | 0.030 |
| npc_openay_v2 | 1.40 | 1.413 | 0.321 | 4.37 | 62.7k | 8.2 | 0.023 |

Head height = evaluated head object (chin to skull crown, ears included), measured. v1 NPCs: about 1/7.5, 122-125k tris at render subdivision.

Compatibility (`verify_v2.py`, JSONs in `previews_v2/npc/verify_*_v2.json`), all seven: no bone, hook, root prop or action missing; bone hierarchy identical; bones added: belly (all), hat (TERAHOP); root props added: p_squash, p_squash_head, p_jiggle_belly, p_jiggle_hat, p_tie_swing, p_accessory (vendors only); 14 shape keys equal, shape-key objects added glints, lips (+ moustache MOLEXX, beard NUBISS/TERAHOP); hole_1 same name/prop/bone/position (v2 radius 0.045, half-length 0.30 baseline x K); 21/21 actions; rest pose differs only for neck tail, head, jaw, eye_*, lid_* (bigger head); worst deviation of v1 actions on the v2 rig 0.023-0.031 m over hands, feet, forearms, shins, chest and the head-bone tail (attributed to the head pivot moved down, as for C1's Gary/Manager).

asm append test (`preview_npc_v2.py asm`): every asset appended with actions; asm.walk, then idle, punch_loop (fight), topple_back (fall) as NLA strips; stills at 540x675 EEVEE 16 viewed: `previews_v2/npc/<id>_asm_test.png` (tiles downsampled 2x). Jacket, backpack, lanyard, cap and headset follow the rig.

Visual checks viewed: front/three-quarter (540x675), lineup on daylight and dark data-hall floor (`npc_v2_lineup*.png`: all seven separate from the dark floor; leather jacket reads by sheen; cream OPENAY tee and silver NVYDIA hair read), face sheets NUBISS (delivered) and MOLEXX (scratch): beard and moustache follow all expressions, v1 vs v2 compare (`npc_v2_compare.png`: v1 crotch/diaper seam gone in v2).

## Switching a scene (for scene agents)

- `asm.append("characters/<id>_v2", actions=True)` instead of `characters/<id>`; `asm.play(a, "walk")` etc. resolve `ACT_<id>_v2_<name>` from `root["asset_id"]`.
- NVYDIA: `npc_nvydia_v2` replaces both `npc_nvydia` and `npc_nvydia_leather` (the jacket is built in).
- Hard-coded names must change: objects `<id>_v2_*`, `ROOT_<id>_v2`, `<id>_v2_rig`, materials `MAT_characters_<id>_v2_*`, actions `ACT_<id>_v2_*`. Current scene code (checked 2026-10-03): S3 `s03_npo.py` line 185 `VEND_ASSET` dict (change the four values to `npc_<x>_v2`; action names are built from `root["asset_id"]`); S6 `s06_fiber.py` line 394 `characters/npc_nvydia_leather` -> `characters/npc_nvydia_v2`, line 406 `characters/npc_openay` -> `characters/npc_openay_v2`. No other literal NPC names in `scripts/film_v1/s0*.py`.
- Vendor accessory off: `root["p_accessory"] = 0` (drivers hide the accessory objects in render and viewport).
- Heads are 1.6-1.7x taller than v1 (measured head height v2 / v1 per asset): anything placed relative to the face (captions, FX at the mouth) should use `HOOK_head_top` or the head bone, not v1 offsets. Hole_1 radius is larger (0.045 vs 0.042 baseline).
- Customers' colours changed (OPENAY cream tee and brown hair; NVYDIA leather jacket over the green tee): s03_fx crumbs for vendors still match (vendor shirt/skin hex unchanged).

## Decisions

1. Heights and build = v1 (they define the joint table; changing them would break retargeting of actions authored on the v1 skeleton). Silhouette variety comes from radii (wide, arm_r, leg_r, belly), head scale (headS per asset with the skull crown fixed), hair and accessories.
2. Face hair is generated from the head deformation function, so it carries real expression shape keys (not rigid): beard/goatee = offset shell of head-grid faces, moustache = tube re-lofted through head-grid anchor vertices per expression.
3. Diaper seam fix: untucked tee with a rolled hem that flares over the trouser waist and thigh tops (offset grows from 0.012 at the chest to 0.038 at the hem), trousers pelvis tapered into the legs (same taper as the v2 skin).
4. NVYDIA wordmark kept on the shirt (where v1 had it) and shown through the open V of the jacket (open from z 1.10 baseline); the leather jacket has no logo.
5. Customers lightened for dark sets (critic S6): jacket base #222227 with sheen 0.40 (light blue-grey tint) and coat 0.35, silver buzz cut, OPENAY cream tee with dark text, brown bob, slate trousers.
6. Accessories as a toggle (`p_accessory`) so scenes can drop them; TERAHOP's cap rides a new `hat` bone so `p_jiggle_hat` works.

## Open questions for Wentao

1. OPENAY v1 identity was a black tee with white text; v2 uses a cream tee with dark text for dark-set readability. Keep, or revert to black?
2. NVYDIA wordmark is small (visible only in the jacket V). Bigger V or drop it?
3. Leg length is still the v1 skeleton (same open question as C1).

## Progress log

- 2026-10-03: read plan, C1 devlog, spec, v1 DevLog-003/004, critique, kit code, v1 JSONs, previews.
- 2026-10-03: module and build script written; all 7 variants built without actions; first look: identities distinct; fixed tee sleeve hems poking through the NVYDIA jacket sleeves, removed fold-shader bands on the tees (read as stripes), lengthened the TERAHOP cap brim, tucked the ponytail.
- 2026-10-03: MOLEXX full build (21 actions) and `verify_v2.py`: bones none missing (added belly), hierarchy identical, hooks identical, props added p_accessory + v2 props, 14 keys (shape-key objects added glints, lips, moustache), hole_1 present, 21/21 actions, rest pose moved only for neck tail, head, jaw, eye_*, lid_*; v1 actions on the v2 rig worst 0.030 m (hug_leg; head pivot), evaluated tris 62.2k (v1 122.9k); head 0.388 m = 1/4.64 of the 1.80 m stature. asm test: walk, idle, punch_loop, topple_back play correctly.

## Audit corrections (fresh-context subagent, 2026-10-03)

No MAJOR findings (contract, untouched v1 files and kit modules, sizes, triangle budget, parody-only wordmarks confirmed). MINOR findings acted on:
1. NVYDIA JSON note said leather sheen 0.55; code uses 0.40: note fixed.
2. TERAHOP triangle count differs between the builder JSON (63.7k) and `verify_v2.py` (66.3k) for the same blend; not resolved (both under 90k); the table quotes the verify value.
3. Moustache was weighted 100 percent to head while its tips sit on head rows with about 0.35 jaw weight: now each moustache vertex takes the head/jaw weights of its nearest anchor vertex.
4. Beard looked grey and see-through at grazing angles: beard sheen removed (rough 0.8). The rage tile (p_anger + p_flush) still shows the lower face mostly red at thumbnail size; not investigated further.
5. AYARR backpack strap crossed the A of the wordmark: wordmark width 0.24 -> 0.21 (checked in the front view).
6. NVYDIA head looked sunk between the jacket shoulders: collar lowered and thinned (height 0.034 -> 0.024), upper sleeve pad tapered (0.012 at the shoulder); still high-shouldered (kit body limit). TERAHOP cap read as a helmet: brim lengthened (0.135 -> 0.160) and flattened (tilt 0.12 -> 0.06 rad); now reads as a cap.
7. Devlog wording corrected (faces sheet location, retarget claim, head growth range, plan step).
8. `run_npc_v2.sh` removes its own default temp dir; `preview_npc_v2.py` PV_CLOSE extras stay in the scratch TMP by design.
Not changed: the 8th tile of the compare sheet is black (7 assets on a 4-column grid); ANTHROPY is small at lineup scale on the dark floor.

## Remaining problems

- All NPCs share the kit face (same eyes, nose, mouth); identity comes from head shape, hair, face hair, glasses and accessories.
- Long legs and high pill shoulders come from the v1 skeleton and the kit's sleeve caps (same as C1).
- Trousers are similar dark tones for all vendors (v1 values kept).
- NVYDIA wordmark visible only inside the jacket V (small).

## Progress log (cont.)

- 2026-10-03: all seven built with actions and verified; previews rendered and viewed; audit run; corrections applied; full rebuild with `run_npc_v2.sh` (log in the session scratchpad): all seven re-verified (no missing items, worst 0.023-0.031 m, 59.6k-75.2k tris), previews 9.9 MB, blends 7.9-9.2 MB.
