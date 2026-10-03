# DevLog-003-characters: clay character library (Gary, Manager, NPC family)

| Field | Value |
|---|---|
| Date started | 2026-10-02 |
| Owner | characters agent (v1.0 asset library) |
| Spec | `assets/ASSET_SPEC.md`, cast table in `DevLog/DevLog-001-story-scenes-assets-proposal.md` Section 3 |
| Status | in progress |

## Plan

1. Shared character library `scripts/assets/characters/chars_common.py`: numpy loft geometry, clay and hole shaders, humanoid armature (FK + IK limbs, finger curl, jaw, lids, eye look-at), face part system with expression shape keys, hole system, hooks, metadata.
2. `build_gary.py` (+ `gary_holes30` variant), `build_manager.py`, `build_npc.py` (family: base + six shirt variants incl. 0.8 scale hugger).
3. `char_actions.py`: 19+ actions per rig as `ACT_<asset>_<name>` with fake users.
4. `render_previews.py`: front / three-quarter / back, face expressions, hole close-up, pose sheets.
5. Verify visually, audit with a fresh-context subagent, write metadata.

## Checklist (final status, 2026-10-02)

- [x] gary.blend and gary_holes30.blend: body, t-shirt, overalls (bib, straps), tool belt with pouches/pliers, boots, orange hard hat with brim and ridge, 14 expression shape keys (the 10 requested plus angry, scared, happy, yell), FK/IK armature, 6 holes, hooks, 21 actions
- [x] manager.blend: suit, shirt, red tie, shoes, hair; p_anger (brows, lids, snarl, yell mouth, nostril flare, vein bump), p_flush; hooks HOOK_steam_L/R, HOOK_gun_grip_R, HOOK_gun_support_L, HOOK_whip_grip_L, HOOK_paper_R; 21 actions
- [x] npc family (separate .blend each): npc (blank grey tee), npc_molexx, npc_nubiss, npc_terahop, npc_ayarr, npc_nvydia, npc_openay (0.8 scale hugger, OPENAY over ANTHROPY); each has hole_1 (shoot-able), expression rig, hooks, 21 actions
- [x] previews (900x675 views; face sheets; hole straight-on and oblique; pose sheets): gary, manager, npc_molexx, npc_openay full sets, other npc variants front/three-quarter/back; category previews 23 MB
- [x] metadata JSON per asset: rig, hooks, holes (axis, position, radius), props, actions, materials, sources, simplifications
- [x] append test: a fresh file appended ASSET_gary + actions; expression, hole and rim drivers follow the appended root
- [ ] fresh-context audit subagent not run (token budget); self-audit below
- [ ] not done: thumbprint close-up preview; eyelid-blink/expression keys inside actions (expressions are driven from the root props); toe bones; per-asset LOD

## Decisions

- Baseline modelling at 1.75 m crown height with uniform scale K per character; actions author IK targets in baseline coordinates and scale by K, so each rig has its own baked actions.
- Hole: shader node group per asset, bone-parented hole empties (local Z = axis), cutout radius = base_radius x root property, rim tube from front/back ray casts. Default radius properties 0.0 (intact); `gary_holes30` defaults 0.3.
- Thumbprint bump uses a mesh attribute (rest_pos) so the pattern follows the deformed body.
- NPC variants are separate files, not one family file, so each can be appended on its own.
- Wordmarks are raised text meshes (Arial Black outlines converted at build time), parody names only.
- Gary 1.75 m total with hat (crown 1.69), Manager 1.85, vendors 1.62-1.88, hugger 1.40.

## Open questions for Wentao

1. Hugger height: 0.8 x 1.75 = 1.40 m was used; should he instead be about 1.6 m?
2. Should expressions be keyframed inside the body actions (currently on the root custom properties, keyed by the assembler)?
3. Manager gun-holding geometry assumes the shotgun forend is about 0.25 m ahead of the grip; adjust once the props asset exists.
4. Anatomical limb fractions were quoted from memory (Drillis and Contini); verify if exact numbers matter.

## Audit corrections (self-audit)

- Found and fixed: pose engine overwrote poses because the action was evaluated while solving (now solved with no action, then baked); flush colour covered hands (now masked above neck height); veins appeared at the nose (scaling bug in the vein window); drops visible at neutral (basis now zero size); tool-less previews of holes were off-axis (now straight on).
- Known remaining issues: shirt hem and trouser top overlap shows a diaper-like seam on NPCs; Gary bib sits slightly proud of the torso; fingers are thin at small previews; walk is stiff; tears/sweat drops are simple lathes; actions were checked in stills for Gary, Manager and npc_molexx only.

## Progress log

- 2026-10-02: read spec, storyboard, crude cast code. Verified in Blender 4.2.3 headless EEVEE Next that (a) a Texture Coordinate node bound to an Empty plus alpha 0 gives a clean see-through cylinder cut, (b) a driver on a node-group Value node reading a custom property of the root empty changes the radius live.
- 2026-10-02 (after resume): library modules written and tested: `chars_geo.py` (numpy loft kit), `chars_mat.py` (clay, thumbprint bump group, hole cutout group), `chars_head.py` (head with modelled mouth, 14 expression shape keys on 10 face parts), `chars_body.py` (armature with FK/IK, finger curl drivers, hole empties and rims, hooks), `chars_outfit.py`, `chars_actions.py` (pose engine + 21 actions), `render_previews.py`. `build_gary.py` ran: `gary.blend` and `gary_holes30.blend` saved (34k tris, 6 holes, 10+ expressions, hooks). Actions are not yet baked into the saved blends (tested on a scratch copy; IK pose engine works).
- 2026-10-02: gary/gary_holes30/manager saved with actions (pose engine fixed to solve poses with no action assigned, then bake keys). Manager preview sheets rendered. Next: npc family.
- 2026-10-02: npc family (7 blends) built; shirt wordmarks are raised text geometry projected onto the shirt (Arial Black outlines converted to mesh at build time; no font or image dependency). `run_all.sh` rebuilds everything and renders previews.
- 2026-10-02: all assets rebuilt with `scripts/assets/characters/run_all.sh`; metadata extended with `finalize_meta.py`; previews pruned to 23 MB; category size 92 MB (10 blends about 6-7 MB each). Tris (base mesh): gary 34k, manager 35k, npc 30-33k; with modifiers 49-56k.
