# DevLog-005-characters-v2: Gary and Manager v2 (chunky clay)

| Field | Value |
|---|---|
| Date started | 2026-10-02 |
| Owner | agent C1 (plan: `DevLog/DevLog-005-v1_2-polish-plan.md`, phase 1a) |
| Inputs | v1 kit `scripts/assets/characters/*.py` (not edited), `assets/components/characters/{gary,manager}.{blend,json}` (not edited) |
| Outputs | `assets/components/characters/{gary_v2,manager_v2}.{blend,json}`, `scripts/assets/characters/v2/`, `assets/components/characters/previews_v2/` |
| Status | in progress |

## Plan

Design values (not measured from a source; flagged as design choices): Gary crown 1.69 m (hat top 1.753 m), Manager 1.85 m, head (chin to skull crown) about 1/5 of crown height, shoulders and hips wider, limbs thicker, mitten hands, bigger boots and hat.

Compatibility strategy (kept strictly):
- Same bone names and hierarchy; same joint table J and same K as v1 (so every limb bone head/tail, hand frames, finger bones and IK rest are identical to v1). Only the head bone pivot, jaw, eye and lid bones move (bigger head). New bones are added (belly, hat, tie chain, ...).
- Same HOOK_* names, root custom properties (`p_expr_*`, `p_anger`, `p_flush`, `p_hole_*_radius`, `p_head_hole_radius`, `p_stature_m`, `p_scale`) plus new `p_squash`, `p_squash_head`, jiggle props.
- Same shape-key names on the same object names (`head`, `pupils`, `lids`, `brows`, `nose`, `teeth_up`, `teeth_lo`, `tongue`, `drops`); new face objects get the same 14 keys and drivers.
- Asset ids are `gary_v2`, `manager_v2` (and `gary_v2_holes30`), so actions are `ACT_gary_v2_<name>` (the assembler builds the name from root['asset_id']); the action set and keys are the v1 library re-baked by the same pose engine on the v2 rig.
- Head is built with the v1 head model (so the 14 expression shape keys keep their semantics) and then mapped by an affine head transform T (bigger, rounder), with larger eyes, thick lids, thick lips, glints, bigger nose and ears.

Phases and checklist:
- [ ] M0 kit copied to scripts/assets/characters/v2, head transform, new head parts
- [ ] M1 Gary v2 body, mitten hands, outfit (t-shirt, overalls with folds, bib pocket, buckles, belt/pouches/pliers, boots, hard hat)
- [ ] M2 Gary rig additions (jiggle, squash), holes, hooks, actions, metadata, saved blend; gary_v2_holes30
- [ ] M3 Gary previews and compatibility test in scratch scene with asm.py
- [ ] M4 Manager v2 (suit with lapels and collars, shirt collar, tie with jiggle chain, hem, trouser breaks, shoes)
- [ ] M5 Manager previews, compatibility test, v1 vs v2 comparison sheets
- [ ] M6 audit and final report

## Decisions

(see progress log)

## Open questions

(see progress log)

## Progress log

- 2026-10-02: read plan, ASSET_SPEC, v1 DevLog-003, JSONs, kit, asm.py; v1 kit copied to `scripts/assets/characters/v2/`.
