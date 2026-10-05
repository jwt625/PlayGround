# DevLog-005-characters-v2: Gary and Manager v2 (chunky clay)

| Field | Value |
|---|---|
| Date started | 2026-10-02 |
| Owner | agent C1 (plan: `DevLog/DevLog-005-v1_2-polish-plan.md`, phase 1a) |
| Inputs | v1 kit `scripts/assets/characters/*.py` (not edited), `assets/components/characters/{gary,manager}.{blend,json}` (not edited) |
| Outputs | `assets/components/characters/{gary_v2,gary_v2_holes30,manager_v2}.{blend,json}`, `scripts/assets/characters/v2/`, `assets/components/characters/previews_v2/` |
| Status | built, previews rendered, self-audit done (see below) |

## Plan

Design values (not measured from a source; flagged as design choices): Gary crown 1.69 m (hat top 1.754 m), Manager 1.85 m skull crown, head (chin to skull crown) about 1/4.5 of height, wider shoulders and hips, thicker limbs, mitten hands, bigger boots and hat. All unchanged heights from v1 so real-scale hardware keeps working.

Compatibility strategy (kept strictly):
- Same bone names and hierarchy; same joint table J and same K as v1, so every limb bone head/tail, hand frame, finger bone and IK rest equals v1 (checked: only neck tail, head head, jaw, eye_*, lid_* moved). New bones: `belly` (both), `hat` (Gary), `tie_1..3` (Manager).
- Same HOOK_* names and meanings; Manager steam hooks moved to the ears of the bigger head; all other hook positions equal to v1.
- Same root custom properties plus new `p_squash`, `p_squash_head`, `p_jiggle_belly`, `p_jiggle_hat`, `p_tie_swing`.
- Same shape-key names (14 keys) on the same objects; two new face objects (`glints`, `lips`) carry the same keys and drivers.
- Asset ids `gary_v2`, `gary_v2_holes30`, `manager_v2`; actions `ACT_<asset_id>_<name>` (21 each, same names as v1), re-baked by the unchanged pose engine on the v2 rig (equal to v1 keys because the rest pose is equal); v1 actions also play on the v2 rig (tested, see compatibility).
- Head: the v1 head model is built in v1 head units and mapped by an affine transform T (x 1.9, y 1.42, z 1.68 about the chin, chin at 1.358 m baseline, 0.02 m forward) so all expression semantics survive; eyes, lids, lips, teeth, glints re-proportioned in head units first.

## Checklist (status 2026-10-02)

- [x] M0 kit copied to `scripts/assets/characters/v2/` (v1 modules untouched), head transform, new head parts (glints, lips, thick lids, bigger brows, nose, ears), mouth ring moved up for a longer chin
- [x] M1 Gary v2 body (thicker torso/limbs, taper at the pelvis), mitten hands, t-shirt with hems and neck band, overalls (waist band, legs with rolled cuffs, crease tubes, z-banded fold bump, bib patch flush with a pocket, pen, buckles, rivets, straps), tool belt (buckle, holster with pliers, flap pouch, back pouch, cable-tie roll), work boots, tilted hard hat with ridge
- [x] M2 Gary rig additions (belly and hat bones, squash and jiggle drivers), 6 holes, hooks, 21 actions, metadata, `gary_v2.blend` and `gary_v2_holes30.blend`
- [x] M3 Gary previews and compatibility test in a scratch scene with `asm.py`
- [x] M4 Manager v2 (open-front jacket shell with welded V and lapel ribbons, jacket collar, shirt collar band and points, cuffs, tie knot and wide blade on a 3-bone chain, trousers with a break and crease arcs, thick shoes, hair with quiff and nape)
- [x] M5 Manager previews, compatibility test, v1 vs v2 comparison sheets
- [x] M6 self-audit (below), final report

## Decisions

1. Keep the v1 skeleton rest pose for everything except the head cluster, so v1 actions and any C3 actions authored on the v1 skeleton retarget by bone name without re-authoring. Cartoon proportions come from mesh volume and a 1.68x taller / 1.9x wider head, not from bone lengths. Consequence: legs are as long as v1 (hip 0.925 m baseline); the figure still has long legs, softened by thick thighs, wide trousers and big boots.
2. Head ratio 1/4.5 (measured), the chin sits at the shoulder-joint height (no visible neck): clothing necklines were lowered to z 1.385-1.395 baseline so the shirt does not cover the chin or mouth.
3. Head pivot (head bone head) lowered from 1.55 to 1.47 baseline (centre of the bigger head); neck bone tail follows.
4. Finger bones now use XYZ rotation mode so the existing curl drivers work (v1 left them in quaternion mode, so v1 hands never curled). Authored curl values in actions are therefore now visible.
5. Overalls: bib as a structured grid patch on the torso front (clean boundary, flush within 5 mm, solidified inward), waist band as a tapered tube, legs start above the hip with round caps; pelvis taper removes the U-shaped crotch bulge and the diaper seam of v1. Hems end open with a rolled cuff ring; no flat caps (the flat cap z-fought with the skin shin in the first build).
6. Manager jacket: each ring runs from the left V-edge around the back to the right V-edge (clean edge, welded at the closed lower front), lapels are ribbon tubes along the edge, so there is no jagged face removal and no white wedge. Tie blade lies inside the V at 19 mm offset (jacket at 26 mm).
7. Expression gains (brow x1.7, corner x1.5, gap x1.2, lid x1.15, smirk and cheek x1.5, width x1.25) applied in `chars_head.E()` so the 14 keys read at 200 px head height. Neutral mouth gap 7 mm (head units) so lips read as a soft line, closed expressions (flat, smug, dead_eyed) stay closed.
8. Squash axis: pose-bone scale on the `root` bone (about the floor) and the `head` bone; the bone Y axis is the long axis (first version used Z and gave inverted squash; caught by the numeric test).
9. Hole base radius 0.045 m (v1 0.042) and cut half-length 0.30 m (v1 0.14): the belly and the outer layers (belt, creases) reach 0.23 m in front of the hole axis; with 0.20 m the belt and crease tubes were not cut (found in the hole close-up). Positions, bones and property names are the same as v1.
10. No logos on either character (none needed).

## Compatibility test results (2026-10-02)

Script `scripts/assets/characters/v2/verify_v2.py` (results saved in `previews_v2/verify_*_v2.json`):
- bones: none missing; hierarchy identical for all v1 bones; added gary: belly, hat; manager: belly, tie_1, tie_2, tie_3.
- hooks, shape keys (all 14, same objects plus lips and glints), holes (names and properties), 21 action names: all present.
- root props: all v1 props present; added p_squash, p_squash_head, p_jiggle_belly, p_jiggle_hat, p_tie_swing.
- rest pose: only neck (tail), head (head), jaw, eye_L/R, lid_up/lo_L/R differ from v1.
- v1 actions played on the v2 armature vs the v1 rig (hands, feet, forearms, head-bone tail; 13 actions, 6 frames each): worst deviation 0.031 m (hug_leg, Manager; jolt_hit 0.030 m) and 0.028 m (Gary), typical actions below 0.01 m (idle 0.003, walk 0.002, point and slap 0.0 m); the deviation is the moved head pivot (0.077 m x rotation angle), hands and feet match exactly for IK-driven actions.
- squash test: p_squash_head 0.5 raises head height from 0.378 to 0.567 m (x1.5); p_squash -0.4 lowers the figure to 0.6 of its height.
- evaluated triangles (render subdivision, all modifiers): Gary 74.3k, Manager 77.9k (v1: 126.5k and 129.7k at render subdivision, 53-56k at viewport subdivision).
- measured head height (evaluated head mesh): Gary 0.378 m = 1/4.47 of the 1.689 m crown (v1: 0.225 m = 1/7.5), Manager 0.414 m = 1/4.50 of 1.862 m (v1 0.246 m = 1/7.55).
- heights with hat/hair: Gary 1.754 m (v1 1.753), Manager 1.862 m (v1 1.862).
- scene test via asm.py (`test_v2_asm.py scene`): NLA sequence walk, idle, point, aim_gun, jolt_hit, topple_back with a shotgun on HOOK_gun_grip_R (Manager) and HOOK_hand_R (Gary); stills 540x675 EEVEE 16 samples in `previews_v2/*_v2_scene_test.png`; holes keyed on the roots cut cleanly (`*_hole_*` previews and `holes_closeup`).
- v1 vs v2 comparison sheets: `previews_v2/{gary,manager}_compare_NN.png` (12 poses each, v1 left, v2 right).

## Open questions for Wentao

1. Legs: the skeleton heights are the v1 ones (compat). If the figures should be stockier (shorter legs, 1/4 head), the skeleton and all actions must be re-authored (C3 scope); say so and I would rescale joint heights consistently.
2. Gary t-shirt colour (cream, as v1) against blue overalls and white hat-brim shadow: keep, or switch to a darker tee for silhouette contrast?
3. Manager tie width and lapel width are design choices (blade 0.096 m at its widest, lapel 0.09 m).
4. The v1 finger-curl bug fix changes how hands look in v1 actions (now curled where the action authored curls); confirm this is wanted for the scene agents.

## Audit notes (self-audit, 2026-10-02)

- Found and fixed during the build: flat cap of trouser hem z-fighting with the skin shin; boots shaft thinner than the skin shin (patchy); shirt neckline and neck tube covering the chin and poking through the mouth bag (neck tube shortened to end below the chin); torso front at the neck deeper than the chin so the mouth was hidden (upper torso depth reduced); glints outside the pupil; lips too thin and octagonal (loop upsampled and smoothed); fold shader banding looked like stacked rings (amplitude and band count reduced); squash axis inverted; finger drivers dead (rotation mode).
- Known remaining: elbows/knees are ball joints (capsule overlap at deep bends); Gary hat is a single piece with no inner liner; hair is shell plus blobs; shirt bottoms and trouser tops are hidden under outer layers but exist as open shells; the Manager's sleeve elbows have no crease geometry (only the suit bump); tie_1..3 chain has no automatic secondary motion (drive p_tie_swing); `HOOK_muzzle_self` keeps the v1 position (6 cm clearance from the head side).
- Fresh-context audit (subagent, 2026-10-03) result: compatibility OK-checked (bones, hierarchy, hooks, props, 14 keys, holes, 21 actions; verify script reproduced); v1 files untouched; sizes within limits (blends 8.7-9.0 MB, previews 20 MB); no emoji or absolute paths. Findings acted on: (MAJOR) Manager tie blade was a thin stick: np.interp was given descending x values, so the width profile collapsed; fixed (ascending xp), V opening widened (half-width 0.128 m at the neckline), jacket sleeves thinned by 10 mm to reduce the boulder-shoulder look; (MINOR) skin patches at the boot tops: boot shaft and collar enlarged, trouser hem flare raised so the hem covers the collar; devlog and JSON corrections (hole half-length 0.30 m, Manager triangles 77.8k, `HOOK_muzzle_self` is 0.016 m lower than v1 in z, v1 `dimensions` and `hole_system_usage` keys restored by `finalize_meta_v2.py`, Manager stature definition note). Not changed: leg length and boot size are bounded by the v1 skeleton (open question 1); hole_1 clips a strap buckle (positions are v1's); pliers handles are visible only from the right side; hair edge stair-steps near the ear (head grid resolution); the jiggle strip is subtle by design (props at rest 0).
- Scene-agent trap list (from the audit): `scripts/film_v1/s01_copper.py` line about 365 hard-codes `ACT_gary_pull_cable` and object names; all object, armature, root, material and node-group names carry the `_v2` suffix (`gary_v2_rig`, `ROOT_gary_v2`).

## Progress log

- 2026-10-02: read plan, ASSET_SPEC, v1 DevLog-003, JSONs, kit, asm.py; v1 kit copied to `scripts/assets/characters/v2/`.
- 2026-10-02: first Gary build (head transform T, new torso table, thick limbs, mitten hands, overalls, hat). First look: head 1/5, diaper bulge at the crotch and nude torso showing through; fixed taper and neckline; head enlarged to 1/4.5 (Sx 1.9, Sz 1.68).
- 2026-10-02: faces: bigger eyes with glints, thick lids, lips tube, nose and ears re-proportioned, mouth moved up two rings (IM 14) for a visible chin; neck tube shortened (it showed through the open mouth).
- 2026-10-02: Gary 14-expression sheet reads at 200 px head height; hole and head-hole cut-outs verified; actions verified on the v2 rig (walk, aim_gun, point, shout, fist curl).
- 2026-10-02: Manager v2 built (jacket shell with V, lapels, collars, tie chain, trousers break, shoes, hair); heights matched to v1 (hair and hat tuned: Gary 1.754 m, Manager 1.862 m).
- 2026-10-02: numeric verification script written and run for both; squash axis bug found and fixed; metadata JSON extended (measured values, assembler notes).
- 2026-10-02: hole close-up showed belt and crease tubes uncut (cut half-length too short for the belly); half-length raised to 0.30 m (head hole 0.26 m). Preview fixes: pose reset after the pose board (the anger sheet inherited a pose), topple framing, fist curls reset.
- 2026-10-03: audit subagent run; tie width bug (np.interp with descending x), boot/hem gaps and sleeve thickness fixed; JSON touch-up script `finalize_meta_v2.py`; Manager rebuilt with `run_manager_v2.sh`.
- 2026-10-02: full rebuild with `run_all_v2.sh` (builds, previews, scene tests, verification); outputs listed below.

## Outputs

- `assets/components/characters/gary_v2.blend/.json`, `gary_v2_holes30.blend/.json`, `manager_v2.blend/.json`
- `scripts/assets/characters/v2/`: `chars_geo.py`, `chars_mat.py`, `chars_head.py`, `chars_body.py`, `chars_outfit.py`, `chars_actions.py` (copies of the v1 kit, extended), `chars_head_keys.py`, `build_gary_v2.py`, `build_manager_v2.py`, `meta_v2.py`, `render_previews.py`, `test_v2_asm.py`, `verify_v2.py`, `run_all_v2.sh`, `run_manager_v2.sh`, `finalize_meta_v2.py` (run after the builds)
- `assets/components/characters/previews_v2/`: front, three-quarter, back, closeup head, `*_faces.png` (14 expressions plus anger+flush), `*_hand_closeup.png` (open, fist, point, grip), `gary_v2_hole_*`, `gary_v2_holes_closeup*`, `gary_v2_headhole*`, `*_poseboard.png`, `*_squash_jiggle.png`, `*_compare_NN.png`, `*_scene_test.png`, `verify_*.json`
