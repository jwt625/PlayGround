# DevLog-004-s06-new-assets: S6 v1.1 (MPO assets, leather NVYDIA, whip physics, effects)

Date: 2026-10-02. Owner: S6 agent (v1.0 -> v1.1 feedback wave, DevLog-004-v1_1-feedback-wave.md rows 0:51, 0:58, 0:59). Backups of the v1.0 scene blend and script: session scratchpad `s06/s06_fiber_v1_0_backup.{blend,py}`.

## Status checklist

- [x] 0:51 MPO connectors and receptacles: new asset `mpo_connector_and_adapter` (4 ASSET_ collections), used in the S6 intro
- [x] 0:58 whip physics: baked verlet chain, bending stiffness, tapered mass, travelling wave and crack, contact, hit-stop, squash, SFX markers, thicker bundle
- [x] 0:59 leather jacket: new asset `npc_nvydia_leather`; S6 customer swapped
- [x] known issues: whip thickness, Manager visibility, glow reduction, shotgun note (below)
- [x] cross-cutting: eased camera, handheld noise, hit shake, motion-blur shutter, crumb puffs, FX intensity
- [ ] not done: fresh-context audit; no MPO plugs dressing the pile or tray; ribbon cable not modelled; asm.py/INDEX.md changes only recorded (below)

## New assets

| id | file | size | tris | accuracy |
|---|---|---|---|---|
| mpo_connector_and_adapter (ASSET_mpo_plug, ASSET_mpo_adapter, ASSET_mpo_adapter_plate4) | assets/components/interconnect/mpo_connector_and_adapter.blend (+ .json, previews/mpo_connector_and_adapter_*.png) | plug 12.4 x 8.0 x about 220 mm incl. 150 mm cable stub; adapter 15.0 x 10.0 x 34; plate 106 x 33.6 x 2 | 5,606 / 560 / 824 | ferrule face A (IEC 61754-7 geometry, 6.4 x 2.5, 0.7 pins at 4.6, 0.25 pitch, 8 deg APC), plug housing C, adapter width B, adapter length and plate C |
| npc_nvydia_leather | assets/components/characters/npc_nvydia_leather.blend (+ .json, previews/npc_nvydia_leather_*.png) | 1.82 m stature, 7 MB | 34,936 | C (stylised clay jacket, no logo, no likeness) |

Build scripts: `scripts/assets/interconnect/build_mpo_connector_and_adapter.py`, `scripts/assets/characters/build_npc_nvydia_leather.py` (run lines in the docstrings; the leather build reuses the character modules of build_npc.py, so the base npc_nvydia.blend is untouched). Previews were not size-trimmed: MPO previews are 0.6-0.7 MB each (spec says about 400 KB).

### MPO asset
- Sources (all accessed 2026-10-02, details in the JSON): IEC 61754-7 / 61754-7-1:2014 listings (6.4 x 2.5 mm MT ferrule, 0.7 mm guide pins at 4.6 mm, 0.25 mm fibre pitch; the sample PDF text was not extractable, values come from search summaries), vendor texts for housing about 12.5 x 7.6 mm and 12.4 x 8.3 (existing fiber_connectors estimate), adapter footprint 0.59 x 0.39 in, adapter plate class 106 x 33.6 mm. TIA-604-5 and the standard drawings were NOT read: housing, sleeve, boot, adapter length, plate thickness and pitch are estimates (level C).
- Contents: plug with variants MPO-12 female (default), MPO-12 male (pins), MPO-16 female; 8 deg APC sheared face with fibre holes and fibre cores; key rib on top; push-pull sleeve with ridges and latch recesses (driver `p_sleeve_pull` 0..8 mm on the plug root); crimp shell, boot, 3.0 mm yellow cable stub; adapter with key slot, centre partition, side clips, flange ears; 4-port plate.
- The existing `fiber_connectors` already had boxy MPO-12/16 items; they were not used in v1.0 and are lower detail, so a new asset was built.
- Mating convention: plug face at y = 0 looking -Y, body toward +Y. Adapter mate plane at y = 0, openings at y = -17 / +17, plate plane at y = -8. To insert from the front: yaw the plug by pi about Z. Hooks: plug HOOK_mate / key / grip / cable_end; adapter HOOK_port_front / rear / mate_plane; plate HOOK_port_1..4 and HOOK_mate_plane_1..4.
- Simplifications: no spring, clip, label or dust cap; no polarity flip; no ribbon cable; bright preview seams on the plate are bevel highlights.
- Needed by coordinator: add the two assets to assets/INDEX.md (not edited by me).

### Leather jacket asset
- Build: same body, rig, hooks, holes, faces, shape keys and action set as npc_nvydia (same build code) plus a jacket: closed torso shell (solidified), hem band, cuffs, long sleeves, standing collar with two snapped lapels, metal front zipper with pull tab, two slanted hip zips; dark grey plain shirt below, no wordmark. Materials `MAT_characters_npc_nvydia_leather_leather` (glossy black, coat 0.2) and `_zip`; both carry the bullet-hole cutout.
- Rig: jacket meshes are skinned to the existing armature with build-time vertex weights (torso blend hips/spine_1/spine_2/chest; sleeves upper_arm/forearm) equal to the weighting of the shirt and suit shells. Tested: ACT_npc_nvydia_{walk,slap} from the BASE asset assigned to the leather rig deform the jacket (mean vertex displacement 0.02 m walk, 0.08 m slap, same as the leather asset's own actions), pose-preview sheets rendered.
- Swap procedure (S3 and S6): replace `asm.append("characters/npc_nvydia", actions=True)` by `asm.append("characters/npc_nvydia_leather", actions=True)`. Action names become `ACT_npc_nvydia_leather_<name>` (asm.play / asm.walk resolve them from root["asset_id"], so scripts need no other change; scripts that build the name by hand must use the new id). All `p_expr_*`, `p_anger`, `p_flush`, `p_hole_1_radius` and the hooks (HOOK_hand_L/R, gun_grip_R, gun_support_L, head_top, hug_target) keep their names. Do not use the old and new NVYDIA in the same scene (shared datablock names get .001 suffixes).
- S6 uses it (done); S3 coordinator patch pending.

## Whip physics (0:58)

Module `scripts/film_v1/s06_whip_sim.py`, called from `s06_fiber.py`. Deterministic, no RNG.
- Chain: 24 segments x 0.125 m = the rig's bones, points 0..4 rigid on the hand frame; masses taper 1.0 -> 0.07 (momentum transfer gives the travelling wave and tip speed-up); distance constraints (24 Gauss-Seidel passes x 8 substeps per frame), bending stiffness as second-neighbour constraints with strength 0.9 exp(-(i-3)/7) (the rig's stiffness profile), gravity, damping 0.9996 per substep plus quadratic drag, floor at z = 0.03 m with friction, capsule colliders from Gary's bones (hips, spine_1, spine_2, chest, neck, head, thighs, shins; inelastic with friction).
- Bake: per-frame bone quaternions for whip_01..whip_24 computed by parallel transport from the segment directions, written as F-curves into action `ACT_s06_whip_baked` on `aoc_whip_rig` (frames 97-300); the demo `ACT_whip_crack_demo` is no longer used. Whip root scale is 1; thickness is applied to the meshes (cross-section x7 for cables/ties/tape, x3.5 for plugs; cable 3 mm real -> 21 mm, bundle about 6 cm).
- Two passes: pass 1 finds the first torso contact (speed > 1.5 m/s) and the crack frame (peak tip speed between 7.4 s and contact); pass 2 re-simulates with hit-stop and Gary's reaction. Result at build: crack frame 243 (t = 8.067 s, tip 18.6 m/s, handle point speed about 11 m/s), contact frame 250 (t = 8.300 s, 5.3 m/s).
- Hit-stop: 3 frames (250-253): whip chain frozen and the Manager's arm_whip action frozen (split into 3 NLA strips A / freeze / B). Gary's tying loop is not frozen (0.1 s, not noticeable); jolt_hit starts at frame 253 with a 2-frame blend; Gary squash (scale 1.07, 1.07, 0.88 -> 0.97, 0.97, 1.04 -> 1) keyed around contact; Gary's dread face at contact.
- Sound cues: scene timeline markers `SFX_whip_crack` (frame 243, t = 8.067 s scene-local = 58.07 s film), `SFX_whip_hit_thump` (frame 250, 58.30 s), `HITSTOP_start` 250, `HITSTOP_end` 253. Caption CRACK moves to the crack time, shockwave at the tip position at the crack frame, impact stars + shockwave + 14 clay crumbs (seed 6, ballistic keyed paths) at the contact point.
- Staging changes: Manager moved 0.5 m closer to Gary (x 2.55 -> 2.05) because at 2.3 m the simulated tip passed 0.13 m short of Gary's torso; the Gary tying loop now runs to 10 s (the loop strip ended at 8.0 s and Gary would otherwise pop to T-pose when sim pass 1 evaluated him).
- Limits: the Manager's hand path comes from the library arm_whip action and is not edited; only the whip responds. Tip speed 18.6 m/s is a stylised crack, not a physical Mach 1 value. Collision is point-vs-capsule (arms are not colliders).

## Other changes
- Intro (0:51): glowing MPO patch cord (4-point NURBS, hook to the plug's cable end) and an MPO plug at scale x8 approach a 4-port MPO adapter plate, touch at 0.85-0.95 s with the key rolled 0.28 rad and the plug yawed 0.2 rad (does not seat), then rattle until 1.55 s; stars burst at the jam. v1.0 LC patch cord/panel removed; fiber_cables_and_trays is kept hidden as the source for bundle_cords / cable_tie. Camera shots 2 re-aimed at the port.
- Manager visibility: Manager visible from 3.0 s; new cutaway 6.7-7.4 s shows him full body walking in with the whip trailing; wind-up shot widened (lens 22-24, camera y = -3.95; y below -4.05 is inside the row-a racks, an earlier try at -5.2 showed a rack door); whip arcs up to z about 3 m stay in frame.
- Shotgun: S6 has no shotgun in the storyboard (Manager owns the shotgun in S1-S5, the whip in S6); the whip uses the same convention as the shotgun in S1/S2/S4 (rotation z = pi on the hand grip hook), so the mount orientation is consistent. Nothing else to change.
- Glow: patch cord glow emission 2.2 -> 0.9, thread fibre 1.5 -> 0.8, crack / hit FX intensity 0.4-0.6 (v1.0 drew a white blob over the contact). Compositor bloom is applied by render_scene.py (cartoon preset); not changed.
- Camera: all moving shots use bezier ease; deterministic noise F-modifiers on camera and target location (handheld strength 0.012 m, scale 24 frames), restricted-range shake at the crack (0.05 m, 8 frames) and the hit (0.10 m, 14 frames); `render.motion_blur_shutter = 0.5` set (motion blur itself is on only in the hero preset).

## Verification
- Contact sheet `scenes/v1/s06_fiber_contact.png` (12 stills 4x3 at 540x674, t = 0.3 .. 9.5 s) and whip strip `scenes/v1/s06_fiber_whip_strip.png` (frames 232, 238, 243, 246, 250, 256) viewed.
- Cost at RENDER_PRESET=draft, 50 percent (540x675): typical 0.5-1.4 s per frame; dense whip/Gary frames 1.5-2.1 s (frame 232 2.1 s, frame 243 1.5 s); first frame of a process +5 s shader compile. Above the 1.5 s target on those frames (hall and Gary dominate; the whip bake adds only bone F-curves).
- Build time of the scene about 40 s (the whip bake is two passes of about 10 s).

## Remaining problems
- Intro macro is small in the first dolly frame (plate far), plug looks plain at draft resolution; MPO latch details are low (level C).
- Whip colours read mostly blue/dark under the hall lights; tip plugs are stubby because of the x3.5 plug scale.
- The contact stars/shockwave still read bright at draft in the bloom pass.
- Needed framework changes (recorded, not made): asm.show on collections should key objects (this script still monkeypatches asm.show); asm.shot could take a handheld option; an asm helper for NLA split/hit-stop would remove the local mgr_arm_strips().

## Progress log
- 2026-10-02: read briefs; extracted v1.0 stills; backed up v1.0 blend and script.
- 2026-10-02: npc_nvydia_leather built, previews and rig-deformation test (base actions) passed.
- 2026-10-02: mpo_connector_and_adapter built (4 ASSET collections), previews viewed.
- 2026-10-02: whip sim prototyped on dumped v1.0 inputs; contact required Manager 0.5 m closer; integrated, camera fixed (y limit), thickness x7, FX intensity lowered; final build and contact sheet.
