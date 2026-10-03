# DevLog-003 scene S2 (Retimers: inside the rack, row after row)

Film time 10-20 s, scene-local 0-10 s, 300 frames, 30 fps, 1080x1350. Date 2026-10-02.

Build: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s02_retimers.py -- scenes/v1/s02_retimers.blend`
Output: `scenes/v1/s02_retimers.blend` (24 MB). Textures (tex_gen, same parameters/seeds as crude s2): `assets/generated_textures/v1/s02/eye_s2` (300 frames, BER 2.4e-2 -> 5.4e-9, min 6.7e-11) and `graph_s2` (300 frames). Contact sheet: `assets/generated_textures/v1/s02/contact_sheet_540x675.png` (frames 13, 37, 79, 139, 178, 223, 268, 292).

## Checklist
- [x] Flash cut 0-0.8 s: bench scope with closing eye + BER, Gary beside the bench, Manager visible behind
- [x] Rack interior (rack_interior_tray_stack, stylised), 68 retimer chips (1 row trays 0-1, 2 rows trays 2-4, 3 rows trays 5-7), crude timing (first row 0.9 s, rows 0.45 s apart, 0.1 s between chips, slide 0.5 s, fade 0.45 s, p_glow flash)
- [x] Camera rise 1.6-5.4 s, crane 5.4-6.4 s, wide 6.4-8.6 s, turn/aim 8.6-9.2 s, BANG 9.2-10 s
- [x] NARROWCOM sign (chip valley mark + wordmark meshes scaled x100) on the backplane
- [x] Eye sequence on bench_oscilloscope, graph sequence on whiteboard_big, ENERGY / BIT and LATENCY labels riding the curve tips
- [x] Manager shotgun at HOOK_gun_grip_R, muzzle_flash x2 (both barrels) + smoke_ring at 9.2 s, caption BANG, Gary hole 2 + topple_back
- [x] Captions, narration, labels, cards, FX notes copied from crude s2, timecode S2
- [ ] Fresh-context audit not run

## Decisions
- Layout (1 unit = 1 m): rack stage at x = -6 (no scale change except the chips); lab around x = 0-4: bench (2.7, -0.8), whiteboard bottom-centre (1.4, 1.5) at 0.9 m height, Gary (1.3, -1.5), Manager (-0.4, 0.8) beside the board's left end (so the bench is not on the line of fire and Gary does not hide him). Floor: 40 m checker plane (1 m tiles x2). Lighting: `daylight` rig (scale 1.6) plus a travelling area light inside the rack (energy only 0.8-5.6 s).
- Chip scale: x4 (rack JSON p_detail_scale; chip 15 mm -> 60 mm, silkscreen footprint 88 x 112 mm). Chips placed at HOOK_retimer_row<r>_tray<j>_<k> (68 of the 96 hooks used, as the storyboard rows rule needs).
- Chip instancing: BASE parts of narrowcom_retimer_chip (substrate, mold, marking, pin-1 marks) joined into one mesh; every chip is root + linked-data body + glow quad (3 objects). Body materials get an Object Info alpha mix (DITHERED) for the 0.45 s fade-in (object colour alpha keyed). Glow quad material: driver on the Mix factor removed (it pointed at the first chip root); reads `p_glow` through an Attribute node on the glow quad, which has a driver from the chip root's `p_glow`. p_glow keyed 0 -> 1 (peak, tk+0.2) -> 0.55 (settled orange, tk+1.1); colour white-hot above 0.55; round falloff (gradient) added so the 18 mm quad is not a hard white square; quad scale animated 1 -> 3.6 -> 2.6.
- Rows appear on the crude schedule: first row of tray j at t_arrive(j) - 1.2 s (tray 0: 0.9 s). Because the real trays have pans and a 0.55 m gap, the rising camera sees one tray slot at a time (about 0.4 s per tray); the later rows are therefore mostly done or flashing when their tray passes the camera, which still reads (orange glow on the passing tray).
- NARROWCOM sign: placed in the tray-0 gap (z = tray0 + 0.40), not at HOOK_nvl_sign (2.025 m: under tray 3's pan, only partly visible and not in the first shot). Visible 0.8-3.0 s as in the crude.
- Gary expressions (not in the crude): worried until 6.4 s, relieved 7.4-8.5 s, scared at 8.85 s; turns to the Manager 8.5-8.85 s. Manager: thinking pose, p_anger/p_flush up 8.0-9.0 s, turns 8.45-9.0 s, aim_gun at 8.6 s at speed 2.0 (shot frame 36 lands at 9.2 s), shotgun shown from 8.55 s. Gun attach rotation (0, 0, pi) about the hook (barrels along the fingers); verified in stills, the two-handed aim pose of the action is a stiff extended pose.
- Holes: hole 1 (shot at film T = 9.2 s in S1) uses the brief's schedule r = max(0.3, 0.9 ** n) in steps of n = floor((T - 9.2) / 1.5): 1.0 at S2 t = 0, then 0.9, 0.81, 0.729, 0.656, 0.59, 0.531, 0.478 at t = 0.7, 2.2, 3.7, 5.2, 6.7, 8.2, 9.7. Hole 2 appears at 9.2 s with radius 1.0. Note: the brief's parenthetical "0.9^(10/1.5) ~ 0.5" at S2 start disagrees with its own formula (T_shot = 9.2 gives 1.0 / 0.945 at S2 start); the formula was followed.
- Gary topple_back played at speed 2.4 so he is down by about 9.9 s (the action is 2 s long at speed 1).

## Framework issues found (asm.py not edited)
1. `asm.show` / `asm._finalize_collections` keys `Collection.hide_render` / `hide_viewport`, which are not animatable in Blender 4.2 (TypeError at finalize). Workaround in `s02_retimers.py`: `show()` keys every object of the collection through `blender_lib.V`; `fx()` copies `asm.fx` with that `show`. Needed fix in asm.py: key object-level visibility (or layer-collection exclude via a driver) instead.
2. `asm.fx` uses `asm.show`, so it fails for the same reason (own copy used).
3. `asm.append` of the same asset twice duplicates its materials (.001); code looks up objects by substring for that reason.

## Asset observations
- rack_interior_tray_stack: each tray has a dark, thick-looking block at the front centre of the board (part of `trayN_components`); it dominates the lower foreground in the rise shots. HOOK_nvl_sign is under tray 3's pan (see above). Trays are stacked 0.55 m apart with pans, so only one tray is visible per camera height; the near-camera pan fronts sweep through the frame as white bands while the camera rises at y = -0.8 m (0.12 m from the pan fronts).
- narrowcom_retimer_chip: the glow quad (MAT_packaging_chip_glow) has its mix driver bound to one root object (not reusable on clones without rewiring), and the settled glow is a flat 18 mm square (replaced here by an Attribute-driven soft glow).
- Crude label overlap: FX note (2 lines when longer than 48 characters) overlaps the LAB line at 0.8-1.6 s; this is the crude layout, kept.

## Performance (EEVEE Next standard, 24 samples, 1080x1350, measured full-size, warm process)
- Rack shots: 1.7-2.1 s/frame; lab/room shots: 0.9-1.0 s/frame; first frame of a new process 6-16 s (shader compile, one-off).
- Chips are hidden until they appear (object visibility windows); rack hidden outside 0.8-5.9 s, lab hidden while the camera is in the rack.

## Open questions
- Compositor bloom (NG_comp_post) is not set up in this scene; the chip flash would gain from `hero_glow`/`cartoon` bloom. Left to the coordinator's render pass.
- Third row timing: kept the crude formula; the camera sees the later rows only in passing.

## Progress log
- 2026-10-02: scene assembled, 5 test sheets viewed and fixed (layout, camera, gun attach, glow, topple speed); final blend written and a contact sheet of 8 stills viewed.

---

# v1.1 changes (2026-10-02, feedback wave DevLog-004)

Build unchanged: `Blender -b --python scripts/film_v1/s02_retimers.py -- scenes/v1/s02_retimers.blend` (4-6 s; textures are reused). v1.0 blend backed up in the session scratchpad (not in the repo). Contact sheet: `assets/generated_textures/v1/s02/contact_sheet_v1_1_12stills.png` (frames 8, 33, 61, 97, 130, 172, 191, 226, 241, 270, 281, 292; v1.0 sheet kept). New assets: `DevLog/v1/DevLog-004-s02-new-assets.md`.

## Feedback items
- t=2 (0:12) more varied components, two retimer columns per connector, cartridges visible: DONE. Each of the 8 trays gets one of 3 dressing layouts (gb300_compute, gb300_compute, nvlink_switch, gb300_compute, helios_compute, nvlink_switch, gb300_compute, helios_compute) at detail scale 4, plus a cartridge band in every tray gap (NARROWCOM sign scaled 100 -> 70 so the cartridges beside it show). Each connector has 2 columns (x +-0.12 m) per row, chip scale 4 -> 9 (60 -> 135 mm), 136 chips (was 68). Old per-connector silkscreen footprints removed in the scene (they no longer match).
- t=6 (0:16): DONE. `rack_nvl72_style` (open state) at (4.85, -0.8) next to the bench, three black cables from rack tray-front hooks (OU14, OU15, OU23) to the scope hooks (CH1 BNC, probe tip, trigger knob) drooping to near the floor, with a damped-spring sag settle at 5.4-6.4 s. Shot 5.4-6.4 re-aimed at bench + rack + cables. Rack and cables visible 0-0.8 and 5.3-10.
- t=8 (0:18): DONE. Push-in 6.4-8.6 s from (0.4,-4.4,2.2) lens 26 to (-0.45,-2.7,1.75) lens 30 (more zoomed than t=6); the Manager turns to the camera at 7.5-8.1 s, p_anger 7.6-8.2, p_flush to 8.4, anger tremble 8.15-8.6, ear steam (fx ear_steam on his HOOK_steam hooks) 8.1-9.2, then turns to Gary 8.5-9.0 and shoots at 9.2. Gary at the right edge is partly in frame at t=8 (he stands at 1.3,-1.5); left as is.
- v1.0 issues: reveal timing FIXED. The camera now dwells at each tray (arrivals T = 0.8, 2.2, 2.6, 3.0, 3.4, 3.85, 4.35, 4.85 s; 0.2 s smoother-step hops; dwell height 0.55 m, view 22-27 deg down) and rows reveal at T-0.22 + 0.08 r (slide 0.22 s), so each row lands as the camera arrives; tray 0 dwells 1.3 s (t=2 shot). Chips 4x -> 9x. Third row verified visible on trays 5-7 (stills f130, f136, f150 of the checking runs).
- Physics/effects: eased baked camera (smoother-step), handheld noise (0.25-0.3 mrad sum of sines on the aim point), BANG shake (two decaying shakes at 9.2 and 9.3 s), 3-frame hit-stop (Gary's topple starts at 9.3 s), clay dust cloud + impact stars at the hit point, Manager recoil spring, chip landing overshoot + squash (damped spring), glow flash plus per-bead glow bumps, signal beads (glowing ellipsoids) running connector -> rows (subtle: they merge into the halos), one sparks_burst per tray, cable sag spring. Motion blur: `motion_blur_shutter = 0.5` set; the preset enables it only for hero. All baked (per-frame keys, no solvers), deterministic.

## Camera rig (workaround for asm.py, which was not edited)
`CamRig` bakes one key per frame: camera position, clean aim target, lens, overlay scale, and a noisy aim target that the camera's TRACK_TO follows; the caption holder is re-parented to a clean follower (`CAM_CLEAN`: copy-location of CAM + track to the clean target) so overlays do not shake. Needed asm.py changes (additive): shot() with ease/noise/shake options and the clean-holder rig; asm.show must key object visibility (existing issue 1 below).

## Physics checks and cost (RENDER_PRESET=draft, 540x675, warm process)
- Rack frames 0.45-1.1 s, lab frames 0.45-0.55 s, worst measured 1.8 s (one early frame during a hop); first frame of a process about 5 s (shader compile). 136 chips + 8 dressings + 8 bands + 8 fx + 100 beads.
- Blend 44 MB.

## Remaining problems
- Fan modules (squashed to 60 percent) still dominate the foreground during hops between trays; tall parts in front of rows can occlude them at steep hops.
- Signal beads are hard to separate from the chip halos; chip halos are bright at the flash.
- Hit-stop is a delayed fall plus shake, not a freeze of both characters (the idle strip keeps running).
- Gary is partly in the right edge of the t=8 frame; the lab floor reads blown-out white at the far horizon (unchanged).
- No fresh-context audit. Compositor bloom still not set up in this blend (render_scene.py applies NG_comp_post).
- Known framework issues (unchanged): asm.show/asm.fx on Collection.hide_render (own show/fx copies), duplicate materials on repeated append.
