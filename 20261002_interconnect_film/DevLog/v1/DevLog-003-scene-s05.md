# DevLog-003-scene-s05: scene S5 (film 40-50 s), CPO yield, back through the factory to wafer test

Date: 2026-10-02. Owner: S5 assembly agent. Contract: `scripts/film_v1/SCENE_BRIEF.md`.

## Files

- Build script: `scripts/film_v1/s05_wafer_test.py`
- Scene: `scenes/v1/s05_wafer_test.blend` (64 MB, 300 frames, 30 fps, 1080x1350, EEVEE Next `standard`)
- Textures: `assets/generated_textures/v1/s05/spec_s5/` (66 PNG, ring-spectrum sequence from `tex_gen.spectrum_sequence`), `assets/generated_textures/v1/s05/contact_sheet_s05.png` (8 stills at 540x675)
- Build: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s05_wafer_test.py -- scenes/v1/s05_wafer_test.blend` (about 15 s)

## Plan / checklist

- [x] fab-floor environment (floor, aisle lines, back wall, ceiling panels, lab partition) from boxes, `light_lab` rig x7 (scale 1.6, visible only in each zone's window; desk-lamp spot zeroed)
- [x] hero engine (oe_module_cpo closed) + hero die (pic_die with eic_die flipped on the PIC site hook), glow trail, 48 streak boxes
- [x] 5 camera shots through the line (crude timing), per-station reverse-order animation (oven board, FAU retract/dispense/UV, bonder head, saw spin and table, tape frame diced -> whole)
- [x] CM300-style station: p_drawer shuttle, 3 wafers (668 dies each, exactly 67 green = 1 in 10 per wafer), keyed stage hops, probe card taps
- [x] ring-spectrum sequence on `probe_station_cm300_style_screen` (16:9, 480x270 maps 1:1) and on the DCA scope in the lab
- [x] human scale: Gary + Manager at a lab bench with the DCA, printout "1 / 10", shotgun at HOOK_gun_grip_R, muzzle flash x2, smoke ring, BANG, hole 5, topple
- [x] captions, narration, labels, card, FX notes, timecode (copied from crude s5)
- [ ] not done: fresh-context audit; speed-ramp easing on the line shots (crude is linear too)

## Timeline (scene-local seconds)

| t | content |
|---|---|
| 0.0-0.3 | engine with die on its lid, locked shot |
| 0.3-3.3 | five tracking shots following the die (x 0 -> 8 -> 15 -> 22 -> 29 -> 40 m); stations at x = 8 (oven), 15 (FAU), 22 (bonder), 29 (saw), 40 (probe station); labels, streaks, glow trail; lab "BACK THROUGH THE LINE" |
| 3.3-4.8 | over the probe station; wafer period 0.5 s: fly in 0.10, drawer closes 0.10, 8 hops at 1 frame each, drawer opens, fly out; 3 wafers |
| 4.8-7.0 | close on the station monitor, spectrum sequence (66 frames, all 30 traces by 7.0, final count 3/30) |
| 7.0-8.4 | Gary and Manager (printout), big "1 IN 10" at 7.2 |
| 8.4-10.0 | aim_gun at 8.4 (speed 1.71 so the shot lands at 9.1), flash, ring, BANG, hole 5 = 1.0 at 9.1, topple_back at speed 2.2 |

## Scale factors and why

- Stations, wafers, screens, characters, props: real size, factor 1.
- Hero engine + die: x90 (die 7 x 9 mm becomes 0.63 x 0.81 m; engine carrier 24 x 20 mm becomes 2.2 x 1.8 m). At real size the 80 mm engine is invisible next to 2 m machines; crude had die 0.5 m next to 3 m ovens. The pigtail is therefore 5 m long (cosmetic).
- Probe card: x0.15 (VARIANT_light, 410 mm -> 62 mm). The real 410 mm card covers the whole 300 mm wafer in a view from above.
- Wafer lift: during probing the wafer is shown 75 mm higher than the real chuck (WAFER_LIFT). In the asset the chuck top is 43 mm below the platen top and 61 mm below the probe-ring top, so the stepped wafer is hidden under the deck from above. The lift keys 0 -> 0.075 while the drawer closes and back to 0 while it opens. Probe card tips are lifted by the same amount.
- Stage hops: 0.7 x die-group centroid (max +-100 mm in x, +-120 mm in y): a physical full-wafer probe needs +-150 mm and the wafer would leave the platen opening; the die colouring is therefore decoupled from the exact die under the probe (all 668 dies are coloured over the 8 hops in serpentine bands).
- Bonding close-up (eic_pic_bonding_close): x7 macro inset hovering right of the bonder (p_gap 0 -> 3, p_heat 1 -> 0 reverse order).

## Decisions

- Holes: step function, x0.9 every 1.5 s after the hit, floor 0.3; film times T_shot 9.2 / 19.2 / 28.95 / 38.9 / 49.1. Values at scene start: hole 1 and 2 = 0.3, head hole = 0.3, hole 4 = 1.0 (0 steps at 40.0 s), stepping x0.9 at scene times 0.4, 1.9, 3.4, 4.9, 6.4, 7.9, 9.4 s (0.478 at 10.0 s). Hole 5 = 0 until 9.1, then 1.0 (CONSTANT). The brief formula is continuous (0.93 at scene start for hole 4); the step form was used for continuity with the end of S4 (1.0).
- Staging: Manager at screen left facing +X (his right hand, which carries printout and shotgun, is on the camera side), Gary half-turned toward the camera (yaw -45 degrees) so that the bullet holes, whose axes are Gary's local Y, are visible as ellipses; the Manager shoots along +X. This mirrors the crude (Manager at right).
- Printout "1 / 10": the asset die map (90 dies, 9 green) plus a text object "1 / 10" under the map; it is held by HOOK_paper_R (rotation matrix chosen so the sheet faces -Y), drops at 8.4 (a linked copy falls to the floor) when the gun appears.
- Gun: rotation relative to HOOK_gun_grip_R so gun -Y (barrels) = hook +Y, gun +Z = hook +X (identity attach put the barrels backwards).
- Spectrum: `spectrum_sequence(n_traces=30, seed=5)` as crude but nframes = 66 (4.8-7.0) and hold_from = 0.9 so that all 30 traces (3 passes) are shown by 7.0 s; crude used 156 frames (to 10 s) and reached only ~18 traces by 7.0. The same sequence is plugged into the DCA scope screen in the lab.
- Monitor screen material: `MIX_use_image` keyed 0 (asset brick pattern) -> 1 at 4.8 s (CONSTANT); before 4.8 the sequence frames would show magenta.
- Caption plane: the overlay holder (OVERLAY_HOLDER) is pulled from 0.8 m to 0.32 m in front of the camera (scale x0.4, same angular size) so close shots (screen 0.8 m, station 1.8 m) do not cut the captions.
- Camera clip 0.1-300 m (dies sit 20 um above the wafer; the default 0.05 near clip z-fought).
- Die idle colour overridden to grey (0.50, 0.52, 0.56, glow 0); pass/fail/probing_glow states are the library states from `die_map_tools.STATES`.

## Performance (M-series, EEVEE Next `standard`, headless, 24 samples)

- 1080x1350: 1.65-2.55 s per frame over 6 sampled frames (t = 2.3, 3.6, 5.9, 7.6, 9.3 and the first frame 8.6 s with shader compile).
- 540x675: 0.7-1.7 s; the first frame of a session +4-5 s.
- Costs: the wafer frames (3 x 668 die objects, 449k tris each, shared mesh) and the lab shots with the two characters (about 1.7-2.5 s) are the slowest; line shots 1.1-1.7 s. Visibility is windowed per object, so only the active station, rig and wafer are drawn.

## Framework issues (not edited, worked around in the scene module)

1. `asm.show` / `_finalize_collections` fail in Blender 4.2.3: `Collection.hide_render` is not animatable (`TypeError: property "hide_render" not animatable`). Workaround in `s05_wafer_test.py`: `collection_windows_to_objects()` converts each collection window into per-object windows (`L._VIS`, intersected with object-level windows) before `asm.finalize`. Suggested framework fix: do this inside `asm._finalize_collections`.
2. `Asset.hook(name)` works with the `.001` suffix, but appending several assets that all carry `HOOK_wafer_slot` (probe station, dicing saw) gives suffixed names; always fetch hooks via the Asset, never by name from `bpy.data.objects`.

## Asset problems found

- `probe_station_cm300_style`: from above the wafer (chuck top z 1.017) is hidden by the deck (top 1.060) and the probe ring (1.078) except through the 0.38 m opening; real, but unhelpful for the die-map gag (see WAFER_LIFT). Monitor idle material shows a magenta missing-image look if `MIX_use_image` is 1 without frames.
- `probe_card`: solid 410 mm disc with no window; it cannot be seen through, so it hides a 300 mm wafer.
- `oe_module_cpo` closed variant: pigtail (62 mm to the MT face) becomes 5.6 m at the hero scale; die material is dark navy and reads weakly against the grey floor (the die is tilted toward the camera in flight to help).
- Gary hole 5 at radius 1.0 renders on the lying body as a small tan patch near the belt instead of a clear see-through hole (fine in the standing 3/4 view); the hole cutout is a straight cylinder along local Y, so a profile view hides it.
- `die_bonder_station`/`saw`: head motion and table travel are visible only briefly (shots are 0.5 s).

## Progress log

- 2026-10-02 19:00: read brief, assets, crude s5; layout and plan decided.
- 2026-10-02 19:08: first full build; fixed collection-visibility TypeError (see framework issues).
- 2026-10-02 19:20: die and wafer visibility fixed (hero scale 90, tilt, wafer lift, clip planes); gun and printout orientation derived from hook axes; staging turned 45 degrees for the holes; caption holder pulled in.
- 2026-10-02 19:28: contact sheet (8 stills) viewed and accepted; per-frame cost measured; devlog written.

## Open questions for Wentao

1. OK to show the wafer 75 mm above the chuck during probing, and the 0.15 sub-scale probe card (otherwise the die map is hidden)?
2. Hero die at x90 (0.63 x 0.81 m) acceptable, or push to the crude-like 1.4 m engine (x230 on the die)?
3. Hole decay: step form (used) or the continuous formula in the brief?
4. Keep Manager at screen left (right-hand props face the camera) or mirror to the crude layout and use the left hand?

## Audit corrections

None yet (no fresh-context audit run).

## v1.1 wave (DevLog-004 rows S5 t=1, 4, 8 plus known issues) - owner: S5 v1.1 agent, 2026-10-02

Previous blend backed up in the session scratchpad (s05_wafer_test_v1_0.blend). Build unchanged: `Blender -b --python scripts/film_v1/s05_wafer_test.py -- scenes/v1/s05_wafer_test.blend` (about 10 s, 63 MB).

Status (all done unless marked):
- [x] t=1 layout: stations side by side, fronts aligned at y about -0.75, gaps 0.35-0.5 m (oven x 0, FAU 4.0, bonder 5.55, saw 7.8, prober 9.6; engine start x -4.6). Tool visibility windows widened (neighbours are in frame).
- [x] t=1 camera: baked per frame (300 keys), smootherstep between stops (zero speed at every tool), dwell (s): oven 0.75-1.15, FAU 1.45-1.80, bonder 2.10-2.45, saw 2.75-3.05, prober 3.5-4.8; camera 1.5-1.8 m from the work point with a slow push-in; handheld NOISE modifiers on camera and target; scene motion blur on, shutter 0.5.
- [x] t=1 interior: oven cutaway (Boolean difference trench in the body front/top, zone 7-9 lids hidden, heater bars, board carriage and belt visible), bonder and saw open (glass variants hidden), FAU open, prober drawer open with wafer, probe card and optical test fibers, bonding close-up inset (x4).
- [x] hero die: x90 on the engine, eased to x50 during the line dwell (so it does not hide the tool interior), glow trail scaled down (it was a huge cyan ribbon), tilted to face the camera (v1.0 tilt sign made it edge-on).
- [x] t=4 screen fix: probe_station_cm300_style_screen UV has u along local +Z and v along local -X, so the image showed rotated 90 deg CCW. Fixed in the scene material with Separate XYZ / Math (1 - v) / Combine XYZ feeding IMG_screen (u' = 1 - v, v' = u). Verified on stills (t=5.4-7.0): dips point down, counter text upright. The DCA scope in the old lab is no longer used.
- [x] t=8: the prober (CM300-style) is the set of the human shot. Lab bench, scope and lab wall dropped. Gary works at the prober (thinking action, back to the camera, drawer open with a wafer), turns to the Manager at 8.25-8.85; Manager stands 2.1 m left, shotgun along +X. Gary falls toward +X/+Y, clear of the prober.
- [x] known issues: hole 5 (see below), world labels re-placed inside the frame during each dwell and window-limited, motion blur on, bloom: ceiling emission 6 -> 1.5, screen emission 1.2, fill light 250 W; streak boxes removed (real motion blur instead).
- [x] effects: 90 dust motes (seeded, linear drift), heat shimmer plane over the oven trench, sparks_burst at bonder contact and at the saw blade, cyan test fibers and pulsing probe-tip glow on every stage hop, wafer flight as ballistic arc with decaying spin (baked per frame), die-map hop colouring unchanged.
- [ ] not done: fresh-context audit; speed ramp on the die glow trail; cutaway lids do not animate open; the 7.0-7.9 camera pull-back passes close to the prober gantry.

Hole 5: the hole has a tan clay rim wall, so oblique views show a tan patch. Final camera now looks down onto the lying body (hole axis is vertical when he lies on his back) so the floor shows through; at the standing hit (9.1-9.35) the camera is a medium shot on the torso. The hole is only about 8 cm across, so it is small at draft 50 percent; BANG text covers the belly at 9.1-9.7 (caption timing unchanged).

Cost (draft 8 spp, 50 percent, compositor cartoon, M-series, per frame): line and prober shots 0.8-1.4 s, screen close-up 1.0-1.2 s, human shots with Gary and Manager 1.6-2.9 s (characters are the most expensive assets), first frame of a session 6-8 s (shader compile). With motion blur on the cost is within +-0.3 s of off. Over the 1.5 s target only in the 6.6-10 s human shots.

Framework notes (asm.py not edited): (1) `render_scene.py` with RENDER_PRESET=draft calls apply_render_preset which resets use_motion_blur to False; the coordinator must set `scn.render.use_motion_blur = True; scn.render.motion_blur_shutter = 0.5` after the preset (or add motion_blur to the draft preset). (2) asm.shot only keys two points per shot with hard cuts; S5 bakes its own per-frame eased camera (L.CAM, L.TGT, lens, caption holder scale). A framework `asm.shot_path(stops)` would help all scenes. (3) Collection.hide_render workaround unchanged.

Asset size notes left as is: hero die x90/x50, probe card x0.15, wafer lift 75 mm.

Progress log v1.1: 2026-10-02 21:15 layout, camera, cutaway, screen fix, prober cast, effects built; contact sheet `assets/generated_textures/v1/s05/contact_sheet_s05.png` (12 stills 540x675 at t = 0.23 ... 9.95 s) viewed and fixed (fill light, die tilt and size, labels, human shot framing, final camera).

## v1.2 pass (PHASE2_BRIEF, owner: S5 v1.2 agent, 2026-10-03)

Backup of the v1.1 blend, script and devlog: session scratchpad `p2_s05/backup/`. Build: `FILM_HUD=0 Blender -b --python scripts/film_v1/s05_wafer_test.py -- scenes/v1/s05_wafer_test.blend` (about 15 s). New helper `scripts/film_v1/s05_tex.py` (monitor sequence, numpy), new textures `assets/generated_textures/v1/s05/mon_s5/` (102 PNG). The old `spec_s5/` sequence is no longer referenced (left in place).

Plan / checklist:
- [x] v2 cast: `gary_v2`, `manager_v2`; motion_v2 actions (Gary: thinking, turn_left_180 with yaw hand-off, v1 hold_printout, shot_hit_fall in two strips: frames 0-4 hit-stop at speed 1, rest at 1.5x; Manager: gun_ready, gun_raise_aim_fire speed 1.25 with the shot frame 44 on 9.1 s). Face actions off, expressions keyed on the roots.
- [x] subtitles via `asm.narr_vo(5)`; HUD through `asm.fxn`/`asm.timecode` (built with FILM_HUD=0)
- [x] 43.3-44.7 wafer handling: wafers slide in over the deck from the front-right, sit on the (lifted) chuck, 8 stage hops, slide out front-left while the next comes in; drawer stays closed; positioners and microscope barrel hidden (they sit in the plane of the lifted wafer); probe card (now 0.22 scale) touches down for the hops and lifts 20 mm between wafers; camera front-above. Orange "fireball" was the red die map of the outgoing wafer flying at the lens with bloom: die glow lowered (pass 0.35, fail 0.12, probe flash 0.45) and the exit path moved sideways. Old probe-tip sphere (emission 6, linear scale ramps between wafers) replaced by a 5 mm flash on hop frames only.
- [x] 45-48 "one out of ten": own monitor sequence: 10 rings measured at 4.8 + 0.2 k s (on the trace_ping cues), dips point down, 10-die strip under the plot (9 red, 1 green), the passing trace drawn on top in thick green with a repeating ping ring on the green die; counter n/10.
- [x] 47.3-50 Gary breaks the news: Gary (right) turns to face the Manager/camera and holds up the "1 / 10" printout (sheet tracks the camera), two-shot with both at about 40-45 percent of frame height, feet and tiled floor in frame; Manager raises and fires at 9.1; hole 5 is a see-through disc at the hit-stop (Gary faces within about 13 degrees of the view); camera cranes to a near-overhead of Gary on the floor (hole axis vertical, floor visible through it), calm from 9.55.
- [x] look: tiled floor darker than the walls, walls 0.86 -> 0.70, ceiling emission 1.5 -> 0.8, oven heaters 3.0/2.5 -> 1.3/1.1, test fibres 4 -> 1.6, motes 90 -> 45 at 0.5, sparks intensity 0.8 -> 0.5, prober zone light rig moved to the front and p_energy 0.6, warm area key on the human set, printout paper albedo x0.8, aisle lines darker.
- [x] first 0.5 s calm (engine shot with slow push; the die is yanked out of the static frame at 0.3, the camera starts at 0.5), last 0.45 s slow push over the lying Gary.
- [x] draft p50 render (`outputs/v1/s05_wafer_test_v1_2_draft_p50.mp4`, 540x674, 300 frames) and frame check (frames at 0.2-9.98 s, strip 3.5-4.7 s, YAVG per frame: range 84-152, largest frame-to-frame jump 12.9 at 7.17 s during the pull-back, no flashes in the first or last 0.5 s).
- [ ] not done: fresh-context audit; Manager v2 aim pose reads hunched with the head down at eye level (motion_v2 gun_raise_aim_fire on the big v2 head); hole 5 at 1.0 radius is about 9 cm, small at p50 in the two-shot.

Progress log v1.2:
- 2026-10-03: read briefs; v1.1 draft frames inspected (fireball = outgoing wafer at the lens; floor indistinguishable from the wall; printout under "1 IN 10").
- 2026-10-03: script rewritten (v2 cast, wafer handling, monitor sequence, staging, cameras, look); four still rounds at 540x675 (contact sheets + strips over 3.3-4.4 and 8.3-10.0) viewed and fixed: die position, prober camera, monitor camera behind the gantry leg plane, pull-back path above Gary, Gary facing for the hole, overhead end shot, BANG label shortened to 0.2 s.
- 2026-10-03: draft p50 rendered twice (second after toning the white probe flash of the die map at 4.4-4.7 s). Cost: 332 s for 300 frames = 1.1 s/frame at draft 50 percent (M-series, includes compositor).

SFX retime: none. Cues kept: wafer_slide 43.3/43.8/44.3 (wafer arrives on the chuck at +0.12), probe_tick 43.47/43.97/44.47 (8 hops, one per frame), stage_clunk 43.737/44.237/44.737 (stage returns, wafer leaves), laser_zap 43.4 (fibres on), trace_ping 44.8 + 0.2 k (now exactly one trace per ping, 10 traces), shotgun/clay_thud 49.1, hole_pop 49.35. Notes for the SFX owner: (1) the passing ring is the ping at 45.8 (could get a distinct pitch); (2) the ding at 47.2 still matches "1 IN 10" (now 7.2-7.85); (3) Gary's ground contact is at about 49.63 (no thud cue there yet: candidate clay_thud 49.63); (4) whooshes 40.3 (die yank) and 40.95 camera arrival (camera now starts at 40.5, arrives at the oven 40.95).

Narration vs picture: lines fit; "Gary breaks the news" (47.3-48.9) runs while Gary turns and shows the printout.

Framework requests: (1) caption overlays get motion-blur ghosts on the first frame of a caption while the camera moves fast (seen at 9.2 s: visibility switches inside the shutter); keying overlay visibility at frame boundaries with shutter-safe offsets or disabling blur on overlay objects would fix it. (2) `asm.shot_path(stops)` with smootherstep (still baked locally). (3) The collection-window workaround stays (`collection_windows_to_objects`).
- 2026-10-03 09:17: rebuilt under the post-overlay framework (card/FX/TC now in `scenes/v1/s05_wafer_test.blend.overlays.json`, not rendered); final draft re-rendered via render_all/bslot: 224 s for 300 frames = 0.75 s/frame at draft 50 percent. Frames checked at 0.3, 3.9, 5.9, 6.6, 8.5, 9.17, 9.6 s.
