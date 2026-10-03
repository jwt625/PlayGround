# DevLog-003-scene-s03: S3 "NPO: closer, and everyone has a different idea" (film 20-30 s)

Date: 2026-10-02. Files: `scripts/film_v1/s03_npo.py`, `scenes/v1/s03_npo.blend` (83 MB), contact sheet `assets/generated_textures/v1/s03/s03_contact_sheet.png` (10 stills at 540x675: t = 0.5, 2.3, 3.6, 4.6, 6.4, 8.4, 8.6, 9.0, 9.3, 9.8 s).

Build: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s03_npo.py -- scenes/v1/s03_npo.blend` (about 20 s).

## TODO

- [x] Room, lights, cast, hardware, props, actions, FX, captions, cameras
- [x] Contact sheet viewed and fixed (camera framing, labels, fall directions, Manager entry, door, gun/IK)
- [x] Render cost measured
- [x] Finale reworked as quick cuts (coordinator decision, 2026-10-02), see below

## Layout (scene frame = world frame, 1 unit = 1 m)

- conference_room at the origin, storyboard table 5.6 x 2.2 m, top z = 0.76. Chairs hidden (far/near sets) because vendors stand and fall behind the table; head chair left.
- Vendors stand on the far (+Y) side at y = 1.4: TERAHOP -2.4, MOLEXX -1.4, NUBISS 1.4, AYARR 2.4 (pairs outward so the two victims fall past the table ends, where the camera can see them; bodies that fall behind the table are hidden by it from any reasonable camera, which is why).
- Gary at (3.1, -0.9) facing the camera (the +X table end, near corner). Manager enters through the door in the -X wall (door swings open outward at 6.2-6.7 s), walks from (-4.35,-1.9) to (-2.3,-1.8) at 1.9 m/s (gait slightly faster than its natural 1.17 m/s: feet slide a little).
- ASIC = packaging/xpu_package_rubin_style at (0.8, 0); NPO module = photonics/oe_module_npo (VARIANT_closed) slides from x = 2.55 to 1.85 at 0.2-1.0 s; external laser = photonics/els_laser_source VARIANT_elsfp_closed at (-0.55, 0.65), visible 1.9-3.0 s; "in-package laser" = a copper emissive block on the XPU (generic block, not an asset), visible 1.8-3.0 s.

## Scale factors (hardware vs 1.8 m people)

| Item | Factor | Resulting size | Why |
|---|---|---|---|
| XPU, NPO module, ELS (true relative sizes) | x7 | XPU 1.26 x 0.70 m, NPO module 0.34 x 0.21 m, ELSFP 0.16 x 0.71 m | readable on a 5.6 m table (crude: ASIC 1.0 m) |
| bga_lga_family packages (fcbga_1p0_35, lga_1p0_45, fbga_0p8_23) | x12 | 0.42, 0.54, 0.28 m | ball/land arrays readable from 2.4 m |
| loose dies (eic_die, pic_die, hbm_stack exposed 16-hi, generic logic die slab 0.6 x 0.6 m) | x50 in xy, z x150 (3x exaggeration on top) | 0.3 x 0.21 .. 0.6 x 0.6 m | macro gag; the z exaggeration keeps the tower from looking like sheets |
| calendar (wall_calendar) | x2.2 | 0.66 x 0.95 m | floats above the table, p_flip 0 -> 3 (pages flip, April shows) |
| price balloon, envelope, shotguns | x1 | real | balloon nominal radius 0.25 m inflates 0.2 -> 1.6x |

Relative sizes between factor groups are not consistent with each other (packages vs dies); they are separate gags.

## Timing (scene-local seconds, copied from the crude s3)

- 0.2-1.0 NPO slide; labels NPO / LASER / BGA vs LGA ... / DIE SIZE / 2D vs 3D ... / THE VENDORS LOSE IT, card (Cheng 2025), FX notes, narration chunks, "@#$%!" and "!!" balloons, "GARY'S GUN", BANG BANG 8.3-8.9, BANG 8.95-9.55, S3 timecode: all as in the crude.
- Vendors: idle -> shout_loop 1.8 -> shove 3.0 / 4.4 -> punch_loop 5.6-7.3 -> shout; p_expr_shouting from 2.0, p_expr_angry from 3.2; small lunge keys per burst. Dust clouds (fx_dust_cloud) at 3.2-6.4.
- Packages stand up (flip about X) at 3.0 / 3.1 / 3.2, die row 4.15-5.2, tower 5.2-7.0 with growing wobble, balloon 5.4-8.3, envelope in Gary's hand 6.7 -> released 7.51 -> flies -> in AYARR's hand 8.0-8.4.
- Shots: MOLEXX 8.3 (TERAHOP topples, p_hole_1 = 1), NUBISS 8.4 (AYARR topples), Manager shoots NUBISS and Gary shoots himself at 8.95 (p_hole_1 NUBISS = 1.0; p_head_hole_radius = 1.0). MOLEXX is the only vendor standing (smug at 9.0). Muzzle flash + smoke ring on each bang (fx_muzzle_flash, fx_smoke_ring on the shotgun's HOOK_muzzle_R, rotated -90 deg about X).
- Gary's earlier holes follow `r = max(0.3, 0.9 ** ((T_now - T_shot)/1.5))` with T_now = 20 + t: hole 1 (T_shot 9.2) 0.47 -> 0.3, hole 2 (T_shot 19.2) 0.95 -> 0.48, keyed every 0.5 s.

## Technique notes

- Guns for MOLEXX, NUBISS, Manager: props/shotgun (one master + linked clones) parented to the character's HOOK_gun_grip_R with a fixed rotation matrix (gun +Z -> hand +X, gun -Y -> hand +Y, gun +X -> hand +Z, derived from the aim_gun pose); aim_gun strips start so the recoil frame (36) lands on the bang time (vendors play it at 1.5x).
- Gary's self-shot: gun parented to HOOK_muzzle_self (bone-parented empty on the head), pointing at the head from his right (table side); a Copy Location constraint on pose bone `ik_hand_R` (influence keyed 0 -> 1 over 8.1-8.45) drags his hand to the gun's HOOK_foregrip_L. Result: arm extended to the fore-end, stock over the table; looks acceptable, not physically exact.
- Gary faces the camera (profile would put the fall into the TV wall); he turns to AYARR for the envelope (6.8-8.0) and back; topple_back at 8.95 (speed 2.0) lands him toward -Y, clear of AYARR and the wall.
- Victims keep the 3/4 fight stance so the chest hole faces the camera; NUBISS turns toward the Manager (yaw -0.6) before being hit.
- Envelope uses three linked clones (in Gary's hand, flight, in AYARR's hand) because parenting cannot be keyed.

## Framework problems found (asm.py not edited, worked around)

1. `asm.show()` keys `Collection.hide_render`, which Blender 4.2 cannot animate: `asm.finalize` raises `TypeError: property "hide_render" not animatable`. Workaround in `s03_npo.py`: monkeypatch `asm.show` with a version that keys the visible objects of the collection through `blender_lib.VA` (so `asm.fx` also works). Suggested framework fix: key object visibility instead of collection visibility in `asm._finalize_collections`.
2. `asm.play` creates a new NLA track per call; later calls sit above earlier ones, so a base idle must be played before `asm.walk`, not after.
3. `F(t)` uses banker's rounding: keys at 8.94 and 8.95 land on the same frame (the later key overwrites). Keep key times at least 1 frame apart.

## Render cost (EEVEE Next, standard preset, 24 samples, this machine, other jobs possibly running)

- 540x675: 1.8-2.5 s per frame typical (first frame of a process 6-10 s incl. shader compile).
- 1080x1350: 2.7-3.8 s per frame measured at t = 3.6, 5.0, 6.5, 8.5, 8.6, 9.4 s (first frame 10 s). Heaviest scene content: the three BGA/LGA packages (up to 416k tris each, visible only 3.0-4.2 s), 7 characters (about 50k tris each with modifiers), conference room 39k tris. No frame measured above 8 s.

## Asset problems / notes

- bga_lga_family: roots are placed 60 mm apart and not at the origin; root location had to be reset. Package z = 0 is the substrate underside plane; the standing pose uses a -90 deg X rotation (underside to -Y).
- els_laser_source and oe_module_npo contain all variants laid out side by side, so the unused variants must be hidden by collection (done) or the bounding box is wrong.
- Shotgun origin is the stock wrist (good for hands); the gun is long (1.15 m) so Gary's self-shot cannot be held realistically.
- Hole rim on npc shirts overlaps the shirt wordmark (TERAHOP chest hole cuts a letter), acceptable.
- The smoke ring effect is large and pure white against the walls; reads fine but dominates at close range.
- wall_calendar pages flip over the top; the page backs are blank white, so the "+3 MONTHS" read relies on the world label.

## Deviations from the storyboard / crude

- Vendor order along the table and positions changed (see Layout) so the falls are visible; MOLEXX and NUBISS shoot TERAHOP and AYARR as in the storyboard, and the shot spans do not cross other people.
- NUBISS and the other victims lie behind the table at the end and are mostly hidden from the front camera; only the table-end overhangs and Gary lying in front are visible.
- Final wide shot (8.3-10 s, lens 22 mm from y = -5.2) is wide because it has to hold Manager (-2.3), Gary (3.1) and all four vendors; people are about 15 percent of frame height. "Tight on all seven" is not achievable with a 5.6 m table in 4:5 without cropping someone.
- Manager enters 6.7-7.8 s (crude 7.2-7.8) because the real door is 2 m from his stopping point; he is visible from about 7.2 s.
- Calendar, price balloon and envelope are real props instead of boxes; "$$$" comes from the balloon asset (no separate label).

## Open questions

- Should the finale be cut into two shots (vendors / Gary) to read bodies and the head hole? Needs a storyboard decision.
- The compositor bloom for the HDR muzzle flash was not enabled in this scene (flash reads as a white-orange star without bloom).

## Progress log

- 2026-10-02 19:08 first full build (83 MB), asm.show workaround written.
- 2026-10-02 19:15 lighting reduced (6 ceiling areas 30 W, front fill 120 W), cameras moved closer to hardware.
- 2026-10-02 19:20 Gary self-shot reworked (facing camera, right-side gun, IK hand), victims' falls and positions reworked.
- 2026-10-02 19:26 Manager entry fixed (door outward, earlier walk), final contact sheet written.

## Finale rework (2026-10-02, coordinator request)

Cuts (same events and times, except NUBISS now fires at 8.5 s instead of 8.4 s so each shooter gets his own cut):
- 8.3-8.5 s medium shot from the left of the table end: MOLEXX shoots, TERAHOP falls out past the table end (open floor, not behind the table).
- 8.5-8.75 s medium shot from the front-right: NUBISS shoots, AYARR falls past the right table end.
- 8.75-9.15 s Manager (left foreground, gun) and NUBISS across the table; Manager muzzle flash at 8.95 s.
- 9.15-9.6 s close-up of Gary's head from the +X side, along the head-hole axis (lens 50 mm, camera 0.9 m away, tracking the head through five sub-shots): the hole is see-through at radius 1.0 and the gun barrel at the head hook shows through it. Gary's topple now starts at 9.35 s (speed 3.0, lands about 10.0 s) so the head holds still for the close-up.
- 9.6-10.0 s wide aftermath (22 mm): MOLEXX standing, Manager, Gary down.
- BANG BANG / BANG captions moved to the upper part of the frame (custom overlay kind BIGHI with the same style as BIG) so they do not cover the action.
- Smoke ring: scaled 0.6, radius end 0.13, travel 0.6, intensity 0.5, own material copy with grey base colour and half alpha. Dust clouds scaled 0.5.
- Bullet holes of TERAHOP, AYARR and NUBISS moved 0.14 m down (hole empty translated at frame 1; rim tube follows), so the hole no longer cuts a wordmark letter.
- asm.py fixes now in the framework (asm.show keys objects, F rounds half up): the local asm.show monkeypatch was removed.

Remaining: bodies of NUBISS (behind the table) and TERAHOP/AYARR are visible only in the cuts, not in the final wide shot; the rim tube of the head hole shows as a ring, the barrel seen through the hole is a silver disc.

## v1.1 feedback wave (2026-10-02, S3 agent)

Files: `scripts/film_v1/s03_npo.py` (assembly), new `s03_fight.py` (pose/bake engine), `s03_brawl.py` (choreography), `s03_fx.py` (dust, crumbs, papers, caps), `s03_cam.py` (camera), `s03_sky.py` (day/night + exterior). Build unchanged: `Blender -b --python scripts/film_v1/s03_npo.py -- scenes/v1/s03_npo.blend` (about 25 s, 86 MB). Previous blend: scratchpad backup only. Sheets: `assets/generated_textures/v1/s03/s03_contact_sheet_v1_1.png` (12 stills, t = 0.5 2.2 3.3 4.0 4.8 5.5 6.0 6.5 7.5 8.4 9.0 9.8), `s03_fight_strips_v1_1.png` (two 6-frame strips, 0.15 s spacing).

- [x] 0:20-0:25 fight rework. Poses authored per frame with the character IK pose engine (`chars_actions.Rig`, imported, assets untouched) and baked to new actions `ACT_<npc>_s03_brawl` (1.70-7.50 s, played as an NLA strip over the idle). Two pairs with different choreography: TERAHOP/MOLEXX (argue, chest bump, shove, collar grab and tug, hook exchange, uppercut, haymaker whiff + duck) and NUBISS/AYARR (argue, head-butt into belly, shove, belly flurry, open-hand slap, windmill flurry, hook, grab and shake). Anticipation (easeInBack segments), overshoot (easeOutBack), 3-4 frame hit-stop holds, squash/stretch (chest, head, hips bone scale, root scale on landing), belly/shirt jiggle (damped 6.5 Hz oscillation on spine_1/2 and chest scale after each impact), head/neck spring lag (6 / 7.5 Hz), body lean and crouch, procedural foot planting/stepping (feet stay world-fixed, step when the hips drift more than 0.2 m), eyes tracking the partner, face shock pulses on hits. Hands are IK-targeted to the partner's computed face/chest position. Registered impact events drive: dust puffs and clay crumbs (victim shirt/skin colours, ballistic with bounce, `s03_fx.py`), 4 paper bursts (drag + flutter, land on table or floor and stay), two flying caps (NUBISS green at 5.72 s, TERAHOP purple at 5.98 s; worn on HOOK_head_top, then ballistic with bounce), and camera shake. `fx_dust_cloud` assets are no longer used. Ties: none of the S3 cast has a tie (Manager is static), so the secondary motion is on shirts/bellies/heads.
- [x] 0:26 exterior time-lapse. Interpretation: no full cutaway (it would hide the die tower, calendar flip, balloon and punch exchange, all in 5.2-7.0). Instead the roof lifts off (5.24-5.50 s up 9 m, returns 6.72-6.98 s), blinds and sill are hidden, and for 5.4-7.0 s the world runs 3 full day cycles (closed form, keyed per frame): sun and moon discs (only visible in the window), dusk/dawn glow, stars, sky colour; sun and moon lamps rotate and light the room through the open wall; clay skyline, trees and ground outside the +Y wall. Exterior geometry and both lamps exist only during 5.2-7.0 s (render cost). The camera in that shot looks higher (target z 1.5) to show the sky. Open question: if a true exterior establishing cut is wanted, the calendar/tower gags need to move.
- [x] 0:28-0:29 stable camera: one position from 7.0 to 10.0 s (0, -5.7, 1.7) -> (0.05, -5.3, 1.65), lens 22, only a slow eased push-in plus handheld noise and impact shake; Manager, Gary and all four vendors are in frame (Gary and Manager at the edges). The close-up/quick cuts were removed; the head hole is small in this wide framing.
- [x] Camera: all shots eased (smoothstep), handheld drift (sum of sines, fixed phases), angular impact shake pulses (17 Hz, 0.11 s decay) at fight hits and the three shots, dense per-frame keys on CAM, CAM_TARGET, lens and caption holder (captions do not shake). Shots 3-5.2 s re-framed to keep the fights readable (upper bodies above the packages, both pairs in the 4.2-5.2 shot). Motion blur: `motion_blur_shutter` 0.5 set; blur stays off in draft/standard (hero preset enables it); every cut is between consecutive keys so blur will smear on cut frames (framework issue, see below).
- [x] Gun events, holes, narration, captions, source tags unchanged. Vendor asset ids are in `VEND_ASSET` (one-line swap per vendor). NVYDIA is not in the current S3 cast, so no leather-jacket swap is needed here.

Render cost (draft 8 spp, 50 percent, 540x675, this machine, includes compositor): 1.44 s per frame averaged over 12 frames spread across the scene (v1.0 blend on the same frames: 1.36 s); frames inside the 5.2-7.0 s time-lapse are about 1.9 s (extra shadow-casting suns, exterior). Measured with shadows-on lamps always present: 1.86 s per frame, hence the visibility windows.

Needed framework changes (not made): (1) `asm.shot` should support dense/eased shots with noise and shake (done locally in `s03_cam.py`); (2) NLA strip blend_out is set in the scene script, `asm.play` has no parameter; (3) a cut-aware motion blur (camera visibility between consecutive frames).

Known remaining problems: fighters interpenetrate slightly in clinches (IK reach, no collision); TERAHOP's cap clips the long hair slightly; one timeline nudge warning (TERAHOP 6.56 s, 1 frame); chest/hit timing was checked in stills only (no video review); at 4:5 the finale people are about 13 percent of frame height; the bottom of the finale frame shows a flat grey-blue ground colour; the dotted pattern inside the sun/ceiling-light glow comes from the compositor grain.
