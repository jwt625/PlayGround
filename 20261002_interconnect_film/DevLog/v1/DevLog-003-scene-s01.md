# DevLog-003-scene-s01: S1 "Copper: the stretch" (film 0-10 s) from the v1 asset library

Date: 2026-10-02. Owner: scene S1 agent.

## Files
- Build: `scripts/film_v1/s01_copper.py` -> `scenes/v1/s01_copper.blend` (33 MB)
- Textures: `assets/generated_textures/v1/s01/eye_s1/` (300 PNG, 384x240, `T.eye_sequence`, seed 11, same ramps as crude s1), `contact_sheet.png` (8 stills, 540x675 each)
- Build command: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s01_copper.py -- scenes/v1/s01_copper.blend [--no-eye]`
- Render: scene is 300 frames, 30 fps, 1080x1350, EEVEE Next `standard` preset (24 samples); coordinator renders.

## Checklist
- [x] data hall shell (RACKS / CONTAINMENT / OVERHEAD collections hidden), lighting rig + two added fill lights
- [x] cartridge wall: 9 linked copies of VARIANT_wall_12x5 (17.3 m wide, 4.1 m tall) + 5 copper conduit rows in front
- [x] Pulse (clay sphere + eyes, MAT_vfx_clay_shirt_yellow / eye_white / pupil_black): hook run on wall conduit, 3 runs along the bundle (Follow Path on bundle cable 00), scale exp(-0.32 L): 5 m 0.20, 2 m 0.53, 1 m 0.73
- [x] rack_pair_for_cable_gag: p_gap constant keys 5 (2.2 s), 2 (3.0), 1 (3.6), then 4.4-6.0 s linear pull 1 -> 2 m; rack pair root x keyed with p_gap so rack A stays fixed
- [x] bundle_14 end hooks driven (world-space drivers) by HOOK_port_A / HOOK_port_B; p_sag_m 0.25 / 0.16 -> 0 (taut) at 6.0
- [x] Gary: idle, pull_cable 4.2-6.0 (hands on rack B side), idle, walks 1.6 m clear at 6.2, topple_back at 9.2; p_hole_1_radius 0 -> 1.0 at 9.2 (constant); sweating / dread / scared expressions
- [x] lab_bench + bench_oscilloscope at HOOK_scope_slot, eye_s1 sequence in SCREEN_IMAGE (SCREEN_FAC 1)
- [x] Manager: walk-in from 5.4 s (arrives 7.3), p_anger 7.0-8.2, p_flush 7.2-8.2, ear_steam from 7.5 s at HOOK_steam_L/R, aim_gun started so the shot frame (36) lands at 9.2 s, props/shotgun at HOOK_gun_grip_R, muzzle_flash + smoke_ring at both HOOK_muzzle_L/R
- [x] captions, narration chunks, big words, labels, cards, FX notes, world labels, timecode (asm.timecode(1)) copied from crude s1
- [x] 9 camera shots (asm.shot), crude sequence re-staged at real size
- [x] contact sheet viewed and fixed (exposure, gun orientation, Gary toppling into rack B, final wide framing)

## Layout (world frame, metres)
x along the hall, +y away from the camera, cameras near y = -4..0. Action line y = 2.7: rack A fixed at x = 0, rack B at x = gap (5 / 2 / 1 / 2 m); Gary at rack B + 0.8; bench centre x = 8.0; Manager stands at x = 10.0 (enters from 12.2). Cartridge wall front plane y = 4.0 (hall wall at 4.25).

## Scale factors and deviations
- Everything at real size (1 unit = 1 m), Gary 1.75 m, Manager 1.85 m. No hardware scale-up.
- Hall: root scaled z x1.28 (ceiling 3.6 -> 4.6 m) because the 4.1 m tall wall asset does not fit under a 3.6 m ceiling; floor tiles and signage are stretched 1.28 in z (barely visible).
- Hall racks, containment and overhead trays are hidden (they fill the 8.5 m wide hall and leave no room for the 5 m gap staging); the hall is used as an empty bay (floor, walls, ceiling, signage).
- Wall: 9 modules (17.3 m) instead of 1 so the fly-along has a wall to fly along (asset module is 1.92 m wide). Five copper conduit rows in front of the wall are a crude-style addition (not in the asset) so Pulse has a cable to sprint on.
- Rack pair gap is centre-to-centre per asset (p_gap); port-to-port bundle span is gap - 0.66 m (4.34 / 1.34 / 0.34 m).
- Stretch is a continuous pull 4.4-6.0 (crude did a linear move 4.2-6.0) rather than a constant jump at 6.0; p_gap lands on 2.0 at 6.0.
- Gary walks 1.6 m away from the racks after the pull (not in crude) so topple_back does not intersect rack B; Gary faces the Manager at the bang (hit on the chest, hole visible).
- Manager walks at the stored walk speed (1.17 m/s), crude ran at about 3.5 m/s; walk starts earlier (5.4 s, off camera).
- Manager holds the shotgun from the start (hanging, barrel pointing down-back while walking: grip orientation solved for the aim pose only).
- Camera distances are shorter and lenses wider (16-28 mm) than crude because the hall is only 8.5 m deep; the 5 m gap shot uses 21 mm at 6.6 m.
- Scope: the real bench scope screen is 0.29 m wide, so the 6.0-6.8 close-up camera is 0.42-0.6 m from the screen.
- Added lights: two area lights (400 W / 260 W) in front of the action; the data_hall rig is rotated along the hall, scale 2.2, strips lowered to 4.2 m.

## Asset / framework problems found
1. `asm.show()` fails with real collections: `Collection.hide_render` is not animatable in Blender 4.2 (TypeError in `_finalize_collections`; any `asm.fx()` call triggers it). Workaround in `s01_copper.py`: `asm.show` is monkeypatched to key `hide_render` on every object of the collection via `blender_lib._VIS`. Needed change in asm.py: do that (per-object windows) instead of keying the collection.
2. nvl72 bundle hooks: the drivers read the hook LOCAL location, so parenting HOOK_bundle_a/b to the rack port hooks (as the brief suggests) would collapse the bundle to the origin. Used world-space transform drivers from HOOK_port_A/B to the bundle hook location instead (works with p_gap keyed; rotation ignored as documented).
3. nvl72 bundle: 14 cables converge to one point at each port plate centre; they do not fan out to the 14 glands of the port plates. Cables are 7.5 mm thin so the bundle reads as a single thin wire in wide shots.
4. nvl72 wall variant is 1.92 x 4.1 m (tall, narrow cartridges), taller than the data hall ceiling (3.6 m).
5. rack_pair_for_cable_gag: at p_gap = 1 the racks (0.6 m wide) leave only a 0.4 m air gap; the port plate protrudes 0.03 m beyond the rack face. Rack pair is dark (black carcasses) against the dark wall; needs front light to read.
6. datahall_environment: only 8.5 m deep, racks fill the middle (y +-1.67); no free staging area of 5 m x 8 m. LED strips are emissive white bars that overexpose in wide shots.
7. aim_gun: hands only 0.26 m apart at the shot frame vs 0.37 m grip-to-foregrip on the shotgun; fine visually but not exact.
8. ear_steam puffs are large blobs (about 0.3 m) at human scale; ok for the gag.
9. Custom-property drivers (p_gap, p_sag_m) need the pose evaluated through `view_layer.update()` (done in the script).

## Render cost (M-series Mac, Blender 4.2.3 EEVEE Next, standard preset)
- 540x675: 1.2-3 s per frame; full 1080x1350: 3-5 s per frame (frame 100: 4.4 s warm). First frame +6-9 s shader compile.
- Breakdown at frame 100 full res: floor + underfloor collections about 2.9 s of 4.4 s; the 8 wall copies about 1 s. Hiding the underfloor/floor tile detail (or a lower-poly floor) is the biggest saving.
- Blend 33 MB; eye sequence 15 MB (300 PNG).

## Progress log
- 2026-10-02: read brief, storyboard, assets; probed hooks, drivers, actions; first full build and 9-frame check.
- 2026-10-02: fixes: Collection.hide_render workaround, gun orientation from aim pose hooks, fill lights (3500 W -> 400 W after overexposure), camera restaging, Gary clearance and turn, final wide framing.
- 2026-10-02: contact sheet `assets/generated_textures/v1/s01/contact_sheet.png` written; devlog closed.

## Open questions for Wentao
1. Is hiding the hall racks acceptable (empty bay), or should a rack row be kept behind the wall?
2. Should Gary walk away after the pull (added) or stay at the rack and the topple be redirected?
3. Wall 4.1 m tall vs 3.6 m hall ceiling: keep the ceiling stretch (x1.28 in z) or drop the top cartridge row (4 rows = 3.3 m)?

## Audit corrections
(no fresh-context audit run)


## v1.1 pass (2026-10-02, DevLog-004 feedback wave)
Files added: `scripts/film_v1/s01_cam_fx.py` (camera director, electron pockets, clay crumbs, spring helper). Build/run unchanged: `Blender -b --python scripts/film_v1/s01_copper.py -- scenes/v1/s01_copper.blend --no-eye`. Backup of the v1.0 blend: scratchpad `bak/s01_copper.blend`. Blend 40 MB.

### Feedback items
- t=1 opening (0-2.2): camera starts inside cable 00 of the bundle (glowing copper ring tunnel, back-face shader) and does a log-eased pull-back out through the jacket (cable 00 jacket translucent until 2.0-2.3 s) to the two-rack framing that the 2.2 s shot continues from. 22 electron pockets (cyan right-moving, amber left-moving, 4 emissive blobs each) baked per frame along the evaluated cable 00 centreline, closed form, seed 3; blob scale grows with camera distance (stylised, capped x20) and fades out by 2.2 s. Racks and bundle now visible from t=0.
- t=7 (6.8-8.5): Manager now walks all the way to the bench (starts 4.1 s from x=11.4, arrives about 6.6 s at x=8.45 beside the scope), turns to the scope, turns angry 7.0-8.2, then turns to Gary. Camera 6.8-8.5 (lens 36-42) keeps the scope screen and the Manager together; traces visible (checked frames 214, 235, 250).
- t=10 (9.2-10.0): wide hold on Gary + scope + Manager (lens 24), then from 9.45 a fast smootherstep dolly/zoom to the scope screen (lens 24 -> 34); the aim point is the true screen mesh centre (HOOK_screen_center is about 8 cm off the mesh centre).
- 14-cable bundle: drivers rewritten (extra driver var `f` = root `p_fan`, 9 slack -> 5 taut; ends x2.6) so the cables fan out between the ports, per-cable sag variation, cable radius x2 (visual exaggeration, asset 3.75 mm), three jacket tones, end blocks scaled x2.6.
- Scope at frame edge 6.8-8.5: fixed by the new reading shot. Shotgun mount: now `asm.attach(gun.root, HOOK_gun_grip_R, rot=(0,0,pi))` like S2/S4.
- Physics/effects: all camera moves eased (smoothstep/smootherstep, per-frame baked) with handheld noise (4 sines per channel, seed 7); motion blur flag set in the blend (shutter 0.5; render_scene.py re-applies the preset so it is on only for `hero`); hit-stop 3 frames at the bang (aim_gun split into two NLA strips, Gary's topple delayed 3 frames); impact camera shake (11-17 Hz, tau 0.17 s) + lens punch; Manager body recoil pitch spring; shotgun recoil (slide back + muzzle flip, damped); Gary squash/stretch spring; 16 ballistic clay crumbs (bounce, seed 5) + dust_cloud + impact_stars fx at the chest; Pulse hop with squash/stretch (hook run) and wobble (bundle runs); taut-cable twang (p_sag_m spring after 6.0 s).

### Measured cost (draft preset, 50 percent, warm, M-series)
0.75-1.6 s per frame on 12 sampled frames (frame 277 at the bang 1.4-2.1 s: fx); first frame of a process +5 s shader compile.

### Remaining problems / notes
- Tunnel part of the opening (0-0.7 s) reads as a copper ring tunnel with a glowing cluster at the vanishing point; pockets passing the camera are mostly sub-frame. The pockets read best 0.8-2.2 s from outside the cable.
- Hit-stop only freezes the Manager aim strip; Gary's idle loop and the particles continue (3 frames).
- Gun and the Manager's gun-arm cross the scope in the 8.2-9.0 s frames (gun hangs low then is raised); aim close-up 8.5-9.2 is side-on.
- Needed asm.py changes (not made): per-object `show` windows; an eased/noisy `shot()` variant (s01_cam_fx.Director does this locally); `asm.fx` clip of effect scale.
- Not run: fresh-context audit.

## v1.2 pass (2026-10-03, PHASE2_BRIEF)
Backup of the v1.1 blend, scripts and devlog: scratchpad `p2_s01/bak/`.

### Plan
- Characters: `gary_v2`, `manager_v2`; motion_v2 strips (idle_breathe, turn_right_90, pull_cable, run, startle, shot_hit_fall; stomp_walk, manager_slow_burn slice, gun_raise_aim_fire split for a 3-frame hit-stop). Roots baked per frame (`FX.bake_obj`) with CONSTANT keys before jumps.
- Stretch 4.4-6.0: three heave cycles (pull_cable at speed 1.875, 2.5 cycles); rack B follows Gary's hands during each heave (handle on rack B's outer face), Gary scoots back with a small hop during each reach; gap 1.0 -> 2.04 m. Popped copper strands + sparks at the strand_pop cues 4.95/5.35/5.65/5.95 with small camera jolts; bundle tremble. Camera 3/4 front on gap + rack B + Gary (Gary about 45-50 percent of frame height). STRETCH caption moved to the top band, 4.4-5.2.
- 6.0-6.4 scope close-up; 6.4-7.95 Manager stomp_walk into frame, stops right of the scope (x 9.1), slow burn facing the screen (profile, traces in frame), steam, red face; 7.95-8.45 camera eases back to a two-shot Gary | scope | Manager; shotgun pops into his hands at 8.07 ("Shotgun time"), raise, aim with tremor, bang at 9.2 (unchanged). Manager stands 1.46 m right of the screen so the raised gun stays right of / above the scope (fixes the v1.1 gun-crosses-scope).
- Gary runs to the scope after the stretch (6.1-7.8), stands 3/4 to camera so hole 1 on the chest reads; startle at the gun; shot_hit_fall (speed 1.35) at 9.2; hole pops open at the hole_pop cue 9.45; dust exits his back (hole not covered).
- 9.2-9.55 two-shot hold (hit, fling, tip), 9.55-10.0 eased-out push onto the scope screen (centred, slow at the end, handheld and shake faded to 0).
- narr_vo(1); HUD gated; one LAB/CARD per shot (LAB lines dropped, cards kept); world label PULSE DECAYS moved into the safe area.
- Look: LED strip emission 12 -> 3, bench light panel 6 -> 2, warmer fills; animated motion-blur shutter 0 on hard-cut frames (removes the 3.6 s ghost frame).

### Progress
- 2026-10-03: helper module extended (`s01_cam_fx.py`: multi-hit shakes, per-time handheld multiplier, `bake_obj` per-frame roots with CONSTANT keys before jumps, `shutter_cuts`, `add_strand_pop`, camera keys CONSTANT before cuts).
- 2026-10-03: scene script switched to v2 characters and motion_v2; built (FILM_HUD=0), three still passes checked. Fixes from the stills: stretch camera pulled back and near-static so rack B visibly slides away; Manager reading yaw changed to face -x (3/4 face to camera, screen 25 deg to his right; facing the screen showed only the back of his head); two-shot re-centred (x 7.8); smoke ring 0.35 s (it drifted across the lens in the push); dust_cloud star puff at the hit removed (stars covered the hole); hole 1 too small to read at the two-shot distance (about 14 px at 1080): radius pops 1.0 -> 1.6 -> 1.9 (hole_pop cue 9.45) -> 1.0 by 9.75 plus a pale disc at the hole centre (seen only through the cut) for contrast.
- 2026-10-03: draft p50 render started.
- 2026-10-03: first draft p50 (516 s, machine shared) reviewed from mp4 frames: stretch, Manager, bang and push strips; cut frames 2.97-3.03, 3.53-3.63, 5.97-6.03, 6.37-6.43 clean (no ghost); YAVG per frame: no 1-frame flashes outside the opening tunnel exit (0.70-0.83, unchanged, liked). Fixes: PULSE DECAYS label 3.0-4.2 only (hidden under "5 m" in the 5 m shot); muzzle flash started 1 frame early so it shows ON the bang frame 9.2 (it grows from 0); smoke ring start 9.27 (its show padding put a white blob on 9.167).
- 2026-10-03: framework updates applied: all Blender runs through `scripts/film_v1/bslot.sh`; FILM_HUD=0 build writes cards/FX/TC to `scenes/v1/s01_copper.blend.overlays.json` (20 entries; not rendered).
- 2026-10-03: final draft `outputs/v1/s01_copper_v1_2_draft_p50.mp4`: 354 s for 300 frames = 1.18 s/frame including process start (draft, 50 percent). Blend 49 MB.

### Critique items (S1 table) and requirements: status
| Item | Status |
|---|---|
| 4.4-6.0 stretch gag (high) | done: Gary grips a bar on rack B and hauls in three heaves (pull_cable), rack slides 1.0 -> 2.04 m with each heave, Gary scoots back between heaves; red strain face; bundle taut with tremble; copper strands spring out with sparks at 4.95/5.35/5.65/5.95 plus small camera jolts; STRETCH at the top band 4.4-5.2 |
| 9.2-9.5 bang readability (high) | done: two-shot Gary - scope - Manager held 8.45-9.55 (Gary about 38 percent of frame height, faces 3/4 to camera); hole 1 pops oversize (1.6/1.9) with a pale see-through disc, settles 1.0 by 9.75; shot_hit_fall x1.35 (tip 9.5, ground 9.74); push onto the scope 9.55-10.0, ease-out, handheld and shake faded to 0 |
| 7.8-9.1 gun clipping / straight arm (medium) | done: Manager stands 1.46 m right of the screen, gun raise stays right of / above the scope; tremor on the gun while aiming; Gary in the same shot (startle at the gun pop) |
| 6.7-7.5 Manager walk-in / pale gun (medium) | done: stomp_walk into frame from 6.4 to 7.15; gun pops in at 8.07 with a puff and is darkened (scene material copies) |
| 4.2-4.3 Gary pop (medium) | done: turn_right_90 3.62-4.31, then pull_cable with blend-in |
| 3.0 / 3.6 ghost frame (low) | done: motion-blur shutter keyed 0 on cut frames, camera/rack/Gary keys CONSTANT before jumps |
| 2.25-4.3 PULSE DECAYS cropped (medium) | done (moved in front of the racks, 3.0-4.2) |
| 2.0-2.5 lamps (medium) | done: LED strips 12 -> 3, bench panel 6 -> 2, signs capped; fills warmed |
| 7.5-8.7 ear steam popcorn (low) | partial: steam emitters at 0.6 scale, start 7.35 (puffs visible about 7.55, steam_hiss cue) |
| 0.0-2.2 opening | unchanged (liked) |
| X1 subtitles / X2 HUD | done: `asm.narr_vo(1)`; LAB lines dropped; cards/FX/TC via asm only |

SFX retime: none (bang 9.2, strand pops, steam, hole pop all at their cue times).

### Remaining / requests
- Hole 1 at base radius 1.0 is about 14 px tall at 1080 in a two-shot; the S1 oversize pop is a workaround. Later scenes may need closer framing or a larger base radius.
- Steam is still a cluster of balls (asset look); a smaller/fewer-ball variant of `ear_steam` would help (asset owner).
- Character materials use DITHERED alpha (hole system); no speckle seen at draft in S1.
- Not run: fresh-context audit.
