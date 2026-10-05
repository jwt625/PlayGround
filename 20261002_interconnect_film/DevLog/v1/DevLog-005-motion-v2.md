# DevLog-005-motion-v2: character motion library v2 (agent C3)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Owner | C3 (motion library), part of DevLog-005-v1_2-polish-plan phase 1b |
| Scope | new folder `scripts/assets/characters/motion_v2/`; outputs `assets/components/characters/motion_v2/`, `assets/components/characters/previews_motion_v2/` |
| Not touched | `scripts/assets/characters/*.py` (v1), `assets/components/characters/*.blend|json` |
| Status | delivered 2026-10-03 (v2.0 of the library; see open questions) |

## 1. Findings on v1 (why it looked stiff)

- v1 `chars_actions.library` authors few keys per action (3-6) with BEZIER easing: no overlap, no hit-stop, no squash and stretch, no secondary motion. The walk (30 frames, step 0.70) ran at 1.17 m/s while its foot travel implied about 1.4 m/s, so the feet slid.
- topple_back = one rotation about the root with 8 keys; no stagger, no ground contact, no bounce or settle.
- The v1 pose engine (`Rig.pose`) solves IK targets by assigning `pose_bone.matrix` per key and bakes the result: that works well and is reused conceptually. v1.1 `s03_fight.py` already showed the next step (dense baking with timelines, spring-filtered head, squash channels, procedural stepping).
- Draft frames (S1 t=2..9.6 s, S3 t=1..7 s) show: walk with straight arms, a single stiff topple, clinches with arms through bodies.

## 2. Design

- Own engine `m2_engine.py` (no import of v1 files): works BY BONE NAME, rest dimensions (hip height, leg and arm lengths, hand frames) read from the armature, K = rest hip height / 0.925 (the `p_stature_m` property is the hat-on height and is NOT K). Works on gary, manager, npc and any later v2 mesh with the same skeleton; bones may be added (ignored).
- Pose language: baseline character space (x left, y back, z up, forward = -y, metres at stature 1.75, times K). One pose dict with fixed keys (root/hips loc+rot, spine (flex, twist, lean), neck, head, jaw, blink, lids, clavicle shrug/forward, hand IK (pos, finger dir, palm normal), elbow pole offset, finger curls, foot IK (x, y, z, pitch, yaw), knee pole, gaze, squash/stretch scales on chest/head/spine/hips/neck, ground clamp flag, hand/foot follow weights).
- Follow weights: `hfol` makes a hand target ride the chest (or head/root) so two-handed holds stay rigid when the torso rotates; `ffol` makes feet ride the root bone (falls).
- Dense baking: every action is sampled at 30 fps (loops over three cycles so spring filters are periodic), overlap spring filters (head, neck, hands) are applied to the sampled arrays, then solved per frame with the IK rig and written as LINEAR fcurves (Ramer-Douglas-Peucker key reduction below 0.2 mrad / 0.3 mm). Constant channels get one key.
- Locomotion is analytic: heel strike, flat, heel lift about a world-fixed ball pivot, swing with toe-up; stance foot moves back exactly at the body-frame speed, hips height from leg reach (so the bob emerges), pelvis yaw and roll, spine counter-twist, head stabilisation, arm counter-swing from the shoulder.
- Falls: stagger, tip about the feet (root bone pivot moved to the feet), ground hit with a capsule-model floor clamp (no mesh evaluation), bounce, flop, settle, twitch.
- Hit-stop: 3-frame holds (`HS=3`) flagged in the manifest (`hitstop` ranges); receivers start AT the attacker's impact frame.
- Face: separate `ACT_<asset>_v2_<name>_face` actions that key the root custom properties (`p_expr_*`, `p_anger`, `p_flush`) on the ROOT empty; `apply()` schedules them on the root's NLA.
- Layering: actions have a `layer_group` (full, upper, lower, arms, head); `_upper` variants of gestures key only upper-body channels so they can sit on an idle or a walk track (separate NLA track, REPLACE blend; ik hand targets are chest-parented so hands follow the underlying torso).
- Mirroring: asymmetric actions are authored left-acting; exported default `<name>` is the right-hand version, `<name>_L` the left (`turn_left_*`/`turn_right_*` are explicit pairs).

## 3. Checklist (final, 2026-10-03)

- [x] Engine, bake driver, manifest, review renderer, foot-slide measurement (`m2_measure.py`), gun muzzle test (`m2_guntest.py`), scene integration test (`m2_scene_test.py`, run 2026-10-03: idle, gait strip, turn with yaw hand-off, upper layer, shot_hit_fall with face NLA, all evaluated)
- [x] Locomotion: walk, walk_brisk, run, stomp_walk, turn 90/180 L/R
- [x] Idles: breathe, breathe_tense, fidget, look_around, arms_cross, thinking
- [x] Fight set (fight_idle, chest_bump, shove, haymaker, hook, uppercut, slap, head_butt, collar_grab_shake, duck, whiff_overbalance, flail_windmill, grapple) and reactions (head_hit, slapped, gut_hit, headbutt, shoved, chest_bump, collar_shaken), falls (fall_back, fall_back_brawl, shot_hit_fall), ground_twitch, get_up
- [x] Gestures: point, point_accuse, shout_rant, anger_outburst, manager_slow_burn, whiteboard_write/underline, typing, gun_ready/aim_hold/fire/raise_aim_fire, flinch, startle, shrug, facepalm, panic_wiggle, kneel_down/kneel_up, tie_fibers_kneel/stand, plug_connector, hug_squashed, hug_give, whipped, pull_cable
- [x] Layer variants: `_upper` for 13 gestures; `_L` variants for 8 asymmetric actions; turn L/R pairs
- [x] Bake gary, manager, npc: 92 actions + 40 face actions each, 13.8 MB per blend
- [x] Review strips: 66 gary, 28 manager, 10 npc PNGs (540x675 tiles, 0.1-0.2 s spacing; falls and long actions up to 0.27 s) in `assets/components/characters/previews_motion_v2/` (largest 267 KB, total 17 MB). Naming: `<rig>_<action>.png`, so the Manager's slow burn is `manager_slow_burn_manager.png`.
- [x] Fresh-context audit (static) run 2026-10-03; corrections applied (section 7)
- [ ] Not done: gary_v2/manager_v2 bake (meshes not delivered yet), gun-prop strips for the remaining gun actions on gary, per-NPC-variant libraries (npc_molexx etc.: one command each), true additive (COMBINE) variants, hold_plate / throw_up / hand_over_envelope / hug_leg / hold_printout v2 (v1 actions remain valid)

## 4. Progress log

- 2026-10-02: read plan, DevLog-003, rig bones, v1 pose engine, asm.py, S3 fight engine, draft frames. Wrote engine and the locomotion, idle, fight, reaction and gesture modules; first strips rendered on gary.
- 2026-10-02: first walk pass measured on the sole meshes: 19.8 mm per cycle, sole penetration -14 mm (toe dip); moved the ball pivot forward (0.19 -> 0.215 m) and re-measured: gary walk 7.4 mm, sole min z -2.2 mm.
- 2026-10-02: reach diagnostics added (IK targets beyond arm or leg length are clamped and reported per action and frame); fixed hanging-arm rest targets (0.285 instead of 0.296 m so arms keep a small elbow bend), turns and whiff now ride the root bone (hand follow), kneel and whiteboard targets pulled in.
- 2026-10-02: face curves, `_upper` variants, scene use table, bake of three rigs; discovered that review renders carried root p_ values over from the previous action (red faces): reset added, all strips re-rendered.
- 2026-10-03: scene integration test and gun test passed; v1 baseline measured; audit corrections applied; devlog finalised.

## 6. Results

Foot sliding (evaluated boot-sole meshes, vertices within 6 mm of the floor, two cycles, root moving at the baked speed; source: `previews_motion_v2/measure_<rig>.json`):

| Gait | Rig | slide per cycle (mean of both feet) | worst single-vertex drift in a contact run | lowest sole point | root speed |
|---|---|---|---|---|---|
| walk (30 frames, stride 0.917 m) | gary | 7.4 mm | 4.8 mm | -2.2 mm | 0.917 m/s |
| walk | manager | 7.3 mm | 5.2 mm | -2.4 mm | 1.004 m/s |
| walk | npc | 7.6 mm | 5.0 mm | -2.3 mm | 0.950 m/s |
| walk_brisk (24 frames) | gary / manager | 9.6 / 8.9 mm | 6.7 / 7.0 mm | -1.1 / -1.2 mm | 1.267 / 1.388 m/s |
| run (20 frames) | gary / manager / npc | 6.3 / 6.9 / 6.5 mm | 6.6 / 7.2 / 6.8 mm | -4.5 / -4.9 / -4.6 mm | 2.897 / 3.171 / 3.0 m/s |
| stomp_walk (36 frames) | gary / manager | 3.2 / 3.4 mm | 2.4 / 2.6 mm | -0.4 mm | 0.684 / 0.749 m/s |
| v1 walk (baseline, same measurement) | gary | 218.8 mm | 179.5 mm | -55.7 mm | 1.17 m/s |
| v1 run (baseline) | gary | 24.4 mm | 24.4 mm | -3.8 mm | 3.0 m/s |

Gun (manager, `previews_motion_v2/gun_muzzle_manager.json`, shotgun mounted on HOOK_gun_grip_R with rot z = pi): in `gun_aim_hold` the muzzle pitch is 0.00 degrees and the yaw stays within -0.042..+0.052 degrees over the loop; `gun_fire` flips the muzzle up by 15.4 degrees at the shot (frame 6) and returns to 0.00; `gun_raise_aim_fire` goes from 42.6 degrees (low-ready) to level and fires at frame 44.

Bake time: all 92 actions of a rig bake in about 10 s (meshes hidden during solving).

## 7. Audit corrections and known limitations

Applied after the fresh-context audit: face strip now follows start/end trimming and hold; root yaw hand-off evaluates the keyed yaw at t0, scales with repeat and trimming, and is only keyed for non-held strips; `ensure_action` accepts an asset id string; `walk()` rejects a single point.

Known limitations (not fixed):
- IK reach: some flung or contact poses ask for hand targets beyond arm length and are clamped (hand falls short, arm straight); the manifest lists `reach_violations_m` and `reach_worst_frame` per action. Largest on gary: get_up 0.58 m at frame 4 (the hand-follow blend between the lying end pose and the ground push), whiff_overbalance 0.36 m at frame 46, whipped 0.28 m at frame 5 (spin), manager_slow_burn 0.12 m at the head-pop (chest stretch scales the follow target). Check these frames if a close-up is planned.
- `meta.root_to_root_m` and other distances are in BASELINE metres (multiply by K, gary 0.9657, manager 1.0571, npc 1.0); `end_root_loc_y_m` and `root_speed_mps` are already K-scaled.
- Hit-stop lengths: 3 frames for attacks and most reactions; 4 frames for shot_hit_fall and react_head_hit's first key (see `hitstop` in the manifest); collar_grab_shake grab hold is 2 frames.
- Body looks (thin arms, small hands) are the v1 meshes; the strips only judge motion.
- Loop strips are sampled at 0.4 s spacing for the slow idles (breathing), 0.1-0.15 s for the rest.
- The `whipped` end pose is sitting, not lying; the `get_up` variant starts from the lying back pose of fall_back only.
- p_expr_shock renders with a strong red cast on the v1 face shader; this is the existing expression, not part of the motion.

## 8. Open questions for Wentao

1. Should the Manager's anger beat (150 frames = 5 s) fit inside a 10 s scene next to the dialogue, or should a shorter 90-frame version be baked? (Design values are in `m2_acts.b_slow_burn`.)
2. Fall travel: shot_hit_fall leaves the body 0.89 m (gary) behind the start point; confirm that the camera framing in S1/S2 tolerates this, otherwise add an offset parameter.
3. Walk speeds are design values (about 0.92-1.0 m/s calm walk, 1.3-1.4 m/s brisk, 3 m/s run); scene timing that relied on v1's 1.17 m/s walk must use `M2.walk` (it reads the baked speed).

## 5. Usage guide for scene agents (README)

Files: library blends `assets/components/characters/motion_v2/<rig>_motion_v2.blend` (actions only, fake users; rigs: gary, manager, npc), manifests `<rig>_motion_v2_manifest.json` (every action: frame range, loop flag, events, hit-stop ranges, hooks, layer group, face action, recommended use; plus `scene_use` per scene), previews in `assets/components/characters/previews_motion_v2/`.

Rule: actions are baked PER RIG because they carry K-scaled IK positions (gary K = 0.9657, manager K = 1.0571, npc K = 1.0). For another NPC variant (npc_molexx, npc_openay, ...) bake its own library: `Blender -b assets/components/characters/npc_molexx.blend --python scripts/assets/characters/motion_v2/motion_v2.py -- bake all` (about 10 s per rig), then `apply` finds `<asset_id>_motion_v2.blend` automatically. After C1 delivers gary_v2/manager_v2 do the same on those blends (same bone names; the bake reads rest dimensions from the armature).

In a scene script (after `asm.append`):

```python
import sys; sys.path.insert(0, "scripts/assets/characters/motion_v2")
import motion_v2 as M2
M2.apply(gary, "idle_breathe", 0.0, hold=True, repeat=4)               # base layer
t = M2.walk(gary, 1.0, [(0,0,0), (0,-3,0)])                             # root path + gait, feet locked (strip speed = speed / baked speed)
M2.apply(gary, "turn_left_90", t, hold=False)                           # root yaw hand-off is keyed automatically
M2.apply(gary, "shrug_upper", t + 1.5, layer="gesture", hold=False)     # upper-body layer over the idle track
M2.apply(mgr, "gun_raise_aim_fire", 6.0)                                # shot event frame is in the manifest: t_shot = 6.0 + 44/30
M2.apply(gary, "shot_hit_fall", t_bang + 3/30)                          # frame 0 = blast frame; 3-4 frames hit-stop are inside
```

- `apply(asset, name, t0, speed=1, hold=True, repeat=1, blend_in=0, blend_out=0, layer=None, blend="REPLACE"|"COMBINE"|"ADD", influence=1.0, start_frame, end_frame, face=True, key_root_yaw=True)` mirrors `asm.play` (short name -> `ACT_<asset_id>_v2_<name>`). `layer="x"` puts the strip on a named NLA track (later tracks override earlier ones); `_upper` actions key only upper-body channels so they combine with an idle or walk strip underneath.
- Looping interior sections: `loop_start`/`loop_end` events (collar_grab_shake, hug_squashed): use `start_frame`/`end_frame` with `repeat`.
- Face: every action with a `*_face` action keys the ROOT empty properties (`p_expr_*`, `p_anger`, `p_flush`); `apply` schedules it on the root NLA. If the scene keys these properties itself, pass `face=False`.
- Events (manifest `events`, frame numbers in the action, 30 fps): `contact`/`impact` (attack landing: start the receiver reaction on the same frame), `shot`, `ground_hit`, `twitch`, `stomp_L/R`, `contact_L/R` (footsteps), `click`, `squeeze1..4`, `steam_start`, `boil_over`. Hit-stop ranges are in `hitstop` (the pose is held there, normally 3 frames; do not add another hold).
- Attack/reaction pairing: attacker `X` (right hand) pairs with `react_head_hit`; `X_L` with `react_head_hit_L`. Distances: `meta.root_to_root_m` is in baseline metres (multiply by the rig K).
- Turn and whiff: actions with `root_yaw_delta_rad` rotate the root BONE; use `hold=False` (apply keys the object yaw step at the strip end; with `hold=True` the bone yaw simply stays and no object yaw is keyed). Falls end with the root bone offset recorded in `meta.end_root_loc_y_m` (body lies behind the start point).
- Gun: mount the shotgun on `HOOK_gun_grip_R` with rot z = pi as in S1/S2. The barrel axis equals the right-hand finger direction. Measured muzzle (manager, `m2_guntest.py`): aim hold pitch constant and yaw within 0.06 degrees, recoil flips the muzzle up by 15.4 degrees (gun_fire), frames 44 (raise_aim_fire) and 6 (gun_fire) are the shot.
- Do not combine two full-body actions on the same bones in different tracks (the later one wins on every channel it keys).

Bake, review and test commands (one Blender at a time): `scripts/assets/characters/motion_v2/run_final.sh`; `m2_review.py` (strips + foot-slide), `m2_guntest.py`, `m2_scene_test.py` (asm integration test, passed).

Module map: `m2_engine.py` (pose engine, timeline, baker), `m2_lib.py` (registry), `m2_loco.py`, `m2_idle.py`, `m2_fight.py`, `m2_react.py`, `m2_acts.py`, `motion_v2.py` (API), `m2_render.py`, `m2_measure.py`.
