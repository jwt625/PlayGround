# DevLog-012: Characters + animation (workstream C)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Delivered 2026-10-10 09:27; v2 helpers (hold_bird, lens look, S6) and owner outfit decision (Subject = exact Gary) added 2026-10-10 11:15 (section 10) |
| Brief | DevLog-010 section 3 (workstream C) and section 4 (interfaces) |
| Depends on | DevLog-003 (hero model structure), DevLog-006 (crude v0.2 crowd proxy, head rig, bird scale 2.6), DevLog-007 section 10 (rack v2 slots, `neck_extension_m`), sibling `../20261002_interconnect_film/DevLog/v1/DevLog-005-characters-v2.md` and `DevLog-005-motion-v2.md` (Gary v2 rig and motion_v2 actions) |
| Owns | `scripts/anim/`, `assets/components/characters/`, `outputs/assets/characters/`, this DevLog |
| Authors | Claude (characters/animation subagent C), at Wentao Jiang's request |

## 1. Plan (written before implementation)

1. Subject: copy Gary v2 rig blend + motion_v2 action library + manifests from the sibling project (source untouched). Film variant built at append time by `scripts/anim/subject.py`: hoodie + jeans + sneakers (recolour of the existing outfit shells, hard hat and tools hidden, hood and sleeves as bone-parented meshes), coffee cup on the right hand hook. API: append, place, walk path (root-motion safe), stop/idle, look at, startle, freeze, glance over shoulder, sip coffee, reach.
2. Pigeon rig (`scripts/anim/pigeon_rig.py`): one pivot layout for the hero GLB and a light proxy derived from it (body, head, two wings, tail, two legs as separate parented parts). Keyframe-baked, deterministic action functions.
3. Behaviours (`scripts/anim/behaviors.py`): pure-Python scheduler (head motion, link re-pairing TX right eye -> partner RX left eye, mutual replies, periodic Subject observation) emitting keys and link events; bpy apply layer.
4. Demo build + EEVEE renders (540x540, 6 s); look; iterate.
5. API doc (section 5).

## 2. TODO

- [x] Read DevLog-010, 006, 003, 007, sibling DevLog-005 characters/motion, build_crude_film.py proxy/head rig
- [x] Copy Gary v2 + motion_v2 library + manifests into `assets/components/characters/`
- [x] Inspect hero GLB part bounds (pivots) and the gary_v2 blend contents
- [x] `pigeon_rig.py`: hero rig, proxy library, action functions
- [x] `behaviors.py`: scheduler + link events + bpy apply; self-test (pure Python)
- [x] `subject.py`: append, hoodie variant, coffee cup, actions
- [x] Demo build + stills + mp4; review; iterations (section 7)
- [x] Scale test (220 birds) and a run on R's hall layout v2 (239 rack birds with neck extensions)
- [x] API doc, as-built table, open issues
- [x] v2 work order (DevLog-010 section 9 #1, #4, misc): hold_bird, lens look, S6 helpers, demo (section 10)
- [x] Owner decision: Subject = exact Gary outfit (default), demos re-rendered
- [x] Owner F1 (DevLog-015): neck deforms with the head (hero + proxies), tests, demos re-rendered (section 11)
- [x] v3 (DevLog-018): feet contact (place_on_floor, perch helpers, 239-bird check) and Gary goes about his day (section 12)
- [ ] Fresh-context audit (coordinator's call)

## 3. As built

| Path | What |
|---|---|
| `scripts/anim/pgeom.py` | Pure-Python geometry: bird frame constants (pivots, eye hooks), head rotation, eye world position, lateral-eye aim solver, link solver, one-eyed stare, neck shift, Catmull-Rom path helpers |
| `scripts/anim/keys.py` | Fast keying (F-curve insert FAST), value evaluation, `flush()` |
| `scripts/anim/pigeon_rig.py` | Hero rig (`build_hero_rig`), proxy library (`build_proxy_library` / `proxy_library`), proxy rigs (`make_proxy_rig`, full or head-only, optional neck extension), all motion functions |
| `scripts/anim/behaviors.py` | `BirdSpec`, `Params`, `schedule()`, `apply()`, `Track`, `LinkEvent`; `uv run --no-project python scripts/anim/behaviors.py` runs the self-test |
| `scripts/anim/subject.py` | Subject = Gary (exact gary_v2 outfit by default; hoodie variant only with `outfit="hoodie"`), cup prop, NLA track layout, walk/turn/idle/look/glance/startle/freeze/gesture/sip/reach, head track; v2: `hold_bird`, `look_at_lens`, `glance_at_camera`, `walk_to_mark`, `look_around_slow`, `startle_freeze`, `face_angle`, `unreadable_frames`, `cup_show` |
| `scripts/anim/demo_characters.py` | Demo build (modes `stills f,f,..`, `flystills f,..`, `anim`) |
| `scripts/anim/test_neck.py` | Neck deform test (owner F1): modes `stills` (extreme poses, gap metrics deform on/off, close-ups) and `crowd` (20 scheduled birds, gap sampling, clip) |
| `outputs/assets/characters/neck_test/` | `sheet_hero.png`, `sheet_proxy.png`, `sheet_proxy_ext126.png` (top row static neck, bottom row deform; poses rest / yaw +110 / yaw +150 roll +35 / roll +60 nod +20; back, right, left views), `neck_crowd_20261010.mp4` (540x540, 3 s, 20 scheduled birds), `contact_crowd.png` |
| `scripts/anim/test_feet.py` | Feet contact check over all 239 crowd slots of R's hall (modes `measure`, `fix`, optional `render`) |
| `scripts/anim/demo_gary_casual.py` | v3 demo: Gary's casual day + birds planted on post / railing / wire (modes `stills`, `anim`); prints perch-contact and face-readability checks |
| `outputs/assets/characters/demo_gary_casual_20261010.mp4`, `casual_still_f060/125/165/220/275.png`, `feet_test/sheet_measure.png`, `feet_test/sheet_fix.png` | v3 demo (540x674, 30 fps, 300 frames, 10.0 s) and feet close-ups before/after |
| `scripts/anim/demo_hold_bird.py` | v2 demo: hand-hold at R's real hero cell + lens glance (modes `anim`, `stills f,..`, `s6`); prints a bird-to-Gary clearance check |
| `scripts/anim/preview_subject.py`, `scripts/anim/test_crowd_scale.py` | Subject preview stills; 220-bird scale/timing test |
| `assets/components/characters/gary_v2.blend`, `gary_v2.json` | Copied from the interconnect film (sha1 identical to the source) |
| `assets/components/characters/motion_v2/gary_v2_motion_v2.blend`, `..._manifest.json` | Copied action library (92 actions + face actions) and manifest |
| `assets/components/characters/pigeon_proxy.blend` | Proxy part meshes (0.6 MB; rebuilt automatically by `proxy_library()` when the hero GLB is newer) |
| `outputs/assets/characters/demo_characters_20261010.mp4` | Demo: 540x540, 30 fps, 180 frames, 6.0 s, H.264 yuv420p CRF 18, +faststart, no audio |
| `outputs/assets/characters/demo_still_f020/040/096/128/150.png` | Stills from the final demo frames |
| `outputs/assets/characters/demo_characters.blend`, `demo_link_events.json` | Demo scene; link events of the demo schedule |
| `outputs/assets/characters/demo_hold_bird_20261010.mp4`, `demo_hold_still_f040/047/090/124/178.png`, `demo_hold_bird.blend` | v2 hand-hold demo: 540x540, 30 fps, 190 frames, 6.33 s, H.264 yuv420p CRF 18, +faststart, no audio |

Proxy part sizes (faces after decimation): body 3,800, head 4,412 (+ 2 x 8 lenslet domes), wing 1,039 each, tail 1,008, legs 1,516 / 1,644; head-only rack bird = rest mesh + head + 2 lids. Hero GLB parts on the rig: head 148, body 92, legs 8 + 10, tail 3 (fan + leash), wings 2 + 2.

## 4. Verification

| Check | Result |
|---|---|
| Scheduler self-test (220 synthetic birds: 200 rack + 20 street, 1,860 frames, seed 7) | 1,550 links (372 in mutual pairs), 2,115 observations, 11 street hop-turns, 219/220 birds linked, 14.2 links per bird; aim error > 3 deg: 0; head-move overlaps: 0; limit violations: 0; line-of-sight violations (beam within 45 mm of another head): 0; deterministic (identical rerun); forced link placed; all 220 birds start the forced observation in its first 10 frames; 1.8 s |
| R's hall layout v2 (`hall_layout.json`, 239 crowd slots, `neck_extension_m` per slot) | 2,157 links (76 % across the aisle, mean beam 2.23 m), every bird linked; head centres match R's `head_center_neutral` + extension within 13 mm (R used the crude primitive's head centre, 5 mm hero behind the eye midpoint) |
| Blender scale test (`test_crowd_scale.py`) | 220 rigs (200 head-only + 20 full), 2,440 objects, schedule 2.1 s, apply 1.8 s, 474,844 keys, frame evaluation 4 ms/frame |
| Sip (evaluated lips vs cup rim, hold frames) | rim to nearest lip vertex 1.7-1.9 cm, no intersection |
| Subject walk foot lock (heel bone world position, straight 4 m walk) | stance drift 0.8 mm/frame (contact phases), i.e. about 7-14 mm per stance, same as the motion_v2 library value (7.4 mm per cycle); ease-in/out keeps the feet locked (strip time follows the travelled distance) |
| Smoke test of every public function (proxy, head-only, subject) | all ran headless without errors |
| Demo | rendered 3 times (stills rounds plus 2 full renders); final contact sheet and full-res frames viewed (section 7) |

## 5. API (for S)

### 5.1 Conventions

- Frames: every function takes absolute scene frames (S: `F(idx, t) = 1 + idx*300 + round(t*30)`); `keys.frame(t, offset)` converts seconds. 30 fps.
- Import: `sys.path.insert(0, "<repo>/scripts/anim")` then `import pigeon_rig as PR, behaviors as BH, subject as SJ, keys as KY`. Modules resolve asset paths from their own location (no absolute paths).
- After keying, call `KY.flush()` once (recalculates Bezier handles of FAST inserts). `BH.apply()` flushes itself.
- Bird frame (pgeom): origin on the ground between the feet; +X beak; +Y bird LEFT = RX (cyan) eye; -Y bird RIGHT = TX (red) eye; +Z up; hero metres (GLB true size); root scale = world size factor (film `S_BIRD = 2.6`). World bird yaw: beak direction (cos yaw, sin yaw); R's hook +X = beak, so yaw = hook yaw.
- Head: euler XYZ = (roll, nod, yaw) on `<name>_head`. yaw + turns the beak to the bird's left; nod + tips the beak down; roll + lifts the RX eye and lowers the TX eye. Eyes are lateral, so a lateral eye is aimed by yaw (bearing) + roll (elevation); pigeon-style one-eyed stares come out of the solver automatically.
- Root: euler XYZ = (roll, -pitch_up, yaw). `PR.place(r, f, pos, yaw, pitch_up, roll)`.
- Subject: front = -Y of the character; world yaw 0 = facing -Y; `SJ.yaw_face(src, dst)`. Positions are world metres, z = floor.
- Determinism: every random choice uses `random.Random(seed ...)`; `BH.schedule(..., seed)` is reproducible (checked). No Python `hash()` is used.

### 5.2 pigeon_rig.py

| Function | Notes |
|---|---|
| `build_hero_rig(name="HERO", glb=HERO_GLB, scale=2.6, loc, yaw, coll=None, hide_leash=False)` | Imports the hero GLB, reparents top-level parts to the pivots, adds TX/RX lids. Returns `PigeonRig` (`r.glb` imported objects, `r.by("HOOK_beak_hinge")` lookup). The GLB tail leash rides on the tail pivot; `hide_leash=True` hides it (S's stretch leash). |
| `proxy_library(path=PROXY_BLEND, rebuild=False)` | Loads `pigeon_proxy.blend` (builds it from the hero GLB when missing or older than the GLB). Returns `{part: mesh}`: body, head, head_full, wing_L/R, tail, leg_L/R, rest, lid_tx, lid_rx. |
| `make_proxy_rig(name, lib, loc, yaw, scale=2.6, mode="full"/"head", lids=True, coll=None, neck_ext=0.0)` | `full`: street/flying birds; `head`: rack crowd (static rest mesh + head + lids). `neck_ext` = R's `neck_extension_m` (world m): head pivot moves forward, neck filler tube added. |
| `PigeonRig` | `.root .body .breath .head .lid{tx,rx} .wflap{L,R} .wopen{L,R} .tail .leg{L,R} .eye{tx,rx} .head_pivot .scale`; `.eye_world(k, f)`; `PigeonRig.from_root(root_obj)` rebuilds handles from a saved blend. |
| `place(r, f, pos, yaw, pitch_up=0, roll=0)`, `rest_pose(r, f)` | Root key; neutral key on every channel (call once per shot). |
| `pose_body(r, f, pitch_up, crouch 0..1, grounded=True)` | Body pitch about the hip line + crouch; legs compensate so planted feet do not move. |
| `head_pose(r, f, yaw, roll, nod, bob)`, `head_snap(r, f, yaw, roll, nod, bob, dur=3, overshoot=0.12)` | Raw key; pigeon saccade (hold, snap in 2-4 frames with overshoot, settle). |
| `head_look_at(r, f, target, eye="auto"/"tx"/"rx", dur=3)` | One-eyed look at a world point (uses the root pose keyed at f). Returns (settle_frame, eye, yaw, roll, exact). |
| `head_track(r, f0, f1, target_fn, eye="auto", step=(6,12), seed)` | Stepwise pigeon tracking of a moving target (`target_fn(frame) -> xyz`). |
| `head_tilt(r, f, roll)`, `head_bob(r, f, amp, n)` | Curious tilt and back; quick thrusts. |
| `breathe(r, f0, f1, rate_hz=0.5, amp=0.012, seed)` | Chest scale keys every quarter cycle (full rigs). |
| `wings(r, f, open 0..1, flap, sweep)`, `wings_fold/half/open(r, f)` | Open = slerp folded -> spread (plus a broader, flatter wing shape); flap + = up stroke; sweep + = tips forward. |
| `flap_cycle(r, f0, f1, hz=4.0, up=55 deg, down=40 deg)`, `glide(r, f0, f1)`, `tail(r, f, up, spread)` | 4 Hz default is a readability choice at 30 fps (real pigeons flap faster). |
| `blink(r, f, eye="both"/"tx"/"rx")`, `blinks(r, f0, f1, seed)`, `tx_disable(r, f0, f1, ramp=3)` | Disc lids, scale z 0 open -> 1 closed; TX lid closed = TX disable. |
| `hop(r, f0, p1, yaw1=None, height, air=8)`, `turn_in_place(r, f0, yaw1, hop_turn=True)`, `perch_shuffle(r, f0, steps=2, direction=+1 left/-1 right)`, `walk(r, f0, pts, speed)` | Ground motion; `walk` has the pigeon head bob (thrust, then the head holds still in space while the body advances). |
| `flight(r, f0, pts, speed=None, takeoff=True, land=True, flap_hz=4, glide_frac=0.35, end_yaw=None, seed)` | pts = world feet positions (perch ... landing spot). Crouch, launch, power flaps, cruise, glide, braking flaps, flare (nose up, wings forward, tail fanned), feet reach forward, touchdown, absorb, fold. Returns `{f_launch, f_touch, f_end}`. `take_off(...)` = no landing; `landing(...)` = starts airborne. |

Channel ownership (do not key one channel from two functions over the same frames): root (place, hop, shuffle, turn, walk, flight), body/legs (pose_body, legs), head (head_*), wings, tail, lids. The behaviour scheduler only writes head keys, lids (blinks) and street hop-turns.

### 5.3 behaviors.py

```python
birds = [BH.BirdSpec(id, pos, yaw, scale=2.6, kind="rack"|"street", yaw_lim=100*deg, roll_lim=40*deg, group="rack",
                     blocked=[(fa, fb)], neck_ext=cell["neck_extension_m"]) ...]
sched = BH.schedule(birds, f0, f1, seed=7, subject=track,                  # track: callable frame->xyz or [(frame, xyz)]
                    force_observe=[(fa, fb)],                               # all heads stare at the Subject (staggered)
                    forced_links=[(f, tx_id, rx_id, hold)],                 # story links (e.g. S4 across-aisle bird -> hero)
                    obstacles=[(centre, radius)], params=BH.Params(...))
BH.apply(sched, rigs, breathe=False, blinks=True)                           # rigs: {id: PigeonRig}
sched.links      # [LinkEvent(f0, f1, tx, rx, mutual, pair, tx_eye, rx_eye)] f0 = both heads aligned
sched.events_json(path)   # {links: [... t0, t1 seconds ...], observes: [...]} for link VFX + SFX blips
```

- Defaults (`Params`): link hold 24-48 frames, reply hold 18-36 (p_mutual 0.3: the partner answers with a reversed link, both heads swap eyes), partner reaction 2-6 frames, saccade 2-4 frames, idle holds 8-30, Subject observation every 75-180 frames for 24-54 frames (stepwise re-aim every 6-12), partner distance 0.25-4.5 m, line-of-sight clearance 45 mm (scaled) from every other bird's head and from `obstacles`.
- Rack birds: head-only, inside `yaw_lim`/`roll_lim` (cage). Street birds (`kind="street"`): may hop-turn when the Subject is outside their head range (`can_turn=False` to forbid).
- `blocked` windows: the scene animates the bird itself there (hero shots, flights); the bird is neither moved nor picked as a partner.
- Positions are static per schedule: for birds that move (flights, shuffles, walks) block those windows or schedule per segment.

### 5.4 subject.py

```python
s = SJ.append_subject()                          # exact Gary + cup prop; NLA tracks base < loco < act < arm < look
SJ.place(s, F(0, 0.0), (x, y, 0), yaw)
SJ.idle(s, f0, f1)                               # base idle (held); walks/acts blend out into it
f = SJ.walk_path(s, f0, [(x0, y0), (x1, y1), ...], gait="walk"|"walk_brisk"|"run"|"stomp_walk", speed=None)
f = SJ.turn(s, f, "R"|"L", 90|180)               # stepping turn; object yaw handed over at the strip end
SJ.look_at(s, f0, f1, target_xyz_or_callable, track=True, w_chest=0.0)   # head/neck/eyes ADD overlay, clamped 80/35 deg
f = SJ.glance_over_shoulder(s, f0, "L"|"R")
f = SJ.startle(s, f0, then_freeze=0); f = SJ.freeze(s, f0, f1); f = SJ.gesture(s, f0, "shrug_upper")
f = SJ.sip_coffee(s, f0, raise_frames=14, hold=22, lower_frames=14)
f = SJ.reach(s, f0, target_xyz)                  # right hand toward a point (clamped to 0.5 m from the shoulder)
track = SJ.head_track(s, f0, f1, step=3)         # [(frame, head xyz)] for BH.schedule(subject=track)
```

- Walk: Catmull-Rom path, ease-in 0.35 s / ease-out 0.5 s, strip time = travelled distance / baked root speed (cyclic), end nudged by at most half a step to stop on a passing pose, blend-out into the idle. Returns the stop frame.
- Any motion_v2 action is available by short name (`SJ.action(s, name)`, list in `assets/components/characters/motion_v2/gary_v2_motion_v2_manifest.json`); face actions (`*_face`) are scheduled on the root automatically.
- `look_at` and `reach` read the root keys at their frames, so key the walk/turn first.

### 5.5 Rack crowd recipe (S, film)

```python
lib = PR.proxy_library()
rigs, specs = {}, []
for c in layout["cells"]["list"]:
    if c["occupant"] != "crowd": continue
    pos = (c["feet_origin"][0], c["feet_origin"][1] + LAYOUT_DY, c["feet_origin"][2]); yaw = math.radians(c["bird_yaw_deg"])
    bid = c["rack"] + "_" + c["bird_hook"]
    rigs[bid] = PR.make_proxy_rig(bid, lib, pos, yaw, mode="head", neck_ext=c["neck_extension_m"])
    PR.rest_pose(rigs[bid], 1)
    specs.append(BH.BirdSpec(bid, pos, yaw, group="rack", neck_ext=c["neck_extension_m"]))
sched = BH.schedule(specs + street_specs, 1, 1861, seed=7, subject=SJ.head_track(s, 1, 1861, 3), force_observe=[...])
BH.apply(sched, dict(rigs, **street_rigs))
```

## 6. Design choices and deviations

- Subject look: SUPERSEDED 2026-10-10 by the owner (section 10): the Subject is Gary with his exact outfit. The earlier hoodie variant (kept behind `outfit="hoodie"`, not used) was: hard hat, bib, straps, tool belt, tools hidden; shirt and a fitted hoodie torso shell (lofted around the overalls/shirt/body vertices + 14 mm, skinned to hips/spine/chest) in heather charcoal (0.16, 0.17, 0.19); long sleeves, elbow balls, cuffs, rolled hood, two drawstrings (bone-parented); overalls legs -> dark denim; boots -> off-white sneakers. Hood down, so Gary's bald crown with side hair shows (no hat). Paper coffee cup with kraft sleeve and lid on `HOOK_hand_R`.
- Sip: right-arm IK overlay; head tips back 0.12 rad. Measured on the evaluated meshes during the hold: cup rim to nearest lip vertex 1.7-1.9 cm (cup top centre 5.9 cm from the lip centre); before the last correction 4.5-5.0 cm (the IK lands about 2 cm below its target; the target compensates).
- Wings: the hero's folded wing is a narrow, curved fin stack; a rigid rotation read as a thin blade, so the open pose also rescales the wing in its folded frame (span x1.35, wrap flattened x0.35, chord x1.6). Open-wing orientation: tip outward, leading edge forward (the folded outer face becomes the underside).
- Lids are simple discs over the eye (scale-animated), not modelled eyelids.
- Flap rate 4 Hz (take-off power flaps 5.6 Hz) for readability at 30 fps.
- Neck extension follows R's suggestion (0 / 0.046 / 0.126 m by tier); at 0.126 m the neck reads long (goose-like) in profile; S can scale it down per slot (the scheduler takes the same value).

## 7. Demo review and iterations

| Round | Seen | Fixed |
|---|---|---|
| Subject preview 1 | overalls covered the torso to the chest (bib hidden but the trouser top is high); cup held horizontally | fitted hoodie torso shell; cup frame corrected (axis opposite the finger direction) |
| Subject preview 2 | bare skin at bent elbows; sip pose beside the cheek | sleeve overlap + elbow balls; sip target derived from the cup geometry |
| Demo stills 1 | beam note at the camera near clip (faded), too large; hero crate off frame | note at 0.03 m, clip 0.01 m, size scaled with lens; crate moved in |
| Flight close-ups | open wings read as thin blades | open-wing rescale (section 6); legs verified tucked (leg pitch 1.31 rad during cruise) |
| Demo full render 1 | a forced hero link drew a beam through the Subject; flight-follow camera lost the bird; Subject sipped side-on | in-frame forced links; constant-offset follow camera that ends low on the landing; `SJ.turn` to face the street |
| Demo full render 2 | rack links passing through intermediate birds' heads | line-of-sight test in the scheduler (0 violations in the self-test) |
| Final | walk in, birds re-pairing (10 links, 13 birds), take-off, flight, flare/landing beside the Subject, pigeon walk toward him, turn, look down at the bird, sip; all birds observe him on the stop | - |

## 8. Open issues

1. Line of sight ignores moving bodies (the Subject, flying birds) and rack geometry (only bird heads + optional static spheres); S can pass rack posts/bezels as `obstacles`.
2. Scheduler positions are static per call; moving birds need blocked windows or per-segment schedules.
3. Hero rig: flex ribbons that span body and legs (e.g. the 26-trace leg flexes) stay on the body pivot unless their centroid is below 15 mm, so hero legs tucked in flight may separate from those ribbons; not checked visually (the demo hero does not fly; proxies have no flexes).
4. Subject turning in place uses motion_v2 turn actions (90/180 only); other angles would need a root-yaw ease (feet skate).
5. `look_at` clamps at 80 deg yaw / 35 deg pitch; beyond that combine with `turn` or `w_chest`.
6. Street-bird link turns are not implemented (only observation turns); street links happen only inside the head limits.
7. Lids (hero and proxy) use the hero cranium material (gunmetal); a darker lid colour may read better for TX disable close-ups.
8. Fresh-context audit not yet run.
9. (v2) `hold_bird` is right-hand only; the carrier is baked from the evaluated hand, so call it after any other Subject keys in its window (later body/arm strips there would leave the bird off the palm).

## 9. Progress log

- 2026-10-10 08:32: briefs read; Gary v2 (`gary_v2.blend`, `gary_v2.json`) and `motion_v2/gary_v2_motion_v2.blend` + manifest copied (sha1 of gary_v2.blend identical to the source).
- 2026-10-10 08:33: hero GLB and Gary blend inspected (part bounds, hooks; Gary bones, actions, hook frames, materials); log `outputs/logs/charC_inspect_*.log`.
- 2026-10-10 08:40: `pgeom.py`, `keys.py`, `pigeon_rig.py`, `subject.py`, `behaviors.py` written; scheduler self-test passed (220 birds, 0.97 s).
- 2026-10-10 08:45-08:55: Subject previews (2 rounds), sip/walk debugging; foot lock confirmed on the heel bone.
- 2026-10-10 08:56-09:07: demo stills (3 rounds), flight close-ups (2 rounds), 2 full renders; wings reshaped; turn added; camera and links fixed.
- 2026-10-10 09:08: line-of-sight test; scale test (220 rigs); R hall layout run; neck extension added (rig + scheduler); final render encoded `outputs/assets/characters/demo_characters_20261010.mp4`; frames, previews and work files deleted.
- 2026-10-10 09:15: smoke test of all functions, lint clean (ruff E/F/W), self-test log `outputs/logs/charC_selftest_*.log`; DevLog written.
- 2026-10-10 09:25: sip measured (rim-lip gap 4.5-5.0 cm), target corrected (1.7-1.9 cm); demo re-rendered and re-encoded with the final code, stills refreshed, frames deleted.

## 10. v2 work order and owner outfit decision (2026-10-10)

### 10.1 Inputs

- Coordinator: DevLog-010 section 9 items 1 (S6: Subject central, frozen at the end, heads turning to him), 4 (hero sits on the Subject's hand, Gary reaches in, hand turns to show the belly) and misc (Subject glances at the lens on "It sees you").
- Owner (verbatim, 2026-10-10, via coordinator): "use the same exact outfit gary has for the guy. i.e., he is gary." -> `append_subject()` default `outfit="gary"`: plain gary_v2 (not holes30) with hard hat, overalls, bib, tool belt and tools exactly as in the source asset; the hoodie code stays behind `outfit="hoodie"` (off). Coffee cup kept as an optional hand prop (`cup=True`); hidden with `cup_show(s, f, False)` during the hand-hold.

### 10.2 API added (subject.py; existing APIs unchanged)

| Function | Notes |
|---|---|
| `hold_bird(s, f0, f1, cell_pos, cell_yaw, rig=None, present_dir=None, end="cell"/"hold", look=True)` | Reach to the cell lip (palm up, fingers into the cell), the bird hops out forward onto the palm (feet planted), the palm dips and rises to chest height while the bird turns crosswise (beak to his left), the bird rolls onto its back about its long axis, the hand tilts belly/feet toward `present_dir` (world vector, e.g. toward the camera), hold, roll back upright, then `end="cell"`: lowered, the bird turns to face out and hops back into the cell (feet on the cell floor); `end="hold"`: stays perched crosswise at chest height (arm strip held). Fixed phase lengths (reach 12, hop 9, raise 16, flip 10, tilt 10 ... retract 12 frames); the present hold takes the remainder (minimum span f1 - f0 = 122 frames for `cell`, 77 for `hold`). Returns `{carrier, frames, matrices}`: `carrier` = world empty `SUBJ_bird_carrier` keyed every frame (location + quaternion; bird root frame: feet origin, +X beak, +Z up, no scale); `frames` = phase frames (reach, hop0, on, raised, flip0, back, present, unpresent0, unpresent, upright, lowered, in, end). With `rig=` (hero PigeonRig) the root is keyed from the same matrices (euler with continuity) plus crouch, wing-balance and tail keys; do not key that root elsewhere in [f0, f1]. Gary's head follows the bird (head pitch limited to 18 deg so the hard-hat brim stays clear). Right hand only. |
| `palm_frame(s)` | Palm-surface centre in the hand hook frame (measured from the rest-pose hand vertices). |
| `look_at_lens(s, f0, f1, cam, ramp=4, w_chest=0.12)`, `glance_at_camera(s, f, cam, hold=24)` | Head + eyes onto a camera object (follows its motion; F-curves if unparented, else evaluated). For "It sees you". |
| `walk_to_mark(s, f0, mark, via=(), face=None, gait="walk")` | Walk from the current root position to the mark; if `face` (e.g. camera position) needs more than 55 deg of body turn, a stepping 90/180 deg turn toward it; smaller residuals are left to look_at. |
| `look_around_slow(s, f0, f1, center, span_deg=40)` | Wary look-around centred on a point (centre -> left -> hold -> right -> hold -> centre), yaw within +-span of the centre so the face stays readable. |
| `startle_freeze(s, f0, f_end, pose_frame=18, look=None)` | motion_v2 `startle` (shock face) up to frame 18 (landed, arms flung out, eyes wide), then held to `f_end`; optional stare at a camera object/point. |
| `face_angle(s, f, cam_pos)`, `unreadable_frames(s, f0, f1, cam, max_deg=100)` | Readability check: angle between the head's facing and the lens. |
| `look_at(..., max_yaw, max_pitch)` | New optional clamps (defaults unchanged: 80 / 35 deg). |
| `cup_show(s, f, show)` | Cup visibility keys (CONSTANT). |

### 10.3 Verification

| Check | Result |
|---|---|
| Feet on palm (carrier and hero root vs evaluated palm-surface centre) | 0.0 mm while perched (frames 35, 47); 0.172 m while on its back = the designed back height (0.066 m hero x 2.6) |
| Bird-to-Gary clearance (hero shell, wings, tail, beak, feet vertices sampled 1/7 vs Gary's evaluated hard hat, bib, overalls, belt, pouches, head, body, shirt, straps, pocket, pen, buckles; excluding 0.16 m around the palm; every 3rd frame of the hold) | first version 1 mm (bird head swept across his arm during the un-flip) and 14 mm (head vs hard-hat brim at the cell); after crosswise turn + roll-only flip, low palm waypoint and head-pitch limit: minimum 22 mm (bird vs hard-hat brim while lowering back to the cell), no contact; tool belt and bib never closest |
| S6 helpers (`demo_hold_bird.py -- s6`, exact Gary) | walk_to_mark settles at frame 93; face angle to the lens 55 / 35 / 14 / 34 / 10 deg through walk, look-around and freeze; 0 frames over 100 deg; freeze pose reads (arms out, shock face, looking at the lens) |
| Hand-hold demo | rendered (4 still rounds, 1 full render) and contact sheet viewed: reach, hop out forward onto the palm, raise, crosswise turn, roll onto the back, belly + gold toe pads toward the camera, hold, roll back, back into the cell, glance into the lens; face 3/4 visible throughout |
| Main demo | re-rendered with exact Gary (`demo_characters_20261010.mp4`, stills refreshed); the proxy library rebuilt itself from H's newer hero GLB (10:59) |

### 10.4 Review iterations (hand-hold)

1. Camera behind the Subject (his right is +Y when facing -X): moved; then a pure profile: the Subject now stands at an angle to the cell (0.45 m out, 0.30 m to the side) so a camera on his right sees his face 3/4 and the hand.
2. On its back the bird lay along the forearm with the pull tab at his mouth: it now lies across the palm (beak to his left).
3. Clearance (above): crosswise turn while raised, roll-only flip (no head sweep), palm dips 7 cm right after the landing and approaches the cell from below on the way back, head pitch limit 18 deg.

### 10.5 Progress log

- 2026-10-10 10:30: v2 work order read (DevLog-010 section 9); hold_bird, lens and S6 helpers implemented; demo script with R's real rack (`HOOK_bird_1_0_1`).
- 2026-10-10 10:45-11:00: hand-hold stills (4 rounds: camera, on-back orientation), S6 stills; first full render (hoodie look).
- 2026-10-10 11:00: owner decision (Subject = exact Gary): default outfit switched; hand-hold re-checked with Gary (clearance check added; 3 fixes); final hand-hold render encoded; S6 stills re-checked; main demo re-rendered with Gary; frames and work files deleted.

## 11. Neck deform (owner F1, DevLog-015, 2026-10-10)

Owner (verbatim): "bird head turning should be modeled better, e.g., neck should deform accordingly instead of just head turn and leaving gaps between head and neck".

### 11.1 Design

- Built at rig time in `pigeon_rig.py`; H's files are untouched. Existing APIs and keys are unchanged: callers still key the head empty `<name>_head` (head_pose, head_snap, head_look_at, behaviours, S's own keys). The eye-aim math (pgeom) is unchanged because the head still rotates about the same pivot.
- Each rig gets a neck armature object `<name>_neckrig` (child of the body; armature space = bird frame). The armature data `PGN_neck_arm` is shared and has 5 bones: `neck_root` (static) and `neck_1`, `neck_2`, `neck_3`, `neck_head`.
  - All bones sit at the head pivot with identity orientation, so a bone-local euler equals the head empty's euler.
  - Simple-expression drivers (no Python) read the head empty. Bone i rotates by a fraction of the head rotation: 0.095, 0.31, 0.61 and 1.0. That is pigeon-like, with most of the turn near the head.
  - Bone i also moves by a fraction of the head's offset from the rest pivot (0.25, 0.5, 0.75, 1.0), so the bob and the tier neck extension stretch the neck.
- **Why the bones share the pivot instead of forming a geometric chain:** a chain that bends about lower joints would move the head away from the pivot the API and the aim solver use. Here the bones act as neck segments through their weights. Each segment turns about the head pivot, so neck rings follow arcs, and dual-quaternion skinning (`use_deform_preserve_volume`) avoids shrinkage between segments.
- **Weights** use the position t along the neck (`pgeom.neck_t`):
  - t = 0 on or under the body shell. The distance is measured outside the hero's side profile, sampled from the GLB.
  - t = 1 on or inside the head (a rounded box around the cranium and jaw).
  - Hat weights over t = 0 / 0.25 / 0.5 / 0.75 / 1 map to the five bones.
  - For meshes that are not pure neck skins, a column gate (|y| < 20-27 mm, z > 34-40 mm) keeps everything outside the neck rigid.
- **Hero:** skinned at rig time are skin_back_neck, skin_front_neck, the neck flex ribbons and their coverlays and traces (power/I2C, TX/RX host lanes), clip_neck_lower, thermal_strap_dsp_to_spine, the panel seams, screws, top screw bosses and sockets, and the sternum silk. 17 objects in total.
  - The neck-crossing parts that used to ride rigidly on the head (power/I2C flex and the upper neck clip) are moved back to the body and skinned. Their head ends follow the head bone.
- **Proxy library v2** (`pigeon_proxy.blend`, rebuilt automatically from older versions):
  - A separate skinned `neck` part (the two neck skins, decimated 0.6).
  - `body` and `rest` no longer contain the neck.
  - `make_proxy_rig` adds the neck object and armature for both "full" and "head" modes.
  - The old neck filler tube is used only with a v1 library. A neck extension now stretches the skinned neck instead.
- `neck_deform=False` on `build_hero_rig` and `make_proxy_rig` restores the old static neck.

### 11.2 Verification (`scripts/anim/test_neck.py`)

Gap metric:
- Take the rest-pose vertices of the head shell (cranium + jaw; proxy head) that lie within 2.5 mm hero of the neck surface; the same for the body shell.
- In each pose, the gap is the largest distance from those vertices to the posed neck surface, in hero mm (world = x2.6).
- At rest it equals the 2.5 mm selection threshold, so "no gap" reads as about 2.5.

| Rig | Poses | Head-rim gap, deform | Head-rim gap, static neck (old) | Body-rim gap, deform |
|---|---|---|---|---|
| Hero | yaw +-110; yaw +-150 roll +-35; roll 60 nod 20; yaw 100 roll -40 + bob | 2.47-2.54 | 8.8-9.8 | 2.48-2.50 |
| Full proxy | same | 2.49-2.60 | 8.4-10.0 | 2.49-2.59 |
| Head-only proxy, neck_ext 0.046 m | same | 2.49-3.92 | 13.8-20.3 | 2.49-2.58 |
| Head-only proxy, neck_ext 0.126 m | same | 2.50-7.02 (the 7.02 is at yaw 150 deg; 5.1-5.5 at yaw 110) | 43.5-50.7 (head floats without the old filler) | 2.49 |
| 20 scheduled head-only birds (tier extensions 0 / 0.046 / 0.126 m), 5 birds x 15 frames | behaviour schedule, 90 frames | max 3.36 | - | max 2.51 |

- **Visual (sheets in `outputs/assets/characters/neck_test/`):**
  - With the static neck, the neck top separates from the head at yaw 110-150 deg, and on the hero the internal flexes, connectors and straps poke out.
  - With the deform, the neck stays attached and twists with the head. Twist folds are visible on the decimated proxy at 110-150 deg.
  - At the tier-2 extension the neck reads as a long, thick goose neck, which is the geometry R's 0.126 m implies.
- **Cost:** for 220 rigs (200 head-only + 20 full), the build takes 13.9 s instead of 0.1 s (one depsgraph update per neck armature) and frame evaluation 16 ms instead of 4 ms. The 20-bird test evaluates at 1 ms per frame.
- **Regression:** scheduler self-test unchanged (0 aim errors, 0 limit violations, 0 line-of-sight violations, deterministic). The demo build and the hand-hold demo run unchanged: feet on the palm stay at 0.0 mm and the minimum bird-to-Gary clearance at 22 mm. Lint is clean. Both demo videos and stills were re-rendered with the neck rig.

### 11.3 Open points

- With a 0.126 m extension at yaw beyond about 110 deg, a 5-7 mm hero (13-18 mm world) seam opens behind the head. The rack birds' yaw limit is 100 deg. A smaller extension, or a higher top rotation fraction, would reduce it.
- The neck is thick when stretched to the tier-2 extension. If the owner wants a slimmer extended neck, the bones could also scale the cross-section.
- I did not count whether the neck surface tucks into the head box. The test prints a "pen" count, but it is informational: a neck tucked into the head is hidden, not a gap.
- Proxies show crease folds at extreme twist, coming from the decimated neck mesh.

### 11.4 Progress log

- 2026-10-10 11:20: F1 received. First weights used seam planes, which wrongly marked the back body shell and the jaw as neck. Replaced them with distance fields from the measured body profile and a head box.
- 2026-10-10 11:28: neck armature, drivers and skinning implemented (hero at rig time, proxy library v2). First test: toe flexes were selected for skinning, so a z gate was added. Gap metrics and close-up sheets viewed.
- 2026-10-10 11:35: 20-bird crowd test (camera re-framed once), scale test, regression; both demos re-rendered; frames deleted.

## 12. v3: feet contact and Gary goes about his day (DevLog-018, 2026-10-10)

Owner (verbatim): "fix the shelves the pigeons are standing on, most of them feet are not making good contacts." and "gary should be unaware and goes abt his day instead of getting scared."

### 12.1 Feet on shelves: measurement and fix

- **Measurement** (`test_feet.py measure`):
  - Set-up: head-only proxies at the 239 crowd `feet_origin`s of `hall_layout.json`. The bird racks are instanced as in the layout. Every toe-sole vertex is ray-cast down onto the rack geometry.
  - Every bird's lowest toe sat 1.39 mm below the floor surface (tier 0: 1.55 mm). The floor top with its gold fingers is above the hook height.
  - Tier-0 birds had 74 % of their toe soles over empty air beyond the tread front edge: feet origin 30 mm behind the face, toes 79 mm forward at x2.6. This is the visible "feet not making good contact". Tiers 1 and 2 were fully supported.
- **Fix:** new `pigeon_rig.place_on_floor(r, f, surfaces=[rack objects] | bvh=...)`.
  - It ray-casts under every toe-sole point. If less than 98 % has floor under it, the bird slides straight back in 2 mm steps (at most 0.10 m). Then the root height is set so the lowest sole point touches the highest floor point under the sole.
  - `keep_head=True` adds the back-shift to the neck extension, so the head stays where R placed it (the neck deform handles the stretch).
  - Instanced racks are expanded into one cached world BVH (`surfaces_bvh`).
  - Result (`test_feet.py fix`): tier 0 moved back 48 mm (neck +48 mm) and up 1.55 mm; tiers 1/2 up 1.39 mm, no shift.
  - Over all 239 birds, the lowest-toe gap is 0.00 mm (min = max), 100 % of every sole is supported, and 0 birds fail.
  - Close-ups `feet_test/sheet_measure.png` (before: toes over the tread edge) and `sheet_fix.png` (after: toes on the gold-finger floor, head still out of the cage).
- **Street perches:**
  - `plant_on_surface(r, f, surfaces)` is for post tops, ledges, signs and AC units. It sets the height only by default; `allow_shift` slides the bird back off an edge.
  - `plant_on_wire(r, f, p0, p1, radius)` is for wires and railing tubes. The bird stands across the wire, with the wire under the middle of the toes, and the closest sole point is dropped onto the wire surface.
  - Demo check: post 0.00, railing 0.00, wire 0.01 mm.
- **Behaviours:** `behaviors.spec_from_rig(bid, r, **kw)` builds a BirdSpec from the placed rig (position, yaw, scale, neck extension), so the scheduler aims from the corrected heads.
- **For S** (crowd loop in `build_film.py`):

```python
bvh = PR.surfaces_bvh([racks[rid][0] for rid in bird_rack_ids])   # instanced bird racks (+ the hero rack copy)
r = PR.make_proxy_rig(bid, lib, tuple(pos), yaw, mode="head", neck_ext=ext)
PR.place_on_floor(r, 1, bvh=bvh)            # before rest_pose / any head keys
PR.rest_pose(r, 1)
specs.append(BH.spec_from_rig(bid, r, group="rack", blocked=bl))
# street: PR.plant_on_wire(r, 1, p0, p1, radius) / PR.plant_on_surface(r, 1, [perch_obj]) then BH.spec_from_rig(...)
```

### 12.2 Gary's everyday actions (subject.py; existing APIs unchanged)

| Function | What |
|---|---|
| `walk_sipping(s, f0, pts, sips=1, speed=0.85, **walk_kw)` | Relaxed walk carrying the cup upright in front of the hip (no arm swing on that side), with sips spread along the path; the legs keep walking (overlay on the right-arm IK). |
| `carry_cup(s, f0, f1)` | The carry pose alone. |
| `glance_phone(s, f0, dur=60)` | Left hand brings a phone (new prop, `make_phone`, hidden until used) up in front of the chest with the screen toward his face; he reads it; it disappears again. |
| `check_watch(s, f0, hold=24)` | Left forearm across the chest, back of the wrist up, looks at a wristwatch (new prop, `make_watch`, bone-parented to the left forearm). |
| `tap_rack(s, f0, point, side="L", taps=2, look_max_yaw=40 deg)` | Casual palm knock on a rack/door; the head only glances (yaw clamped) so the face stays readable. |
| `stretch_yawn(s, f0, hold=20)` | Left arm stretches overhead, chest arches, head tips back, jaw opens. |
| `idle_hum(s, f0, f1, hz=0.75)` | Humming sway (head roll/yaw, chest sway) layered on an idle or a walk. |
| `stroll_unaware(s, f0, pts, phone=True, sips=1, hum=False)` | S6 replacement for `startle_freeze`: walk while sipping, phone half way, optional humming. `startle_freeze` stays callable but is marked deprecated for the film. |

- Internals: the arm IK overlays (`_hand_target`, `_arm_action`, `_palm_hand`) now take `side="R"|"L"`. `sip_coffee` gained `from_pose` / `from_pole`, so a sip starts and ends in the carry pose instead of the hanging arm. `hold_bird` (S2 maintenance) is unchanged.
- Demo `demo_gary_casual_20261010.mp4` (540x674, 10 s):
  - Gary walks in sipping, stops turned toward the lens, knocks on a bird rack, checks his watch, reads his phone while humming, stretches and yawns.
  - Four street proxies watch him from a post top, a railing and a wire.
  - Face check (`unreadable_frames`, step 5): 0 frames over 100 deg.
  - Viewed as a contact sheet plus full-resolution end frames.
- Review iterations:
  1. The rack faced away (side panel occluding the shot), so it was rotated to face the camera.
  2. The post bird was not planted (ray reach shorter than its 10 cm start offset); the reach is now 0.5 m for surfaces.
  3. Wire/rail contact was 1.2-1.6 mm off (no sole vertex exactly over the wire); a measured drop correction was added.
  4. The knock turned his face away for about 1 s (frames 106-131). The glance is now clamped, and the demo ends his walk turned 45 deg toward the lens.
  5. The clip was extended from 280 to 300 frames so the yawn finishes.

### 12.3 Open points

- `place_on_floor` is a per-call ray test. The 239-bird fix pass, including the BVH build, took about 4 s (scene set-up excluded; log `outputs/logs/charC_feet_fix_*.log`). The BVH must include every surface the toes could touch; it does not include other birds.
- Pigeon toes are rigid meshes, so on wires the toes do not wrap. The bird stands across the wire with the closest sole point on it.
- The demo's 45 deg end turn uses a root-yaw ease (slight foot skate over 12 frames). For 90/180 deg use `turn`.
- The phone and watch props are simple boxes/discs (no logos).

### 12.4 Progress log

- 2026-10-10 15:10: v3 items received; DevLog-018 and build_film.py crowd placement read.
- 2026-10-10 15:15: feet measured on all 239 slots (sunk 1.39-1.55 mm; tier-0 toes 74 % over the edge); place_on_floor / plant helpers / spec_from_rig implemented; fix verified (0.00 mm, 100 % support); close-ups viewed.
- 2026-10-10 15:30: casual actions + props implemented; demo stills (2 rounds), full renders (3; staging and length fixes); regression (hand-hold demo, scheduler self-test, lint) passed; frames deleted.
