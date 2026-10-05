# DevLog-003-scene-s04: S4 "CPO: keep it scorching hot" (film 30-40 s)

## v1.2 pass (2026-10-03): v2 characters, motion_v2, critique fixes (DevLog-005)

Build: `FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s04_cpo.py -- scenes/v1/s04_cpo.blend`. v1.1 script, devlog and blend backed up to the session scratchpad (`p2_s04/backup/`).

### Plan / checklist
- [x] Characters: `characters/gary_v2`, `characters/manager_v2`; Gary `hold_plate` (v1 action on the v2 rig) then motion_v2 `shot_hit_fall` at the bang (frame 0 = blast, hit-stop 0-4 inside); Manager motion_v2 `gun_raise_aim_fire` trimmed to raise_start (frame 6) at 1.6x so the shot event (frame 44) lands at 8.9, then `flinch` at the head-egg hit 9.56; faces keyed on the roots (face actions off)
- [x] Subtitles: `asm.narr_vo(4)`; HUD built with `FILM_HUD=0`
- [x] 31.8-32.8 smoke: 5 small, low, short-lived puffs beside the package (x +-0.6 m world, z 0.07, GN overrides Count 7, Life 0.75 s, Scale End 0.16, Fade 0.85, intensity 0.8); times 1.5 + 0.1 q unchanged (steam_hiss cues)
- [x] 33.4-34.5 PIC: chip scale 0.0062 -> 0.017 and the rig centred between bus rows 0 and 1 (about 140 um of the 500 um chip in frame, 8-10 rings); the asset's white-band emission drivers removed; per ring a heat value (fast rise when the front passes, 85 ms decay) keys emission colour cyan -> orange-red at equal luminance (no lighting sweep), ring scale wobble 4.5 percent at 11 Hz while hot, substrate slabs take a warm tint; bus waveguides faint cyan emission; pulse times unchanged (hum cues 33.37 / 33.73 / 34.09)
- [x] 30.0-31.3 haze: dark warm slate floor under the board
- [x] 31.4-32.0 cropped 3D bar labels removed (the LAB line names the bars); "XPU: 4 GPU DIES + 16 HBM" smaller, shown 1.0-2.15
- [x] 34.5-35.5 OE heat colour: S4 copy of NG_heat_glow (ramp cyan -> orange -> deep orange-red), lid metallic 0.85 -> 0.25, base near black, Strength Max 1.0 (was pink: the metallic lid mirrored the pale sky)
- [x] 35.5-38.0 BREAKFAST moved off the OE row (own overlay layer BIG_LOW at y +1.55, over the XPU lid); overcook Brown Extent 0.77, cook ramp 0.45 s
- [x] 38.2 hard cut -> whip pan: 8.05-8.2 accelerating pan right out of the OE macro, 8.2-8.5 decelerating pan into the lab; scene motion blur on (shutter 0.5)
- [x] 38.3-40.0 lab rebuilt: warm wall behind the whiteboard, low-contrast floor, daylight rig at 0.45, warm key spot and cool rim spot on the two characters; two-shot (lens 50, about 40 percent frame height), Gary three-quarter to the camera (yaw cheated 78 percent towards the camera) so his face is never in profile, his fall stays in frame; Manager faces Gary exactly (barrel on Gary); 16 eggs leave the plate at 9.144, flight orientation keeps the egg top to the camera (fried egg reads, no brown coins), FRY 0.20, exaggeration 2.8x; head / near shoulder / chest hits at 9.56 / 9.64 / 9.72, 13 on the floor around the Manager 9.60-9.90; Manager flinch + shock -> anger, flush, small ear steam; camera still from 9.5
- [x] Motion blur: scene shutter 0.5 with `motion_blur_position = START` (shutter [f, f+0.5]); the caption holder and PIC rig scale keys are CONSTANT per frame. With the default CENTER position every camera cut and caption switch smeared one frame (seen at 1.37 s in the first draft)
- [x] PIC look: oxide/cladding were near-mirrors (roughness 0.05-0.1) reflecting the pale sky = the v1.1 haze; all PIC materials matte (roughness >= 0.55, specular 0.12), darker oxide tint, ring cores dark, hot ring colour (1, 0.15, 0.01) x 1.35 (green above about 0.3 clipped to peach/pink)
- [x] Shot A lower and tighter (lens 25 from 0.86 m): board fills the upper two thirds, OEs fly in from the frame edges; first 0.5 s slow (T3)
- [x] Shotgun scaled 0.8 (critique: barrel reached past Gary's face)
- [x] Draft render `outputs/v1/s04_cpo_v1_2_draft_p50.mp4` (540x674, 300 frames) reviewed: 0.25 s sheets, 0.1 s strips at 1.5-2.0, 3.5-4.0, 8.0-8.5, 8.8-9.3, 9.5-10.0, single frames, per-frame YAVG

### Measured
- Draft p50 (render_all, 8 spp, comp cartoon, motion blur on): 232-255 s for 300 frames = 0.77-0.85 s/frame including process start; stills 0.7-1.9 s (lab with 16 eggs 1.3-1.9 s, 9.95 s 3.3 s once), first frame of a process about 5.5 s.
- Blend 134 MB. Per-frame mean luma 92-160 (8-bit); frame-to-frame jumps above 8 only at camera cuts (1.4: +17, 2.4-2.5 dolly: +15/-13, 5.4 board -> OE egg shot: +48, 8.1-8.2 whip: -11..+14). No periodic flicker. First 15 frames 92-94, last 15 frames 135-138 (calm for T3/T4).

### SFX retime
None. Kept: OE touchdowns 1.20-1.46, smoke 1.5-1.9, pulses 3.52 / 3.88 / 4.24, OE egg splats 5.84-7.77 (t_splat formula unchanged), curve_fall 8.45, bang 8.9, eggs leave the plate 9.144 (Gary's shot_hit_fall hit_fling is frame 7 = 9.133), head / shoulder / chest 9.56 / 9.64 / 9.72, floor landings t_rel + 0.46 + 0.04 (k mod 4) = 9.60-9.89 (which egg goes where changed with the random sequence; the range is the same). Suggested new cues (not added, audio owner): Gary ground hit 39.63 (shot_hit_fall ground_hit frame 22) clay_thud; Manager ear steam 39.62 steam_hiss; Manager flinch 39.56 is under the head splat.

### Framework requests
1. `asm.big` / overlay layers with a position argument (S4 defines `BIG_LOW` / `BIG_HIGH` by adding entries to `L.OVL` at runtime).
2. `blender_lib.shot` / HOLD scale: key CONSTANT per frame (or document `motion_blur_position = START`) so motion blur never smears captions at cuts; other scenes with motion blur on will show one ghosted caption frame per cut/caption change with the default CENTER position.
3. `render_all.py` concatenates a full film whenever every scene mp4 of the version exists; my S4 run therefore wrote `outputs/film_v1_2_draft_p50_20261003.mp4` mixing whatever v1_2 drafts existed at 09:29 (not reviewed by me).

### Remaining problems
- 5.4 s cut from the dark board pull-back to the bright OE egg shot (+48 luma in one frame; pre-existing cut C21, not a flicker).
- OE fly-in (0-1.4): the OEs are still small light squares (OE scale bound by the hook pitch); readable as parts flying in, not as OEs in detail.
- Fried eggs on the OEs: the cooked ones read as dark brown discs with a light crescent at draft; the last right-column eggs are still fresh when the whip starts (8.05).
- The whiteboard curve labels are small at this distance; the graph drop itself reads.
- Lab set (wall, floor, light) differs from the v1.1 S2 lab (blue checker sky); S2 should adopt the same wall/floor for continuity (cross-scene request).
- Manager's flinch keeps the gun raised; his face at 9.6-10 is partly covered by the head egg and arm.
- Not done: fresh-context audit.

### Progress log (v1.2)
- 2026-10-03: read briefs, critique, feedback, transitions, character/motion docs; backups; script edits; first build OK. Lab overexposed at first (extra area lights + full daylight rig: bloom over the whole frame) -> rig at 0.45, spots instead of a wide area key (the area key lit the wall in a hard trapezoid and blew out the whiteboard). Lab framing iterated three times (Gary's fall left the frame on the right). OE lids pink -> orange. Plate release flight removed (a spinning plate read as a white ball): the empty plate stays in Gary's hands.
- 2026-10-03: first draft render reviewed: caption ghost at cuts (motion blur CENTER) -> START + constant overlay keys; PIC haze traced to glossy oxide/cladding -> matte; hot rings pink -> deep red; shot A tightened; BANG moved above the heads (own layer BIG_HIGH); gun 0.8. Final draft rendered and checked. All Blender runs after 09:05 through `scripts/film_v1/bslot.sh`.

## v1.1 pass (2026-10-02): feedback rows 0:31, 0:33, 0:34, 0:36, 0:38, 0:39 plus the known v1.0 issues

Files: `scripts/film_v1/s04_cpo.py` (rewritten, no monkeypatch, no `L._VIS.clear()`; asm.py already contains the object-wise visibility, `clone` animation clear and `new_scene(hold_z=...)` fixes, so the v1.0 rebuild hazard is gone), `scenes/v1/s04_cpo.blend` (132 MB), `assets/generated_textures/v1/s04/graph_s4/` (whiteboard graph sequence), `assets/generated_textures/v1/s04/s04_contact_sheet.png` (12 stills at t = 0.6 1.2 2.0 3.2 3.6 4.0 4.7 6.0 7.9 8.6 9.5 9.95 s, 540x675 each, tiled 4x3 at 360 wide). Previous blend and script were backed up to the session scratchpad.
Build (about 10 s): `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s04_cpo.py -- scenes/v1/s04_cpo.blend`.

### Checklist v1.1
- [x] 0:31 OE path: each OE lifts 0.30 m, swings around the frame (alternating y arc, yaw/roll rock), descends with smootherstep timing and a damped settle (bounce + rock); keys per frame (deterministic function `oe_path`), 0.035 s stagger, first camera tilted (22 deg off vertical) so the lift is readable
- [x] 0:33 no black quad: the PIC is a camera-attached rig (PIC_RIG, child of CAM) whose materials are wrapped in Mix(Transparent, original) driven by PIC_RIG["p_alpha"]; the camera zooms onto OE R_4 (2.6-3.25, lens 55 to 85, OE visible about 3.1-3.3), alpha cross-dissolve 3.15-3.5, dissolve out 4.5-4.9 while the camera pulls back (lens 85 to 28). PIC lights are linked to the PIC objects only (light linking works in EEVEE Next 4.2) and keyed with alpha squared. Scene start is the oblique board and the end is the lab, both usable for the coordinator's boundary cross-dissolve; no black frames anywhere.
- [x] 0:34 three distinct heat pulses (3.52-3.78, 3.88-4.14, 4.24-4.50, flat gaps; p_wave_pos sawtooth), colour heat on rings and slabs, plus a rig wobble (rotation and scale ripple at 7-9 Hz, enveloped on each pulse) as the distortion cue. The camera is locked (no keyed motion, no noise) from 3.25 to 4.5 so nothing moves in the camera while the pulses run; no global light change.
- [x] 0:36 closer on the actual OEs: camera 0.17 m outside the column, 0.25 m high, lens 38 (left column 5.4-6.85, target walks with the landings), then the right column (6.85-7.65) and a pull back along the column (7.65-8.2, lens 38 to 32). Eggs: clay ovoid ("shell") tossed in from the far end of the column on a ballistic arc (g = 9.81, flight 0.44 s, spin), first contact on the OE lid, hop (restitution 0.30, horizontal velocity x0.05), second contact = splat: the fried egg appears, shape key `spread` follows a damped spring (overshoot 1.15), yolk wobble spring `p_wob` (6 Hz, tau 0.22 s) drives `yolk_dome`. Landings: left column 5.62 + 0.15 r, right 6.85 + 0.10 r.
- [x] 0:38 overcook: Brown Extent 0.72 (default 0.42), p_cook ramps 0 to 1 over 0.6 s from the splat; at p 1 only a small white crescent next to the yolk stays white (tested with BE 0.42/0.75/1.0/1.3/1.8; p_cook above 1 breaks the shader into dark translucent). Sizzle steam: one `steam` puff per egg and a second on every 2nd egg (scale 0.06; the v1.0 steam was placed in hardware units, i.e. at 1/6 of the right position)
- [x] 0:39 lab bench setting (as S2 layout offset to x = +30 m: bench + scope showing the last S2 eye frame, whiteboard, Gary, Manager with shotgun): whiteboard graph runs back DOWN along the S2 curves (progress 1.0 to 0.2 over 8.45-9.85, tips and labels ENERGY / BIT and LATENCY follow as in S2). Gary carries a plate with a stack of 16 fried eggs; all 16 are thrown at the release frame (8.9 + 0.244 s, staggered 0.012 s): 1 on the Manager's head (hit 9.56), 1 on the shoulder nearer the camera (9.64), 1 on the chest (9.72, sticks and slides down 0.16 m), 13 on the floor around him in a ring (two damped bounces, e = 0.34, slerp to flat, yolk wobble). Eggs scale 1 to 3.5x within 0.16 s of release (cartoon exaggeration, documented). Thrown eggs are fried (Brown Extent 0.5, so white stays visible at 4 m). Camera: one side-on position, slow push, no cuts (Gary near/right, Manager far/left, whiteboard behind).
- [x] Known v1.0 issues: right-column eggs now land inside the right-column shots; OE glow Strength Max 1.5 (was 1.3 and faint at distance; 2.0 and above washes the eggs through the comp bloom); XPU die emission driver `v*4` capped to `v*1.2`, heat-glow group Strength Max x0.4, steel retention frame roughness 0.6 / metallic 0.6 (the huge white specular blob came from it); dolly-zoom shake added (noise F-curve modifiers, 0.06 m at 6-frame scale on camera and 0.6x on the target, restricted to 1.4-2.6 s); caption holder scale re-keyed per frame from the lens curve
- [x] Cross-cutting: eased (bezier) camera moves, handheld noise per shot (0.0035 to 0.010 m, none during the PIC pulses), everything deterministic and baked (random.Random(21), per-frame keys, no simulation caches), motion blur is a render-preset setting (hero) and needs nothing in the blend

### Performance (draft preset 8 spp, 50 percent, comp cartoon, M-series, measured per still)
0.6 to 1.3 s typical (t = 1.0 0.85, 2.0 0.59, 3.5 0.74, 3.7 1.16, 5.3 0.83, 6.7 0.66, 7.6 0.81 to 1.54 before halving the steam puffs, 8.15 0.9 to 1.26, lab 8.8 1.0, 9.3 0.96, 9.97 1.2 to 1.5); first frame of a process 5.2 s (shader compile). Blend 132 MB.

### Needed framework changes / limits (not done by me)
1. True refraction heat shimmer is not possible without a compositor Displace node (EEVEE Next refraction on a thin plane is dark at draft, no ray tracing). `render_scene.py` replaces the compositor tree with NG_comp_post, so an in-scene Displace is lost. Requested: an optional shimmer pass in `render_scene.py` (Displace driven by a noise texture masked by p_alpha / pulse envelope). Current cue: rig wobble + colour heat.
2. The comp bloom (cartoon 0.6) spreads the OE lid glow over the eggs at draft: if more glow is wanted, lower the egg-phase bloom or add a per-scene comp preset.
3. The lab zone reads the S2 eye texture (`assets/generated_textures/v1/s02/eye_s2/eye_s2_0300.png`) by relative path: S2's textures must exist.

### Remaining problems
- Raw-looking (glassy) eggs are visible at the very end of the right column (last egg splats at about 7.75 s and finish cooking at about 8.3 s); the pull-back at 7.65-8.2 shows the early overcooked ones with them.
- The PIC dither noise is visible at draft (8 spp on dithered transparency) and clears at standard.
- At 4.5 m the plate stack of 16 eggs on Gary's plate is small and partly hidden by his body; the bounce/splat of the eggs on the shirt and head is only legible in motion.
- Steam puffs are tiny at hardware scale; the sizzle reads mainly from the second puff.
- S5 start framing was not checked (coordinator).

### Progress log (v1.1)
- 2026-10-02: plan from DevLog-004; first build; fixed pulse key collision (two keys on one frame), blown-out XPU/frame glow, camera inside the retention frame, steam placed in the wrong units, thrown ovoids passing through the camera (now launched from the far end of the column), lab camera behind Gary (plate hidden) -> side view, egg overhang (EGG_OE 0.5), overcook parameters, dissolve timing so the OE is visible before the PIC; 12-still contact sheet viewed; timing measured; devlog written.

---

# v1.0 notes (kept for reference; the hazards listed below are resolved in v1.1)


Owner: S4 assembly agent. Date: 2026-10-02. Blender 4.2.3, EEVEE Next, standard preset.
Files: `scripts/film_v1/s04_cpo.py`, `scenes/v1/s04_cpo.blend` (72 MB), contact sheet `assets/generated_textures/v1/s04/s04_contact_sheet.png` (8 stills, 540x675, t = 0.8 2.0 3.2 4.1 5.0 6.6 8.6 9.75 s).
Build: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s04_cpo.py -- scenes/v1/s04_cpo.blend` (about 4 s).

## Checklist
- [x] hardware group: hgx_baseboard S4 variant (hgx board / OAM variants hidden, s4_board + socket pads + nested XPU shown), 16 OEs, 16 eggs
- [x] OE slide-in 0.2-1.4 s, staggered 0.03 s (right side +0.015 s)
- [x] dolly-zoom with bars, p_heat on XPU root, smoke_puff x5, heat_shimmer
- [x] continuous zoom 2.6-3.45 s onto OE R_4, fade quad, PIC close-up (p_wave_pos -0.3 -> 1.3 over 3.5-4.6), fade back
- [x] pull-back, per-OE heat glow (own material copy, NG_heat_glow, root p_heat driver), 0.03 s stagger
- [x] 16 eggs (own mesh copy + own white/yolk material copies with NG_egg_fry, root p_cook keyed, drivers on materials and shape keys), drop, steam on every 2nd egg
- [x] human zone: Gary hold_plate -> throw_up -> topple_back, Manager aim_gun + shotgun, muzzle_flash + smoke_ring, 3 eggs thrown and landing on the Manager's head by 9.75 s, hole schedule
- [x] captions, narration chunks, labels, FX notes, timecode copied from the crude s4
- [ ] camera shake on the dolly-zoom spikes (crude only had the note; not implemented)
- [ ] compositor bloom (NG_comp_post) not applied; glow is plain emission

## Scale decisions
- Hardware group empty `HW_GROUP`, scale M = 6 (board 1.56 x 1.2 m, package 1.08 x 0.6 m). Why 6: egg is about 57 mm raw / 100 mm fried in the prop; eggs are scaled 0.11 on the group (white about 11 mm in hardware units = 66 mm in world) so they match a real egg and read next to the OE lid (9.25 x 8 mm x 6 = 56 x 48 mm).
- OE footprint reconciliation: library OE carrier 24 x 20 mm (lid 18.5 x 16) vs package hook footprint 14 x 10 mm at 11.5 mm pitch (8 sites span 92 mm of a 100 mm package height). Full-size OEs would overlap (20 mm in Y vs 11.5 mm pitch). Decision: uniform OE scale 0.5 -> carrier 12 x 10 mm, lid 9.25 x 8 mm. Origin of the OE root at each HOOK_oe_L_i/R_i (hook rotation 180 deg on the left gives pigtails pointing outward), pigtail (44 mm) shortens to 22 mm. Cleanest asset-side fix would be to change the package hook footprint to 24 x 20 at 23 mm pitch (needs a wider/taller package) or an OE variant at 12 x 10; recommend packaging agent adopt 0.5-scaled OE (or 14 x 10 OE variant).
- Eggs: fried_egg prop (spread/edge_crisp/yolk_dome/bubbles shape keys) with library MAT_vfx_egg_white/yolk; per egg: own mesh copy (shape keys are per mesh), own material copies, root property p_cook keyed 0 -> 1 over 1.3 s starting at landing (5.4 + 0.06 n + 0.25 s); shape-key and material drivers read p_cook. Material uses the radial fallback (no egg_rim attribute on the props mesh); looks right.
- PIC: microring_array_closeup x15000 variant (real-size variant hidden) scaled 0.065 -> chip 0.49 m, own zone x = -8 m with its own macro_studio rig (p_energy 0.3) and floor.
- Humans: real size, plain room (floor checker 1 m, 3 walls) at x = +30 m, daylight rig. The 3 thrown eggs are copies of the plate eggs, scaled up 3.5x in flight (FLY_EXAG) so they are legible from 11 m (documented exaggeration; the plate eggs themselves are real size).
- Cameras: crude coordinates times k = 0.0659 (package 16.4 crude units = 180 mm x 6 = 1.08 m); top-down shot raised x1.3, left/right column shots x1.15 in X; zoom end and PIC shots re-derived for the real geometry. Human shot re-derived (crude camera was only 3 m from the action in my first try; final: lens 32, 11.2 m back, tilt up 9.0-9.5 s and back by 10 s).

## Holes (Gary), T_now = 30 + t
hole 1 (9.2 s), hole 2 (19.2 s) and head hole (28.95 s) start at max(0.3, 0.9^((30 - T_shot)/1.5)) and step x0.9 every 1.5 s with CONSTANT keys down to 0.3; hole 4 keyed 0 -> 1.0 at the bang (8.9 s). Hole 3 and 5 unused in S4.

## Asset / framework problems found (workarounds are in s04_cpo.py; asm.py not edited)
1. `asm._finalize_collections` / `asm.show` cannot work in Blender 4.2.3: `Collection.hide_render` is not animatable ("property not animatable" error at finalize). Workaround: monkeypatch in s04_cpo.py keys every object of the collection tree (L.VA + L.finalize_visibility). Needed fix in asm.py: do that. Every scene that calls asm.show or asm.fx will hit this.
2. `asm.clone` uses `Object.copy()`, which shares animation_data/actions: all 16 OE clones moved with the first one. Workaround: `root.animation_data_clear()` on each clone before keying. Needed fix: clear animation data in clone.
3. `blender_lib.HOLD_Z = 0.8` puts captions and the fade quad 0.8 m from the camera, behind any geometry closer than that (macro shots at 0.1-0.5 m: captions got cut, fade quad did not cover). Workaround: set `L.HOLD_Z = 0.06; L.HOLD_S0 = HOLD_Z/3` before `asm.new_scene`, camera clip_start 0.01. Needed fix: make HOLD_Z a new_scene parameter.
4. `blender_lib.shot` keys the overlay HOLD scale linearly between the two ends while the lens eases (bezier): captions drift off-screen in zooms. Workaround: re-key HOLD scale per frame over the zoom shots (end of s04_cpo.py).
5. `L.fade_quad` material reflects the sky (navy instead of black); I set Specular IOR Level 0 on it.
6. hgx_baseboard: `SOURCES` collections are left visible-looking but did not show in renders (OK). The nested XPU is parented to HOOK_xpu_origin; the OE hooks have identical names in other appended assets (Gary and Manager both have HOOK_head_top; second gets `.001`): look up by object membership, not by name. HOOK_xpu_origin and OE hooks are consistent with the S4 board (OEs land on the package substrate outside the lid ring, inside the socket frame).
7. materials_vfx smoke_puff at scale 0.5 still reads large next to the dies; heat_shimmer is barely visible.

## Performance (M-series, EEVEE Next standard 24 spp, 1080x1350, measured while other agents were rendering)
2-4 s per frame typical (top-down 2 s, egg scenes 2.8 s, human 2.6-4 s), first frame of a process +4-6 s (shader compile). Worst measured 4.1 s (7.6 s, 16 eggs + steam). Scene is 2.6 M tris with all 16 OEs (156k each as evaluated) but collection/object visibility windows hide the PIC zone, human zone and fx outside their time windows.

## Progress log
- 2026-10-02: scene assembled, 8-still contact sheet viewed and fixed (OE clones sharing keys, overlay plane distance, fade quad, gun orientation (gun root rotated 180 deg about Z on HOOK_gun_grip_R), camera for human shot, egg tilt on the Manager's head).

## Open questions
- Keep OE scale 0.5 (carrier 12 x 10 mm) or change the package hook footprint?
- Egg exaggeration (3.5x) for the thrown eggs: OK?
- Camera shake on the dolly-zoom spikes and bloom: add in the coordinator render pass (NG_comp_post)?
