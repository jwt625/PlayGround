# DevLog-005 (v1): visual critique of film_v1_1 (draft p50, with VO and SFX)

Date: 2026-10-02. Reviewer role: animation / VFX supervisor, read-only on the project. Scope: `outputs/film_v1_1_draft_p50_20261002.mp4` (540x674, 30 fps, 62.000 s; the `_vo_sfx_` file carries the same video stream, h264 + aac 48 kHz stereo).

## 0. Method and limits

- Frames every 0.25 s for the whole film (248 frames, 8x5 contact sheets per scene), plus consecutive-frame strips (every 1-5 frames) at: S1 0-1.2, 3.2-4.4, 6.6-9.9; S2 9.7-10.3, 15.0-15.8, 18.9-19.9; S3 21.6-24, 25-27.2, 28-30; S4 30.8-33, 32.7-34.3, 38-40; S5 40-43, 43.2-45.5, 47.5-50; S6 50.8-53.3, 57.2-60; each cut boundary as a frame pair. Full-size (540 px) looks at 9.3, 10.1, 6.8, 12.9, 11.0, 13.5, 20.3, 22.4, 27.6, 36.8, 38.5, 44.0, 47.9, 49.3, 55.0, 59.6.
- Timestamps are film seconds. Where only a 0.25 s sheet was used the value is good to about +-0.25 s; where a per-frame strip was used it is good to 1-3 frames. Luma numbers are ffmpeg signalstats YAVG (8-bit, 0-255) on the draft.
- Style reference check: the supplied `references/JE3LSo1Rp54.mp4` (2160x3840, 29.97 fps, 50.6 s) shows, in my 1 fps sheet and 4 full frames, a glossy CG Fuji cable build (saturated sky gradient, green/earth ground, strong hero object, bold white caption with dark shadow). I see no clay characters and no comedy beats in it, so the clay and humour comparison cannot be made from this file; the comparison below is lighting, colour, camera, caption and pacing only. Scene-change detection (threshold 0.18, 270 px): 8 changes in 50.6 s (11.1, 11.2, 21.9, 24.9, 25.5, 27.0, 27.4, 47.6) versus 49 flagged in our 62 s (some are blur/flash false positives, but the order of magnitude is clear).
- Not re-reported: items Wentao already flagged in DevLog-004 are listed only where they are still wrong in v1_1. Holes are visible on Gary only at 20.3 (dark patch at hip) and 55.0 (hole in the lower back); not visible at 10.1, 38.5, 47.9, 49.3 (side or back views, so not judged an error, but see S5).
- Scratchpad frames: `/private/tmp/claude-501/-Users-wentaojiang-Documents-GitHub-PlayGround/36a0a473-d33f-4dfa-9d16-9644ea7bb16f/scratchpad/critic/` (sheet_S1..S7.png, k_*.png strips, bsheet.png boundary pairs). Nothing in the project was edited.

Cross-cutting findings that recur in every scene table below (listed once here, referenced as X1-X6):

| ID | Finding | Severity |
|---|---|---|
| X1 | Burned-in white subtitles (`asm.narr` layer "CAP" in each `s0N_*.py`) are the OLD DevLog-001 narration, not the new spoken text in `scripts/audio/make_vo.py` SEGS. Examples: 12.1-14 subtitle "Gary puts a chip on every connector" vs VO "So he puts a re-timer on every connector"; 20.2-23.2 subtitle "Gary moves the optics closer to the logic" vs VO "Gary dumps the pluggables and moves optics onto the board"; 53.3-55.2 subtitle "Thousands of them" vs VO "Manager wants it done yesterday, but the right fiber length arrives next Tuesday"; 55.4-57.6 subtitle "Manager wants it done yesterday" while VO is on "next Tuesday". Subtitle timings are also the old 0.2/3.4 windows. | high |
| X2 | Draft HUD still burned in: yellow "[FX: ...]" line top-left (layer "FX"), "S3 0:07" timecode bottom-right (layer "TC"), grey data cards bottom-left (layer "CARD"), cyan "LAB" labels top-left. They eat 15-20 percent of the frame height at phone size. Must be gated off for finals; keep at most one LAB or CARD per shot. S7 also shows the debug line "[VO ~9 words/s: ...]". | medium (draft), high (final) |
| X3 | Caption legibility: light-weight white sans with thin shadow, about 5 percent of frame height. Fails on pale frames (S2 10-20 blue/white checker floor; S4 30-38 pale board; S5 grey room): e.g. 16.0-17.5 "forever to arrive" is near-invisible on the white checker floor. Reference uses a bold white face with a hard dark shadow. | medium |
| X4 | Light discs: ceiling lights in S1 2-3, S3 20-29, S6 are blown white with a dithered black speckle pattern on the discs (visible at 20.3 and 22.4, dots inside the lamp glare). Looks like a bloom/glare artefact or fireflies at 8 spp. S3 sun disc at 25.3-26.9 shows the same speckle. | medium |
| X5 | Humans are small in all wide shots. Measured on the 270 px strips: Gary is about 19 percent of frame height at 28.0-29.5 and vendors about 12-15 percent; Gary at 2.0-3.0 (S1) about 10 percent. At phone size faces, holes, guns and fight poses cannot be read. | high |
| X6 | No scene shares a colour world: S1 neutral grey lab, S2 and S4-end blue-sky infinite checker, S3 blue-grey conference room, S4 pale haze, S5 flat grey, S6 near-black data hall. Reads as six different films; the reference holds one saturated sky-and-earth palette throughout. | medium |

## 1. S1 Copper (0.0-10.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 0.0-0.75 | VFX / camera | medium | Inside-the-cable tunnel: 12-sided faceted rings (visible polygon edges, hard orange/black stripes), almost no camera travel (rings barely scale from 0.0 to 0.7 s), a single white glow ball instead of "pockets of electrons flying left and right". Reads as a static wallpaper. | `s01_cam_fx.add_electrons` n=18 is not readable at this size: use 6-8 larger, brighter, separated pulses with clear travel left and right; raise ring segment count and add a forward dolly of 2-3 tunnel diameters in 0.7 s. |
| 0.75-1.0 | camera / continuity | low | Exit from tunnel (0.8 s) is a 3-4 frame blurred smear into the data hall; the 3D word "COPPER" is cut by the smear and then overlaid by the subtitle "wants faster" at 1.0 (two text layers fighting). | Hold "COPPER" 0.3 s later or drop the subtitle in this window. |
| 1.0-2.0 | readability | medium | Fly-along along backplane cartridge wall: the Pulse courier is not identifiable (only cyan dots at the cable end), cable fan hides the wall. Gary does not appear until 1.75 (about 8 percent of frame height, far right). | Make Pulse a larger clay character in the first 2 s, or cut it; place Gary at 40 percent frame height at the snap-stop. |
| 2.0-2.5 | lighting | medium | Ceiling lamps and white floor tiles overexposed (see X4); lamp discs about 1/3 frame width. | Lower ceiling emission or exposure in `asm` lighting for S1; add the floor reflection falloff. |
| 2.25-4.3 | readability | medium | "5 m / 2 m / 1 m" captions are good beats, but the 3D "PULSE DECAYS" text slides off the right edge (cropped at 3.3-4.3, "PULSE DECA"), and the 14-cable bundle is a thin line between two black racks; decay of the Pulse is not visible. | Keep text inside the 1080 safe square; make the bundle thicker or add a visible amplitude bar on the Pulse. |
| 3.0 and 3.6 | camera | low | The 5 m to 2 m change at 3.0 and the 2 m to 1 m change at 3.6 are hard cuts with one blurred frame (3.6 shows a ghosted Gary double exposure), not the planned "whip pan" and "fisheye push-in". | `asm.shot` with ease and 4-5 frames of directional blur; or accept as punch cuts but remove the ghost frame at 3.6. |
| 4.2-4.3 | human animation | medium | Gary pops from a standing front pose (4.0-4.1) to a walking/lunge pose (4.3) within 3 frames with a blur frame at 4.2; position jumps. | Key a 6-8 frame run-in (`asm.walk` before `STRETCH_T0` = 4.4). |
| 4.4-6.0 | human animation / readability | high | The central gag "stretch the wire one more meter" is not readable: Gary walks sideways with arms out, never touching the rack or bundle; the taut cable bundle is hidden behind the near rack; the caption "STRETCH" sits exactly on his head. Gary's pose is a walk cycle held for 1.6 s (key_loc 4.4 to 6.0 LINEAR, `idle` repeated), so it looks like strolling. | Camera on the cable gap (`key_gap` 1.0 to 2.0) with Gary's hands on a rack handle, strain pose, rack sliding; move caption to the lower third, shorten to 0.8 s. Add the creak and strand pops (SFX 4.4-5.95) on visible strands. |
| 6.0-6.6 | VFX | low | Scope close-up with eye and BER (3.0E-08 to 4.0E-04) reads well. Good beat. | none |
| 6.7-7.5 | human animation | medium | Manager stands in profile at the right frame edge, still for about 0.7 s (6.8-7.5) with a pale shotgun held low that is nearly invisible against the white wall; he never "walks all the way to the scope" in frame (he is already there at 6.77). | Show the walk-in on the cut at 6.77 (start from frame edge), put a darker gun colour or rim light on the barrel. |
| 7.5-8.7 | VFX | low | Ear steam is a cluster of white balls (popcorn look, 7.77-8.9); face goes red at about 7.9 and is a good readable gag. Steam starts at about 7.7, SFX steam_hiss is at 7.55 (see sync). | reduce ball count 30 percent, vary size, add upward drift. |
| 7.8-9.1 | human animation / clipping | medium | The gun is a thin pale bar; at 7.9-8.0 it sweeps up through his own torso and arm (clip). Arm is dead straight from 8.2 to 9.1 for about 1 s with no tremor or breathing. Barrel points off frame left, Gary is never in shot until the bang. | Raise gun along the hip then shoulder; add 2-3 Hz arm tremor; cut to Gary's face at 8.7-9.1 (dread). |
| 9.2-9.5 | comedy timing / readability | high | BANG cut at 9.2 is on the SFX (good), but Gary is only 10 percent of frame height at the left edge behind a star and the 3D word "BANG"; the hole and the fall are not visible; VO says "Poor Gary" at 9.3-10.0 while the picture is already zooming on the scope (blur from 9.45). | Keep the wide for 0.5 s after the bang with Gary's topple readable (hit-stop 3 frames exists, `HIT_STOP`); start the scope zoom at 9.75. |
| 9.5-10.0 | camera | low | Zoom onto the scope (blur, motion smear) is a good match-cut motif, but the final frame (9.9) is a different scope view than the S2 first frame (10.0): hard cut to Gary in profile in a blue-sky world. | See boundary table, B8. |

## 2. S2 Retimers (10.0-20.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 10.0-10.7 | continuity | high | Cut from the grey lab (S1) to Gary + bench + whiteboard in a blue gradient sky over an infinite checker floor; Gary shows no hole (profile view) and stands in a stiff A-pose for 0.7 s with arms dangling. The location contradicts S1. | Reuse the S1 lab set and light, or at least the S1 floor and walls; 2-3 frame breathing idle. |
| 10.0-10.5 | human modelling | medium | In side view Gary's face reads as a black rectangle with a white dot (hard-hat shadow or eye plate seen edge-on); same at 38.3-38.9. Face is not readable in profile. | Check head material normals / eye geometry in the Gary asset, add a light skin tone fill or rim; avoid 90 degree profile on dark face areas. |
| 10.75-12.0 | VFX / hardware accuracy | medium | NARROWCOM sign and valley mark are readable (good), but the "row of connectors" and retimers are tiny blue slabs and gold flashes; the fans drawn as grey boxes with 60 percent height. | per DevLog-004 known deviation; larger chip scale in `s02_retimers.py` chip row. |
| 12.4-15.4 | lighting / technical | high | Retimer rows bloom into blown-out white glare blobs with a glittering speckle on wall panels and chips (13.5 full-size). YAVG shows a 1-frame flash of +20 to +40 at each camera hop: 12.07, 12.47, 12.87, 13.27, 13.7, 14.2, 14.7. The chips are not identifiable as chips; the "more rows with height" idea cannot be counted. | Cap `p_glow` peak (the white-hot branch above 0.55 in `s02_retimers.py` glow function), fewer simultaneous flashes, add a darker backplane behind chips, reduce hop flash by tying glow to arrival rather than the camera hop (HOP = 0.20). |
| 12.4-14.7 | camera | medium | Seven camera hops (HOP 0.20 s at each T_ARR) read as jump cuts, 0.4 s apart; not a continuous rise ("camera rises through 8 trays"). | Ease the rise with a continuous path and dwell at trays by slowing, not hopping. |
| 14.0-15.4 | readability | medium | Top of rack: four bright window panels, rows overexposed; 15.0-15.4 is five frames of nearly identical glare. | Cut at 14.8. |
| 15.4 | camera / continuity | high | "Camera exits the top of the rack and cranes down" is a hard cut at 15.4 from the glare interior (YAVG 141) to a bright exterior (YAVG 195). No crane move connects them; the rack inside is not the rack outside. | Continuous move up through the rack top using `cam_rig.shot(5.4, 6.4)` with the rack roof/skin cut away; or a whip with matched direction. |
| 15.4-16.4 | readability | high | Exterior crane: characters about 8 percent of frame height, whiteboard blank, bench and rack tiny on a high-contrast blue/white checker; floor pattern dominates. Cables from rack to scope are the only story element. | Lower and tighten the crane; floor material: reduce checker contrast to about 10 percent value difference or switch to the S1 tile floor. |
| 16.4 | camera | low | Hard cut to a wide with Gary's back and Manager at the whiteboard; curves are drawn on the board but are thin 1-2 px lines (16.5-18.5). Labels ENERGY / BIT and LATENCY unreadable at this size. | Thicker curves (3x) and larger labels; the board must occupy at least 40 percent frame width. |
| 17.0-18.5 | human animation | medium | Manager at the whiteboard is frozen with one arm up (about 1.5 s); face colour change to red at 18.0-18.2 is readable. Gary is a static back view. | Add a head-shake/foot shift on Gary; Manager marker tapping. |
| 18.5-19.2 | camera / continuity | low | Camera position changes at 18.6 (sheet) and again at 19.2 (cut into the bang); at 18.9-19.0 the gun barrel points screen-left while Gary stands right, the next view shows Manager facing Gary (line crossing). | Keep one side of the 180 degree line across 18.5-19.5. |
| 19.2-19.9 | human animation | medium | Gary's fall: arms out, straight-backward topple onto the checker, lies flat by 19.7. The star hit effect covers the hole location. | Add a hit-stop pose and a folding fall (knees first), Show the hole on the torso for 2-3 frames. |

## 3. S3 NPO (20.0-30.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 20.0-21.6 | camera / comedy timing | high | Planned "snap push-in" is a locked wide for 1.6 s (20.0-21.6, five or more frames sheet-identical): Gary static in A-pose, vendors tiny (about 12 percent height) behind the table; label "NPO MODULE" on top of Gary's torso. Violates "no static shot longer than 1 s". The optical module on the table is a flat green plate. | Push in from wide to table level (module and ASIC visible) in 0.5 s; Gary breathing idle; move the label off him. |
| 21.9-23.0 | VFX / readability | medium | Whip pan to the fight works, but vendors are 12-15 percent height; "LASER: IN PACKAGE / EXTERNAL" labels float in air over the windows and overlap the fighters; the table laser glow is a flat orange smear. | Camera at table level over the table edge; move labels to the table, 40 percent larger fighters. |
| 23.0-24.2 | hardware accuracy / readability | medium | BGA / LGA / fine-pitch packages are green squares with a faint grid; the undersides cannot be told apart; labels "BGA", "LGA", "BGA (FINE)" overlap people. | Show each package at table-top 3x larger with ball/land pads visible; label under the package. |
| 24.2-25.2 | human animation | medium | Fight: grapple poses are lock-ups with little weight shift; no hit reaction, no anticipation visible at this size. Fight reads as arms flapping. | Use bigger silhouettes and two or three clear beats (shove, swing, stagger) per second; hit-stop on contact. |
| 25.3-27.0 | VFX / technical | high | Day/night cycles: YAVG alternates between about 155 (day) and about 90 (night) three times (dips at 25.53-25.83, 26.07-26.33, 26.57-26.83), each transition in 6-9 frames. Reads as a strobe over a cluttered scene; the speckled sun disc (X4); fighters are small and half-lit. Photosensitivity risk. | Reduce day/night contrast (night floor at about 60 percent of day), lengthen each phase to at least 0.5 s or reduce to 2 cycles, soften the sun disc in `s03_sky.key_cycle`. |
| 25.3-27.0 | comedy timing | medium | The three-month gag is carried by the sky only; "+3 MONTHS" text slides in from the left edge, cropped ("THS"). No calendar page flying legible. | Put calendar pages centre frame. |
| 27.0-28.0 | camera | medium | New camera at 27.0 (wide, Gary far right, Manager far left, vendors in the middle) holds from 27.0 to 30.0 as requested, but all seven people are 8-19 percent frame height; half the frame is empty floor (bottom 45 percent), the label "YEAR-END BONUS" is cropped at the right edge, "GARY'S" label cropped at 28.3-28.8. | Lower camera / narrower lens, move the table to the lower third, keep people at about 35 percent height, keep labels inside the safe square. |
| 28.3-29.1 | readability / comedy timing | high | The gag (two vendors shoot two, Manager shoots one, Gary shoots himself) is not readable: the 3D "BANG BANG" caption spans the full width, cropped at both edges, over the action; muzzle flashes are 2-3 px; the shot vendors vanish behind the table (no visible fall or hole). Vendor count drops without a visible cause (29.07-29.6). | Medium shot of vendors at the table end (three vendors per shot), flash 3x larger, show falls above the table edge; caption centred small. |
| 29.5-29.9 | human animation | low | Gary falls to the right edge and lies half out of frame. | Shift his mark inward. |
| 20.0-30.0 | lighting | medium | Room is flat blue-grey with blown ceiling lamps (X4); skin and shirt colours of vendors muddy; no key light on characters. | Add a warm key and rim on the table group; reduce ceiling emission. |

## 4. S4 CPO (30.0-40.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 30.0-31.3 | lighting / readability | medium | Board in pale desaturated haze (grey-blue, YAVG high, low contrast); the 16 OEs are tiny white squares flying around the stiffener and are not readable as OEs; the board occupies the top half only, bottom 45 percent empty. Camera is oblique, not "locked top-down". | Larger OEs, camera lower and tighter, darker backdrop to separate the board. |
| 31.4-32.0 | VFX | low | Dolly-zoom power/temperature bars read, but the 3D labels XPU POWER / TEMPERATURE run off the frame edge at 32.0 (cropped). | Safe-square text. |
| 31.8-32.8 | VFX / readability | high | Smoke puffs: large white cotton clouds cover the entire XPU for about 1 s (32.0-32.8), hiding the hero object during the zoom. | Smaller, translucent puffs rising away from the lens; keep the dies visible. |
| 32.7-33.3 | camera | low | Zoom through "DIES + 16 HBM" label (label slides across frame), then dissolves into the PIC with about 2-3 frames of cross-dissolve (fixed per DevLog-004: no black). Good. | none |
| 33.4-34.5 | readability / hardware accuracy | high | PIC: low-contrast blue-grey slab; the 24 rings are tiny blue arcs and the heaters dark squares; three heat pulses read as bright light bands crossing the chip (lighting-like), as already feared in DevLog-004. At phone size the rings cannot be seen. | Fill the frame with 6-8 rings, ring glow colour shift (cyan to red) per pulse, grid darkened; add temperature colour to the substrate rather than a white band. |
| 34.5-35.5 | camera | low | Pull back "HEATERS ON": OE colour change (pink/magenta) is saturated and garish rather than "scorching": hot pink instead of orange/red. | Warm the emission ramp in `s04_cpo.py` OE heater colour (cyan to orange to red). |
| 35.5-38.0 | VFX / readability | medium | Eggs: good physics and fried-egg look; but the gold "BREAKFAST" caption (36.25-38.0) lies across the OE row and hides the first eggs; overcooked look not obvious (large white area remains at 37.5). | Move BREAKFAST to the lower third; stronger brown edge ramp. |
| 38.2 | camera / continuity | medium | Hard cut from the glowing OE macro to a bright-sky human scene (YAVG jump); Gary at the bench in the blue checker world holding no readable plate. | Short whip or glow-through. |
| 38.3-38.9 | human modelling | medium | Gary face is a dark mask in side view (see 10.0); Manager gun is a long thin tube extending beyond his reach. | Same as S2; shorten the gun or add a stock. |
| 38.9-40.0 | VFX / comedy timing | medium | Egg toss: eggs leave Gary at about 39.1, 0.2 s after the shot; brown discs read as flying coins; landing on the Manager (head, shoulder, chest) is not distinguishable; Gary falls off the right edge and is cropped at 39.6-39.9. | More eggs spinning in silhouette, slow the arc 20 percent, hold on Manager head hit 0.3 s. |

## 5. S5 CPO yield (40.0-50.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 40.0-40.5 | continuity / lighting | medium | Cut from the bright blue-sky comedy set to a grey room with the die on a plank: desaturated grey, flat light, die tiny. | Same colour world; light the die with a spot. |
| 40.5-43.0 | camera / readability | medium | Station tour: tools are closer (as requested) and eased (REFLOW about 0.4 s, FAU ATTACH 0.4 s, EIC/PIC 0.4 s, DICING 0.3 s), but the die (hero) sits in the bottom-left corner under the subtitle, large station labels ("REFLOW ON SUBSTRATE") overlap the subtitle, heavy blur between stops. | Place the die at the frame centre-lower third; labels 50 percent smaller, one per station. |
| 43.3-44.7 | VFX / feedback not met | high | Wafer-level tester: the "wafers fly in and out" beat is not visible in the strips (43.5-44.5): the station is nearly static with a red die-map glow on the stage and an orange fireball-like noisy sphere (43.87 and 44.37) over the caption; the station label "WAFERS IN / OUT, STAGE STEPS" is about 2 percent of frame height. | Spawn flat clay wafers sliding in/out of the load drawer with visible motion; replace `sparks_burst` fireball (s05_wafer_test.py lines about 696-698) with small sparks; larger label. |
| 44.2-44.7 | camera | low | Orbit to the monitor is a blurred jump. | fine at final; reduce blur strength. |
| 44.7-47.3 | VFX | low | Monitor: dips point down, window green band, counts 1/9 to 3/30, readable. Good. Left black arch occludes 20 percent of frame. | Offset camera. |
| 47.5-48.4 | readability / human animation | medium | Printout "1/10" in the Manager's hand is hidden by the caption "1 IN 10" which sits on it (47.6-48.4); Gary has his back to camera (no reaction face); both stand stiffly. | Caption above the paper, Gary turned three-quarter with a sweat/sag face. |
| 48.4-49.1 | camera | low | Mid-scene camera change at about 48.7 (pushes onto Gary at close range, Gary waving a hand): good dread beat but the barrel is a pale tube that enters from the frame edge. | none |
| 49.1-50.0 | human animation | medium | The hit: the head snaps but the torso does not react; Gary then lies horizontal in mid-air with nothing under him from 49.5 to 49.9 (no floor in shot, background grey): floats. No hole visible on the torso. | Hold the floor in frame; add a ground contact thud pose; hole decal visible on the chest on the first frame after the bang. |

## 6. S6 Fiber (50.0-60.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 50.0-50.9 | lighting / continuity | high | Hard cut from the flat light grey room to an almost black data hall with a cyan neon tube: no shared exposure; neon looks like a light tube, not a fibre. | Match exposure ramp; add real fibre sheath with a glow core. |
| 50.9-51.5 | hardware accuracy / readability | medium | MPO jam macro: green boxes with stars; the MPO housing is a block with no visible ferrule face / 12-fibre array; stars read as a cartoon crash. | Show the ferrule row at the mating face, key notch, 2 pins (level C per DevLog-004). |
| 51.5-52.2 | readability | medium | About 0.6 s of an almost empty dark wall (tray is black on black; only a few sparkles at 51.6); the tray pulled out is not visible until fibres emerge at 52.2. | Light the tray (rim), start the pull at 51.5, spill at 51.9. |
| 52.2-53.3 | VFX | low | Noodle spill works well and is the best FX moment of the film (waterfall, floor spread). | none |
| 53.3 | continuity | medium | Cut to a new camera; Gary is already standing at the pile. | Show him walking in. |
| 53.5-55.0 | human animation | medium | Gary kneeling with his back to camera for about 1.5 s: nearly static (only slight arm motion); face never seen; the bullet-hole thread (orange fibre) is lost in the noodle pile at 55.0. | Face shown in profile, ties bundles with visible hand action; fibre through the hole in close-up. |
| 55.5-56.7 | camera / comedy timing | medium | Low close-up of Gary's backside (big clay overalls) for 1.2 s while the VO line is "Manager wants it done yesterday"; the VO "Gary's ass gets whipped" is 56.9-58.2. It is funny but early. | Move the close-up to 57.0-57.8 so it lands with VO, or cut to Gary's face. |
| 56.7-58.0 | human animation | medium | Manager enters and hangs the AOC bundle, then swings overhead (57.4-58.0): wind-up readable; the AOC whip physics (chain loop) is much better than v1.0; Manager's pose upright, stiff legs. | Add torso rotation and weight shift. |
| 58.0-58.5 | camera / VFX | medium | CRACK at 58.07 while the whip is still in front of the Manager; contact on Gary is a hard cut to a close-up at 58.27 with a white starburst and a ring; the 3D caption sits across the action. | Keep the wide through contact for 4 frames. |
| 58.6-60.0 | readability / lighting | high | Customers: three figures in black on the dark floor and dark wall; the NVYDIA leather jacket and the OPENAY/ANTHROPY small customer cannot be distinguished; the hug is not visible; "SLAP" caption and flash ring (59.1-59.5) are centred exactly on the slap contact; Manager is hidden behind the customers' backs. | Warmer rim light, lighter jacket shade, customers face the camera three-quarter, move SLAP caption up. |
| 60.0 | continuity | low | Hard cut to S7 black. | fine, intentional. |

## 7. S7 Disclaimer (60.0-62.0)

| t (s) | Category | Sev | What is wrong | Fix |
|---|---|---|---|---|
| 60.0-62.0 | readability / layout | medium | Static text for the full 2.0 s: disclaimer in the top 30 percent, sources at about 1.5 percent frame height (unreadable at phone size), bottom 60 percent empty black; the yellow debug line "[VO ~9 words/s: ...]" is burned in. | Centre vertically, enlarge DISCLAIMER block, drop the debug line; on-card text ("vendor and test condition", "Parody; not affiliated...") differs from the VO ("vendor and mood. Void where copper is cheaper."): align. |

## 8. Overall look versus the reference

Reference observations (50.6 s, 8 detected cuts): one saturated blue-sky gradient and green/earth ground in nearly every frame; one hero object occupying 40-70 percent of the frame; wide/fisheye lens and low angles; warm glow and particle bursts on key events with dark vignetting; bold white caption with dark shadow in the lower third, 1-3 words per second.

1. Lighting: pick one key/fill/rim recipe and one sky for all scenes (X6). Today S1/S3 lamps and S2 bloom are blown out (X4), S6 is under-lit, characters have no rim. Cap emission on lamps and glow (`p_glow` above 0.55), add a warm key and cool rim on every human shot, keep character luma at least 35 percent of frame luma.
2. Colour grading: global LUT with one palette (saturated cyan sky, warm clay, orange glow accents); remove the pale haze of S4/S5, the flat grey of S5, and the near-black S6; keep contrast high on hero objects.
3. Camera language: fewer cuts, more continuous moves. Reference has about 1 cut per 6 s; v1_1 has about 25 intra-scene cuts (list below). Use 2-3 camera moves per scene with eased paths and matched-direction whips. Wide lens, low angle, humans filling 35-50 percent of frame height.
4. Pacing: hit cadence is good in S1 (0-3 s) and S4 egg toss, but locked or dead shots at 20.0-21.6 (S3), 15.0-15.4 and 12.4-14.7 (S2), 51.5-52.2 (S6), 53.5-55.0 (S6) break the "one event per 0.5-1.0 s" rule.
5. Background simplification: remove floating labels, FX/TC/CARD HUD (X2), cropped 3D text, the infinite checker floor in S2/S4; unify the set (lab floor and wall tint).
6. Captions: bold face with a hard dark shadow, one line, 1-3 words, kept off the action; sync to the new VO (X1).

## 9. Hard cuts and boundary frame pairs (frame before / frame after)

Detected by scene-change threshold 0.18 plus visual check. Scene boundaries are B*, in-scene cuts C*.

| ID | Film t (s) | Frame before | Frame after | Comment |
|---|---|---|---|---|
| C1 | 0.77-0.87 | tunnel smear | data hall fly-along blur | zoom-out, blurred, ok |
| C2 | 3.0 | 5 m wide, Gary far right | 2 m wide, closer | hard cut, 1 blurred frame |
| C3 | 3.6 | 2 m, ghost frame | 1 m | ghost double exposure |
| C4 | 6.0 | Gary stretch, blur | scope close-up | blur cut |
| C5 | 6.77 | scope close-up (blur) | Manager profile at bench | blur cut |
| C6 | 8.5 | Manager medium close (gun arm) | wide, Manager at frame edge | cut |
| C7 | 9.2 | blur Manager | wide BANG with Gary at left edge | on SFX |
| B1 | 10.0 | blurred zoom into scope (S1 9.97) | Gary profile, blue sky world | hard cut, location change |
| C8 | 10.8 | Gary scene | rack interior with NARROWCOM sign | cut |
| C9 | 12.07, 12.47, 12.87, 13.27, 13.7, 14.2, 14.3, 14.7, 14.8 | tray n | tray n+1 (camera hop) | 9 flagged, all hops with 1-frame flash |
| C10 | 15.4 | glare rack top | bright exterior crane | hard cut, YAVG 141 to 195 |
| C11 | 16.4 | exterior crane, rack and bench | wide, Manager at whiteboard | cut |
| C12 | 18.6 | Manager beside board with gun | Manager closer, alone | cut, line crossing |
| C13 | 19.2 | Manager steam, aiming | BANG wide, Gary fall | on SFX |
| B2 | 20.0 | Gary on the floor, star | conference room static wide | hard cut |
| C14 | 21.8 | locked wide, Gary | table-level fight view | whip pan |
| C15 | 23.0 | fight view | packages on table | cut |
| C16 | 24.2 | packages | DIE SIZE fight view | cut |
| C17 | 25.27 | normal room | sky cycles (roof off) | strobe starts |
| C18 | 27.0 | day/night view | new wide with Manager far left | cut |
| B3 | 30.0 | Manager standing, Gary on floor in conference room | pale board with OEs | hard cut, colour jump |
| C19 | 31.4 | board | dolly-zoom bars | cut |
| C20 | 33.3 | package edge | PIC (cross-dissolve) | intended |
| C21 | 35.4 | board pull-back | egg/OE macro | cut |
| C22 | 36.87 | OE column | other OE column | cut |
| C23 | 38.2 | OE macro with eggs | Manager aiming, Gary at bench | hard cut, scale jump |
| B4 | 40.0 | Manager with eggs around his feet | grey room with the die | hard cut |
| C24 | 48.7 | wide Manager/Gary/machine | close on Gary | push in |
| B5 | 50.0 | Gary floating, grey | dark data hall, cyan fibre | hard cut, luminance jump |
| C25 | 50.9 | cyan fibre wide | MPO macro | cut |
| C26 | 51.5 | MPO macro | wide dark wall | cut |
| C27 | 53.2 | fibre waterfall | Gary standing at the pile | cut |
| C28 | 55.6 | Gary kneeling, wide | close on his back | cut |
| C29 | 56.7 | Gary close-up | Manager, wide | cut |
| C30 | 57.4 | Manager lone | wide with pile, Manager and customer | cut |
| C31 | 58.27 | CRACK wide | close on Gary (hit) | cut |
| C32 | 58.6 | Gary close | wide with customers | cut |
| B6 | 60.0 | customers wide, slap | S7 black card | intentional |

## 10. VO and SFX sync (cue vs visual by more than 3 frames)

Cue source: `scripts/audio/sfx_cues.json` and DevLog-004 section 6; VO windows `make_vo.py` SEGS and the measured VO gaps (8.99-9.30, 16.55-16.90, 28.43-29.10, 39.92-40.20, 48.68-49.20). Checked at frame level: shotgun 9.2 (BANG and cut at 9.2: ok), 19.2 (ok), 28.3 (BANG BANG caption appears 28.3: ok), 28.95 (flash at 29.0-29.07: ok), 38.9 (caption 38.9, flash 38.93-38.97: ok), 49.1 (caption 49.1, flash 49.13-49.17: ok), whip crack 58.067 (CRACK caption at 58.067: ok), slap 59.147 (flash 59.13-59.17: ok).

| Film t (s) | Cue | Visual | Offset | Note |
|---|---|---|---|---|
| 4.95, 5.35, 5.65, 5.95 | strand_pop (4 cues) | no strand or bundle visible in the STRETCH shot (4.4-6.0) | no matching visual | cue placed in window, not frame-checked (DevLog-004) |
| 7.55 | steam_hiss | first steam puff at about 7.7-7.77 | about 5-6 frames early | S1 |
| 28.5 | second shotgun (BANG_N) | no visible flash; first small flash at about 28.6 | about 3 frames (not resolvable at this size) | S3 |
| 25.4 | timelapse / balloon_inflate | sky change visible from 25.27 (frame 25.13 still room) | about 4 frames late | S3 |
| 29.24, 29.44, 29.65 | clay_thud (body falls) | vendors disappear behind the table; Gary reaches the floor at about 29.5-29.8 | cues estimated, no visible fall for vendors | S3 |
| 39.62-39.90 | 8 floor egg_splat | floor eggs only partly visible at the frame edge | not verifiable | S4 |
| 9.3-10.0 | VO "Poor Gary." | picture is a zoom blur onto the scope from 9.45; Gary not in shot | narration vs picture | S1 |
| 10.2-14.3 | VO "Gary just wants to fix the signal. So he puts a re-timer on every connector." | burned subtitle "Signal fading? Gary puts a chip on every connector" | text and timing differ | X1 |
| 20.2-28.6 | VO S3 vs subtitles "Gary moves the optics closer to the logic" etc. | | text differs | X1 |
| 50.2-53.3 | VO "Gary is finally pulling the fibers together. One thousand per tray." | no Gary on screen until 53.3 (neon fibre, MPO jam, empty wall, spill) | 3 s without Gary | S6 |
| 53.3-57.6 | VO "Manager wants it done yesterday, but the right fiber length arrives next Tuesday." | subtitle "Thousands of them" until 55.2, "Manager wants it done yesterday" 55.4-57.6 | subtitle lags VO by about 2 s | X1 |
| 55.5-56.7 | low close-up on Gary's back | VO is on "...arrives next Tuesday"; the whipped-ass line starts 56.9 | joke 1.5 s early | S6 |

Visual luma flicker not tied to a cue: 1-frame flashes at each S2 camera hop (list in C9) and three 0.27 s night phases in S3 (25.53-26.83); no matching SFX other than the S3 timelapse.

## 11. Top 10 highest-impact improvements

1. Replace burned subtitles (X1, X3) with text and timing from `make_vo.py` SEGS, bold face, hard shadow, off the action; gate off FX/TC/CARD HUD for finals (X2).
2. Make humans 35-50 percent of frame height in every wide (X5): S3 27-30, S2 15-19, S4 38-40, S6 58-60; this alone rescues the shooting gags.
3. Rebuild S3 28-30 as two readable medium shots (vendors at the table end; Manager and Gary), with larger flashes, visible falls above the table edge, a small centred BANG.
4. Fix lighting and exposure: cap lamp emission and `p_glow` peak, remove lamp speckle (X4), add key and rim on characters, lift S6 and S5, one palette across scenes (X6).
5. S1 stretch gag (4.4-6.0): hands on the rack, taut bundle visible, strain pose, caption off the head.
6. S2 rack ride (12.4-15.4): continuous rise instead of 0.4 s hops, de-bloomed chips that read as chips, and a real exit from the rack at 15.4 (no hard cut into the exterior).
7. S5 wafer tester (43.3-44.7): visible wafers sliding in and out, replace the orange fireball with sparks; keep the die and printout unobscured.
8. Death and hole readability: hit-stop pose, visible hole on the first frame after each bang, a ground contact for Gary's fall (S5 49.5-49.9 floats; S1 9.2 not visible).
9. S3 25.3-27.0 sky strobe: lower day/night contrast, 2 cycles or longer phases, calendar pages in centre frame; and S4 31.8-32.8 smoke clouds that hide the XPU.
10. S6 dressing and staging: lit tray at 51.5, customers lighter and facing camera, SLAP caption moved off the contact, Gary shown at 50.2-53.3 to match the VO; plus Gary's profile face (black mask at 10.0 and 38.3-38.9).
