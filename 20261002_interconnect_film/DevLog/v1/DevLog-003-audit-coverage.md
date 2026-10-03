# DevLog-003-audit-coverage: storyboard coverage audit of the six v1 scene scripts (S1-S6)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Auditor | fresh-context subagent (read-only; no Blender run, no edits to existing files) |
| Inputs | DevLog-001 Sections 1, 3, 4 (rev 5), 5; DevLog-003 master plan; scene devlogs s01-s06; `scripts/film_v1/asm.py`, `s01..s06*.py`, `SCENE_BRIEF.md`, `render_presets.py`; the six contact sheets |
| Not verified | Blend internals (scene settings were read from the scripts, not from the .blend files); full-resolution frames (only 540x675 contact-sheet stills were viewed, 8-10 per scene) |

Legend: PRESENT / PARTIAL / MISSING / DEVIATES. "Sheet" = contact sheet evidence. Line numbers refer to the scene script named in the section.

## 0. Cross-scene findings (summary)

| # | Finding | Evidence |
|---|---|---|
| X1 | Settings consistent: every scene builds through `asm.new_scene(preset="standard")` = 1080x1350, 30 fps, `frames=300` (S7 `frames=60`); fps and resolution come from `render_presets.apply_render_preset` defaults. Not read from the .blend files. | `asm.py` L36-41; `render_presets.py` L29-35; s01-s06 call `new_scene` without overriding w/h/frames |
| X2 | Narration word counts match the storyboard in all six scenes (22 / 21 / 23 / 21 / 22 / 20 = 129). Chunk windows are inside the storyboard windows. | counted from `asm.narr` calls in each script |
| X3 | Timecode overlay: `asm.timecode(n)` is called in every scene with n = scene number, text "Sn 0:ss", scene-local seconds (S1..S6 use 10 entries; S7 2). Consistent. Question for Wentao: the in-frame "[FX: ...]" production notes (yellow, top-left) and the timecode are leftover crude annotations; they are rendered into every v1 frame. Decide whether they stay in v1 renders. | all scripts; visible in all sheets |
| X4 | Bullet-hole schedule: all scenes follow r = max(0.3, 0.9^n) with T_shot 9.2 / 19.2 / 28.95 / 38.9 / 49.1. Method differs: S2, S4, S5, S6 use floor-stepped CONSTANT keys; S3 uses continuous values with LINEAR keys every 0.5 s (`key_heal`, s03 L81-86); S4 keys the first value continuous (0.9^7.2 = 0.468 for hole 2) then jumps to the stepped 0.43 at 1.2 s. Handoffs are numerically consistent within about 0.02 (S2 end 0.478 vs S3 start 0.47; S4 end hole 4 = 1.0 vs S5 start 1.0; S5 end hole 4 0.478 vs S6 start 0.478; hole 5 1.0 at S5 end vs S6 start 1.0). The S6 devlog text ("hole 4 0.46", "hole 5 0.94") does not match the S6 code (0.478, 1.0). SCENE_BRIEF's "0.5 at S2 start" is inconsistent with its own formula (S2 agent noted this). Cosmetic. | s02 L345-358; s03 L81-86, 351-352; s04 L397-412; s05 L95-110; s06 L227-239 |
| X5 | Gary and Manager are the same library assets in all scenes, so appearance is consistent. S5 deliberately mirrors the staging (Manager screen-left, Gary yaw -45 deg); S3 has a one-off self-shot IK on `ik_hand_R`. | scripts |
| X6 | Shotgun mounting convention differs between scenes: S1 solves the orientation from the two hand hooks; S2 and S4 use `rot=(0,0,pi)` with no roll; S3 and S5 use the matrix gun-Y -> hook -Y, gun-Z -> hook +X, gun-X -> hook +Z. S2/S4 vs S3/S5 differ by a 90 degree roll about the barrel axis (barrel-pair orientation, trigger side). Not visible as a defect at contact-sheet scale, but not one handling rule. | s01 L383-400; s02 L34, 374; s03 L204; s04 L420; s05 L526-528 |
| X7 | Caption style (CAP chunks lower third, LAB cyan top-left, BIG yellow centre, CARD bottom, FX notes yellow top, wl world labels) is consistent across sheets. S3 adds `BIGHI` (BANG in upper frame); S4 and S5 re-parent the caption holder for macro shots. | all sheets |
| X8 | Latent rebuild hazard in S4: `_finalize_collections_objectwise` (s04 L46-63) calls `L._VIS.clear()` before keying collection windows. All captions, world labels, floors, egg windows, power bars, flying eggs and the fade quad are registered in `L._VIS` via `L.V/VA` (`ov`, `wl`, `fade_quad` in blender_lib). With the current `asm.finalize` (which calls `_finalize_collections` first, then `L.finalize_visibility`), re-running s04 would drop all of those windows (everything always visible). The shipped `s04_cpo.blend` (19:15) predates the asm.py edit (19:27) and its sheet shows correct windowing, so the existing blend is fine; the script is not reproducible as is. Fix: delete the monkeypatch and the `L._VIS.clear()`; asm now keys object-level visibility itself. | s04 L46-63; blender_lib `ov`, `wl`, `fade_quad`; asm.py L307-312, L334-337 |
| X9 | S1, S2, S4 blends were built before the `asm.F` round-half-up and `asm.show` edits (blend mtimes 19:15-19:16 vs asm.py 19:27). Keys that fall on x.5 frames can land one frame earlier than the scripts imply; no collisions found in S1/S2/S4 timing lists, so low risk. Rebuild S1/S2 for consistency only if S4/S1 are rebuilt anyway. | file mtimes |
| X10 | Motion blur is off in the `standard` preset (render_presets L24). Every storyboard "motion-blur whip / shuffle" (S5 0.3-3.3 s, S4 whips) is approximated by emissive streak boxes only; compositor bloom/vignette/CA/grain is applied by `render_scene.py` (cartoon). Camera shake on S4 dolly-zoom and S1 "handheld" is not implemented anywhere (no shake keys in any script). | render_presets; s04 devlog "[ ] camera shake" |
| X11 | The bullet hole is the running gag but it is not legible in most "after hit" frames at contact-sheet scale: S1 final wide (hole edge-on), S2 BANG still, S4 final, S5 final (devlog admits "small tan patch"). Only S3 head close-up (silver disc visible through hole) and S6 sheet show clear holes. | sheets S1 #8, S2 #8, S4 #8, S5 #8 |
| X12 | Camera FOV note for future camera edits: for 1080x1350 Blender AUTO sensor fit applies the 36 mm sensor to the larger (vertical) dimension, so horizontal FOV is about 54 deg at 28 mm, narrower than assumed. S1 6.8-8.5 s framing suffers from this (see S1). | s01 L428; sheet S1 #6 |

## 1. S1 Copper (film 0-10 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-2.2 fly-along NVL72 backplane copper cartridge wall; Pulse sprints; snap-stop onto Gary; "COPPER"; card SemiAnalysis | PARTIAL | `asm.shot` L422, `big` L434, card L440, Pulse hook run L299-302. Sheet #1: wall reads as a dark panel with copper conduit lines; cartridge ports/handles are not legible; Pulse is a tiny yellow ball. |
| 2.2-3.0 racks 5 m, bundle of 14 cables, Pulse decays to about 0.2; "5 m"; 802.3bj card | PARTIAL | p_gap keys L35, 187; Pulse scale exp(-0.32 L) = 0.20 L320; card L441. Sheet #2: 14 cables converge to one point at each port and read as one thin sagging wire (devlog problem 3). Rev 4 request "at least a dozen cables" is met in geometry, not on screen. |
| 3.0-3.6 racks 2 m, decay 0.5, "2 m", 802.3ck | PRESENT | L35, L304, L442; sheet #3 |
| 3.6-4.2 racks 1 m, decay 0.7, fisheye push-in, "1 m", 802.3dj | PARTIAL | content present; camera is lens 20 mm normal, not fisheye (L425) |
| 4.2-6.0 Gary pulls rack one more meter, bundle goes taut, handheld close, "STRETCH" | PARTIAL | continuous pull L189-200, pull_cable L249, big L438, sag to 0 L233. Sheet #4: Gary pulling; bundle not visible in that framing. No handheld shake. |
| 6.0-6.8 bench scope close-up: ONE eye, shakes, nearly closed, BER climbs; lab "EYE NEARLY CLOSED BER RISING" | PRESENT | `T.eye_sequence` with noise/jitter ramp L344-351 (single trace set, no overlay); lab L445; sheet #5 shows one eye and "BER 1.2E-05" on a bench scope. |
| 6.8-8.5 wide frame of bench, scope and Manager; Manager walks in, stares, face red, brows, steam | PARTIAL | Manager walk L368, p_anger/p_flush L378-381, ear_steam L403-410. Framing L428 targets x = 9.9 with horizontal FOV about 54 deg: the scope (x = 8.0, 1.9 m left of target) sits at the left edge. Sheet #6: Manager in profile, scope not clearly in frame. Rev 4 request "framing scope and Manager together" not clearly met. |
| 8.5-9.2 Manager turns, raises shotgun, push-in on barrel | PRESENT | aim_gun L373, shot L429; sheet #7 |
| 9.2-10.0 BANG, smoke ring, first hole, topple | PARTIAL | muzzle_flash x2 + smoke_ring L411-413, hole 1 0->1 at 9.2 L268-269, topple_back L251, "BANG" L439. Sheet #8: Gary standing, arms out; hole not visible. Gary walks 1.6 m away after the pull (not in storyboard). |
| Narration 4 chunks 0.2-7.5 | PRESENT | L432-433 (22 words) |
| Cards, labels, FX notes, world labels, timecode | PRESENT | L439-455, `timecode(1)` |
| One set of eye traces per scope | PRESENT | see 6.0-6.8 |
| Scope on lab bench | PRESENT | `lab_bench` + `bench_oscilloscope` L334-339 |

Verdict S1: needs fixes (P2 items below), no blocker.

## 2. S2 Retimers (film 10-20 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-0.8 flash cut: bench scope closing eye + BER, Gary watches; lab "EYE CLOSING" | PRESENT | `make_eye_sequence` L85-96, lab L520, shot L509; sheet #1. Scope small in frame; BER text not legible at this size. |
| 0.8-1.6 camera inside rack at first tray, row of 4 connectors, first retimer row slides in + fades in + glow flash, NARROWCOM sign on backplane | PARTIAL | chips: 4 per row, slide 0.5 s, fade 0.45 s, p_glow flash L401-435; sign L441-479; lab L521. Sheet #2: sign and 4 connector footprints visible, no chips yet at ~1.2 s; chips are 60 mm (x4) and read tiny. No front frame visible (met). |
| 1.6-5.4 camera rises through 8 trays; chips slide in tray by tray; 1 row low, 2 middle, 3rd row near top | PARTIAL | row counts 1/2/3 by tray L404; `t_row` = arrival - 1.2 s (+0.45 per row) L406. Because rows appear about 1.2 s (about 1.6 m of rise) before the camera reaches the tray, the slide/fade is mostly off-screen; the camera sees finished glowing chips. Devlog confirms one tray visible at a time, pan fronts sweep as white bands, and a dark thick block dominates the foreground (sheet #3, #4). Sheet #4 shows one row of 4 glowing chips; the 3rd row is not visible in any still. Rev 5 request "retimers appear slower and more dramatically, 3rd row when camera nears the top tray" is not demonstrably met. |
| 5.4-6.4 exit rack top, crane down into room; "OUT OF THE RACK" | PRESENT | shot L512, lab L523; sheet #5 |
| 6.4-8.6 Gary stares at recovering eye (BER falls); Manager at big whiteboard, two curves, no units, labels ENERGY / BIT and LATENCY ride the tips | PRESENT | eye ramp L90, graph sequence L103-110, tip-riding `wl` keys L534-557, card L525; sheet #6 (labels very small: size 0.12 on a 2.9 m board) |
| 8.6-9.2 Manager turns, raises shotgun, push-in | PRESENT | aim_gun L362, shot L514; sheet #7 |
| 9.2-10.0 BANG, hole 2, topple | PARTIAL | flash+ring L378-381, hole 2 L357-358, topple_back speed 2.4 L330; "BANG" L531. Sheet #8: Gary on floor, hole not visible; muzzle flash not visible. |
| Scope on proper lab bench | PRESENT | L302-307 |
| NARROWCOM parody logo (valley mark, wave line) | PRESENT | sign sheet #2 shows valley mark over wave over "NARROWCOM" |
| Narration | PRESENT | L518-519 (21 words) |
| Hole 1 recovery | PRESENT | stepped schedule L346-356 |

Verdict S2: needs fixes (rack-rise legibility is the main storyboard beat and the weakest).

## 3. S3 NPO (film 20-30 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-1.8 conference table, module slides beside ASIC, four vendors in parody shirts lean in; "NPO"; Cheng 2025 | PARTIAL | NPO slide L224-226, vendors L186-190, lab L519, card L525. Sheet #1: only 2-3 vendors visible, small; ASIC/module small on a 5.6 m table. |
| 1.8-3.0 laser fight, in-package vs external; vendors shout and shove | PRESENT | els + copper block L228-239, shout_loop/shove L412-415; sheet #2 |
| 3.0-4.2 real BGA, LGA, fine-pitch BGA standing, undersides to camera; shouting continues | PRESENT | L242-260; sheet #3 (clear) |
| 4.2-5.2 mismatched dies; vendors keep fighting | PRESENT | L262-299; sheet #4 |
| 5.2-7.0 dies side by side then wobbling tower; calendar "+3 MONTHS"; price tag inflates; brawl to punches | PRESENT | tower L293-299, calendar L302-309, balloon L312-318, punch_loop L417; sheet #5 (fight is far and small; calendar back pages blank, label carries the gag) |
| 7.0-8.3 Gary hands vendor the YEAR-END BONUS envelope; Manager walks in and watches | PRESENT | hand_over_envelope L335, envelope clones L321-331, Manager walk L489; label L524 |
| 8.3-8.9 MOLEXX and NUBISS shoot TERAHOP and AYARR; Gary pulls his own gun | PRESENT | L426-440, 8.3 and 8.5 s (NUBISS moved 0.1 s later); gary gun visible from 8.3 L385; sheet #6, #7 |
| 8.9-10.0 Manager shoots NUBISS, Gary shoots himself in the head, MOLEXX only one standing | PRESENT | L454-475, head hole L353-354, IK L387-395; head close-up sheet #9 shows hole with barrel visible as silver disc; sheet #10 wide aftermath. |
| Vendors shout / shove / beat each other | PRESENT | see rows above |
| Parody shirt logos | PARTIAL | raised-text wordmarks only (no mark), illegible at the distances used (sheet #2: tiny "MOLEXX" on chest). Meets Rev 5 "text wordmarks" literally. |
| Bullet holes on vendors cut no wordmark letter | PRESENT | hole moved 0.14 m down L477-485 |
| Narration | PRESENT | L513-514 (23 words) |
| Final BANG BANG / BANG captions | DEVIATES (minor) | moved to upper frame (BIGHI L515-518); NUBISS fires at 8.5 instead of 8.4 |

Verdict S3: ready, with minor notes (readability of the fight at 1.8-7.0 s; finale in 0.2-0.4 s cuts).

## 4. S4 CPO (film 30-40 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-1.4 top-down DGX/HGX-style board, Rubin-Ultra-style package (gold lid, 4 dies, 8+8 HBM), 8+8 OEs slide onto left/right edges | PRESENT | hgx S4 variant L134-141, 16 OEs L201-244; sheet #1 (board with regulators/MLCC, lid frame, 4 dies, OEs on edges). OEs scaled 0.5 to fit hook footprint (asset mismatch). |
| 1.4-2.6 dolly-zoom, power/temperature bars lurch, smoke puffs | PARTIAL | bars L257-271, smoke L274-276, shot L476; no camera shake (storyboard note "shake" not implemented); sheet #2: smoke puffs large and hide the dies |
| 2.6-3.45 continuous zoom onto one OE (no cut) | PRESENT | shot L479-480 lens 55->85 from the same pose as the previous shot; sheet #3 |
| 3.1-3.8 quick fade to black and back in on the PIC | PRESENT | fade quad L292 |
| 3.5-4.6 PIC close-up, 24 rings on 28 slabs, heat wave left to right across rings and substrate, label | PRESENT | `p_wave_pos` -0.3 -> 1.3 L285-287, labels L288-289; sheet #4 shows ring colors and slabs in a left-to-right gradient |
| 4.3-4.9 fade out and back on the board | PRESENT | L292 |
| 4.6-5.4 pull back, every OE changes color cyan/orange/red and glows (staggered) | PARTIAL | p_heat keys L238-243 (0.03 s stagger); sheet #5: OEs read as faint pink/cyan dots on a 1.56 m board, weak. |
| 5.4-8.2 one egg per OE (16), drop, glassy -> opaque, edges brown, bubbles, yolk, steam; camera along left column then right | PARTIAL | 16 eggs with NG_egg_fry L316-359; sheet #6 shows left column eggs frying. Timing: egg n lands at 5.4 + 0.06 n, so right-column eggs (n = 8..15) land 5.88-6.3 s and finish cooking about 7.5 s, while the camera reaches the right column at 6.8 s (shot L486). The right-column drop and early fry happen off-camera; only the second half of the cook is seen. |
| 8.2-8.9 human scale: Gary holds plate, Manager aims | PRESENT | L372-392, 7.0 hold_plate; sheet #7 (very plain room, small figures) |
| 8.9-10.0 BANG, Gary throws eggs up as he is hit, eggs fly out of frame and drop on the Manager's head, hole 4 | PRESENT | throw_up L386, 3 flying eggs x3.5 L451-471, tilt-up shots L490-491, hole 4 L413-414; sheet #8 shows egg on Manager's head. Only 3 eggs (plate eggs), not 16 (storyboard says "the eggs"). |
| 16 OEs (8+8), one egg per OE | PRESENT | |
| OE color/glow heating | PRESENT but weak | see 4.6-5.4 |
| Narration | PRESENT | L494-496 (21 words) |
| Captions/labels | PRESENT | L497-509 |

Verdict S4: needs fixes (X8 rebuild hazard; right-column egg timing; OE heat legibility; no shake).

## 5. S5 CPO yield (film 40-50 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-0.3 die on the engine, locked | PRESENT | shot L565; sheet #1 (die small and flat) |
| 0.3-3.3 die yanked back through reflow, FAU attach, EIC/PIC bonding, dicing + tape (merged); motion-blur streaks; "BACK THROUGH THE LINE" | PARTIAL | five stations, labels L585-588, shots L566-570, streaks L204-215, tape diced -> whole at 2.55 s L276-281. Motion blur is not enabled (X10); linear camera moves, no speed ramps (devlog). Sheet #2, #3: world labels cropped by the frame edge ("TRATE" = "SUBSTRATE", "EIC / PIC BONDI"). |
| 3.3-4.8 CM300-style station: fixed probe head, wafers fly in/out one after another, stage steps under probe, die map lights up, 3 wafers in 1.5 s | PRESENT | `build_cm` L291-399: 3 wafers, period 0.5 s, drawer shuttle, 8 hops x 1 frame, 67/668 green per wafer; sheet #4 shows station from above with wafer and die map. Deviations: wafer lifted 75 mm, probe card 0.15 scale, hops limited to +-100/120 mm. Only one sheet frame in this window; flight and stage motion not verifiable from stills. |
| 4.8-7.0 screen with Lorentzian dips, new traces appear (newest white), narrow green spec window, running count, 1 in 10 | PRESENT | `spectrum_sequence` L408-432, label L591-592; sheet #5 shows dips, white newest trace, green window, "2/17" counter |
| 7.0-8.4 human scale: Manager holds "1 / 10" printout next to Gary; "1 IN 10" | PRESENT | L515-523, L599; sheet #6 |
| 8.4-10.0 Manager raises shotgun, BANG at 9.1, hole 5 | PARTIAL | aim_gun L472, flash/ring L551-553, hole 5 L535-536; sheet #7 shows flash on Gary. Sheet #8: hole 5 on the lying body renders as a small tan patch, not a see-through hole (devlog admits). |
| Parody / wafer-level test done with CM300-style station | PRESENT | |
| Scope on lab bench | PRESENT | `lab_bench` + `sampling_scope_dca` L436-444 |
| Narration | PRESENT | L594-595 (22 words) |

Verdict S5: needs fixes (hole 5 legibility; world-label cropping in the line shots; optionally motion blur).

## 6. S6 Fiber (film 50-60 s)

| Storyboard beat | Status | Evidence |
|---|---|---|
| 0.0-1.5 single glowing fiber flows; end-face macro; connector jams; "FIBER"; SMF-28 card | PARTIAL | glowing cord L333-340, jam keys L347-357, impact stars L467, label/card L461-465; sheet #1, #2. No x40 end-face macro (devlog deviation). |
| 1.5-3.2 tray pulled out, fibers spill like noodles | PRESENT | p_slide L93-95, pile growth L188-190; sheet #3 (waterfall of strands; tray body not readable) |
| 3.2-7.2 pile grows, Gary kneels and ties bundles, fibers squirm, a fiber threads through a shrunken hole; footnote 9,072 | PRESENT | kneel/tie L223-224, bundles L434-449, thread L359-418, card L464; sheet #4, #5 (orange thread visible). Pile is 600 strands, not thousands (documented). |
| 7.2-8.0 Manager walks in with AOC bundle as whip and winds up | PARTIAL | Manager is walking in from 3.2 s and is visible in the 6.0-7.4 background (show window L253 is 6.0, devlog says before 6.4); arm_whip L260; whip L265-276. Sheet #6: whip rises out of frame. |
| 8.0-8.7 CRACK, AOC bundle whips Gary | PARTIAL | crack FX L452-453, jolt_hit L225, big L472. Sheet #7: ring/stars and caption cover the impact; the whip is a thin line (3 mm x2.5), not a visible bundle; whip path is the generic demo wave (passes at belt height, sags to the floor afterwards). |
| 8.7-10.0 NVYDIA customer slaps Manager; OPENAY/ANTHROPY hugs NVYDIA's leg; "SLAP", "THE CUSTOMER" | PRESENT | L287-316, labels L462, L466, L473; sheet #8: green-shirt and dark-shirt customers beside Manager. Leg hug legibility at this scale unclear. |
| Narration | PRESENT | L459-460 (20 words) |
| Holes recover, never heal | PRESENT | L227-239 (holes 1, 2, head at 0.3 floor; hole 4 0.478 -> floor; hole 5 1.0 -> 0.9...) |

Verdict S6: ready with minor notes (whip legibility is the main one).

## 7. Prioritized fix list (owning scene script)

P1 (storyboard beat not delivered, or reproducibility)
1. S4 `s04_cpo.py` L46-63: remove the `_finalize_collections_objectwise` monkeypatch and `L._VIS.clear()` (asm now keys object windows); then rebuild and re-check captions/eggs/floors/fade windows on 6 stills. Without this the script cannot be re-run safely (X8).
2. S2 `s02_retimers.py` L401-435 and shots L510-511: make the retimer slide-in and fade visible to the rising camera. Options: delay `t_row` so each tray's rows play when the camera is at that tray (use `t_arrive - 0.2` instead of `- 1.2`), enlarge chips (S_DETAIL x4 -> x6-8) or lower the camera tilt so the connector rows fill the frame, and show the 3rd row in at least one still near 4.8-5.2 s. Remove or mask the dark foreground block and white pan fronts (hide `trayN_components` front block or move camera y).
3. S1 `s01_copper.py` L429 and the bundle: make 14 cables read at 2.2-4.2 s. Scene-level: add 14 fanned copies of the bundle cable (or scale the bundle cable thickness x4-5 via its geometry-nodes od input if exposed) with fan-out at both ports. Also S1 L428: re-aim so scope and Manager are both inside the frame at 6.8-8.5 s (target x about 8.9, lens 22-24, or camera further back).

P2 (beat present but weak or partly off-camera)
4. S4 L348 (`t0 = 5.4 + 0.06 n`): stagger right-column eggs to land at about 6.8 s + 0.06 (n - 8) so the camera sees them drop and fry, or reverse camera order per the lengths; confirm 16 eggs finish by 8.2 s.
5. S4 heating 4.6-5.4 s: raise visibility of OE color (larger Strength Max L226, or add the glow plate scale) and consider pulling the camera closer; add camera shake keys on 1.4-2.6 s spikes (storyboard FX note).
6. S5 hole 5 at 9.1-10 s (L535-536, L470 topple): choose Gary's yaw/fall so the hole axis faces the camera at the end, or raise the hole rim/cutout radius for this scene, so a see-through hole is visible on the lying body. Apply the same check to S1 (9.2-10 s), S2 (9.2-10 s), S4 (9.0-10 s): add a 0.2-0.3 s insert or yaw the victim toward the camera.
7. S5 L585-588: world labels at 0.45 m are cropped by the tracking shots; reduce size or place them clear of the frame edges (labels must stay readable for 0.5 s per station).
8. S6 whip (L265-276, L269 scale): make the AOC bundle read (cable thickness x4-6, or add 2-3 duplicate strands), and tune the tip path so the crack at 8.0 s lands on Gary visibly rather than at belt height; cut the CRACK ring/stars size so the whip impact stays visible. Show the Manager only from about 7.0 s (move `asm.show(mgr, 6.0, 10.0)` and the walk-in so he is not in the far background at 3.2-6.4 s, or accept as foreshadowing).
9. S2 L542-543: increase label size (ENERGY / BIT and LATENCY) to about 0.2 so they read at 6.4-8.6 s; whiteboard framing already adequate.

P3 (consistency and polish)
10. Unify shotgun mount orientation across S1-S5 (use the S3/S5 matrix `Matrix(((0,0,1),(0,-1,0),(1,0,0)))` in S2 and S4, or confirm the roll difference is invisible).
11. Unify hole-recovery method (stepped CONSTANT keys everywhere; S3 `key_heal` -> stepped; S4 first value floor-stepped).
12. S3 1.8-7.0 s: vendor fight is small and far; add one tighter cut or move the camera closer so the shove/punch reads; consider larger wordmarks on shirts (Rev 5 asked for logos that read).
13. S1 3.6-4.2 s: lens 20 mm is not a fisheye (storyboard); handheld shake absent at 4.2-6.0 s.
14. S6 0-1.5 s: add the end-face macro (fiber_connectors x40) as the "jam" close-up if time permits.
15. Decide whether "[FX: ...]" yellow notes and the "S? 0:ss" timecode remain in v1 renders (X3). Update the S6 devlog hole values (0.478, 1.0).

## 8. Overall verdict per scene

| Scene | Verdict | Blocking items |
|---|---|---|
| S1 Copper | needs fixes | 14-cable bundle reads as one wire; scope not in the Manager-stare frame |
| S2 Retimers | needs fixes | rack-rise chip appearance largely off-screen / tiny, no visible 3rd row, foreground block |
| S3 NPO | ready | minor readability notes only |
| S4 CPO | needs fixes | rebuild hazard (X8), right-column egg timing, weak OE heat color, no shake |
| S5 Wafer test | needs fixes | hole 5 not legible, cropped world labels; wafer-in/out motion unverified from stills |
| S6 Fiber | ready | whip legibility is the main minor item |

All six scenes cover the storyboard beat list structurally (every action, caption, card, narration chunk present in the script); the defects are legibility, timing relative to the camera, and one reproducibility hazard rather than missing scenes.
