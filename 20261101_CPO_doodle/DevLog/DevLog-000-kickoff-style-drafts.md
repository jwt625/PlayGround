# DevLog-000: CPO assembly line as a Japanese doodle animation - kickoff and style drafts

| Field | Value |
|---|---|
| Date | 2026-11-01 |
| Status | Style-draft gate. No animation is built until style and open questions are answered |
| Scope | Compact isometric manufacturing-line animation of the CPO assembly flow and its test points, in a Japanese comic doodle style |

## 1. Request (owner, verbatim, 2026-11-01)

```text
read the video skill, and then I want you to work on the following project:
1. are you familiar with the Japanese comic doodling style of drawing/animation? Isometric 3D, Simple and relatively thick lines, Simple colors, shaky boundaries for the objects. Confirm this first.
2. the animation I wanted is the CPO packaging and testing points demonstrated as a compact manufacturing line animation. Find the OFC 2026 notes and images from my local personal blog and find the picture on broadcom CPO assembling flow and test points.
3. Find the OSFP pigeon animation materials, where I have modeling of some of the chips. As well as modeling in [private deck] the most recent slides, and also in gary the scale-up architect the second last scene has the OE flying thru the assembling steps.
4. find the direct 3d modeling repo/playground folder where I have images of the nvidia CPO switch as well as the CPO package.
Ask me clarification questions you have after quick inspection and research. Draft a few different styles of still frame / modeling of the OE to let me confirm the style.
```

## 2. Source material found

Paths are relative to the parent of this PlayGround folder unless noted.

| Item | Location | Use |
|---|---|---|
| Broadcom "CPO Assembly Flow" slide | `jwt625.github.io/assets/images/2026/OFC2026/IMG_4084.JPG`, post `_posts/2026-04-04-OFC2026.md` line 3692 | Station order and the four Known Good gates |
| Broadcom reliability table, TH5-Bailly | same folder, `IMG_4085.JPG` | Reliability tests after the line, if wanted |
| Optical engine test points slide | same folder, `IMG_4125.JPG` (GF logo, Advantest 93K ExaScale + LightBeam) | Detail of what each test gate physically does |
| Gary film, S5 "back through the line" | `20261002_interconnect_film/outputs/v1/s05_wafer_test_v1_2_draft_p50.mp4` | Precedent shot; station list and CM300-style prober |
| Gary OE assets | `20261002_interconnect_film/assets/components/photonics/` (`oe_module_cpo`, `pic_die`, `eic_die`, `pic_eic_stack`, `fau_v_groove`, `els_laser_source`), `packaging/`, `fab_test/` | Dimensions and part list |
| Pigeon chip and assembly assets | `20261009_pigeon_pluggable/assets/components/optics_tx`, `optics_rx`, `assembly/`, `research/assembly_refs.md`, DevLog-019 | Bonder, active alignment, UV cure, test head references |
| NVIDIA Spectrum-6 CPO photos and model | `20261005_direct_3d_nvidia_switch/source/images/` (package close-ups `IMG_5711`, `IMG_5712`), `outputs/package_v4/` | Real OE look |

The private deck renders named in the request were read for reference only and are not cited or copied here.

## 3. Flow as drawn on the Broadcom slide (IMG_4084)

Row 1: EIC + PIC -> **Known Good Die** -> EEngine Assembly -> OEngine Assembly & Test -> **Known Good OE** -> ASIC to Substrate -> OEngine to Substrate -> Final CPO Assembly & Test -> **Known Good CPO**.

Row 2: CPO to System Assembly -> Fiber Cable Routing -> Final Assembly & Test (separate arrow in from a pluggable part, read as the pluggable laser source; `IMG_4085` lists "Pluggable Laser Source (PLS)") -> **Known Good System**.

Not stated on the slide: what the EEngine is physically and what the two square blocks added at the OEngine step are. Do not assert these on screen until resolved.

Test points from IMG_4125 (not Broadcom's slide): inline optical WAT (wafer acceptance test, grating coupled), PIC wafer sort (periscope probe), optic-attached sort, thinned wafer sort on a glass carrier, E-O test after MPC (Metallic PIC Coupler, per the slide at post line 4564) attach, double-sided optical engine sort (probecard + plug on one side, EIC side on chuck).

## 4. Pipeline for the drafts

Owner, 2026-11-01, mid-turn: "are you using image gen for the images? javascript preferred. See the hybrid bonding video in the advanced packaging repo". No image generation was used. A first Blender Freestyle pass was dropped and removed; drafts are now JavaScript.

- Same approach as the hybrid bonding short (advanced-packaging-atlas `media/hybrid-bonding-short`, `STYLE.md`, `scripts/render_video.py`): self-contained HTML scenes, Canvas2D, no libraries, no images, deterministic `DD.frame(t)`, headless Chromium capture.
- `lib/doodle.js`: true isometric projection, AABB painter sort, hand-line stroker, five style fills, line boil on 8 fps with a 4-frame cycle.
- `lib/models.js`: three OE models and the mini line. `scenes/*.html?style=A..E`. `scripts/render_stills.py` writes `outputs/style_drafts/`.
- Lettering font: `lib/fonts/xkcd-script.ttf`, copied from the local xkcd-font clone (CC BY-NC per that repo; check before any commercial use).

Exaggerations versus source assets: heights x2, fiber pitch x2 at the FAU. Ring and bump counts are illustrative. Mini-line machines are style placeholders, not reference-accurate tools.

## 5. TODO

- [x] Inspect references, write this log
- [x] Doodle OE model, generic open module (from `oe_module_cpo.json`)
- [x] Doodle OE model, Spectrum-6 look (from own photos)
- [x] Doodle OE model, Broadcom slide icon look
- [x] Style stills, OE and mini line, JavaScript
- [x] Comparison sheet and questions to owner
- [x] Owner picks style and answers questions, section 7
- [x] Reference ingest into the atlas repos, OE facts into the model
- [x] Loop GIF, first review cut

## 6. Progress log

- 2026-11-01: references located and read, plan written.
- 2026-11-01: Blender draft superseded by the JavaScript renderer on owner request. Five styles rendered; line boil checked by frame difference.

## 7. Decisions (owner, verbatim, 2026-11-01)

```text
1. style: A.
2. OE model: find a few most relevant references from [private] reference folder, and ingest the reference using skills in various atlas repo.
3. they are the EIC and the PIC.
4, sounds good. Each one should have a pile of reject next to them, with animation ejecting bad dies every few dies randomly. i.e., the full animation sequence could be long because of varying random rejections at different stations.
5. no vendor name. These are well known and the photo is from a public conference with thousands of people attending and many others writing reports about it.
6. It should be no music gif, with just the assembly line.
7. no narrative or music, see 6.
8. yes. It is illustrative and doodling/cartoon style.

single frame looped animation gif of the assembline line. Maybe draw 3~4 lines in parallel and with extent covering one full line so other lines running in parallel are partially visible.
```

Reading of item 3: EEngine = EIC, OEngine = PIC added; the OE on the line shows EIC and PIC as the two blocks of the slide icon, side by side on the carrier. To confirm with owner.

## 8. Loop design

- One fixed camera, GIF, no audio, no narration. 4 parallel lines; frame covers one full line corner to corner (the isometric +X axis at 30 deg matches a 16:9 diagonal), neighbors cropped.
- Stateless and periodic: `render(t)` derives everything from `t mod T`; a part's fate is a hash of (line, gate, part index mod N), so random rejects repeat exactly each period and the loop closes.
- Reject piles: each gate has one next to it. A flying reject lands on an identical copy already in its slot, so the pile looks constant and the loop has no jump.
- Buffers decouple rates: 2 dies per good OE, 8 OEs per package; a rejected package leaves a real gap downstream.
- Stations follow the slide's order; gates: Known Good Die, OE, CPO, System. Labels on machines, no vendor names.

## 9. References and OE model

Ingested into the packaging, EO modulator and PIC component atlases (uncommitted there): ECTC 2025 compact universal photonic engine, VLSI 2026 800 Gbps/fiber open-cavity transceiver, ISSCC 2026 23.1 3D-stacked 7 nm EIC / 65 nm PIC link.

- All three stack the EIC on the PIC: face-down hybrid bond in the ISSCC and ECTC papers, flip-chip in the VLSI paper.
- ISSCC stack drawing: PIC 6 x 10.7 mm, EIC 4.6 x 5.8 mm (printed labels look swapped), FAU beside the EIC over the grating-coupler rows.
- Connector type and assembly order are not reported in any of the three. The slide's EEngine-then-OEngine order is drawn as: EIC on carrier, lifted, PIC slides under, FAU on the PIC. That sequence is illustrative.

## 10. Render speed and layering (owner feedback 2026-11-01)

Owner: "did you benchmark the rendering? why tf is it taking so long?" and "some of the layering is wrong. Inspect carefully by regions."

Speed, `scripts/bench_render.py`, per frame at 1280x720, 1354 solids:

| Chromium flags | JS | Raster flush | PNG |
|---|---|---|---|
| swiftshader, copied from the hybrid bonding renderer | 31 ms | 2993 ms | 12 ms |
| none | 95 ms | 9 ms | 11 ms |

Root cause: swiftshader flags are for WebGL scenes; Canvas2D does not need them. Removing them took the full loop from 564 s to 17 s (360 frames, 3 processes). No benchmark was run before the first full render; it should have been.

Layering: painter order is only ambiguous where AABBs intersect, so `scripts/check_layers.py` lists those pairs. Started at 13199 pairs over 60 frames, now 0 over 300 frames. Real collisions fixed: adjacent systems overlapping at the 15-unit pitch, tool heads entering the chassis, kickers crossing rails, parts spawning inside the feed box and trays, belts into trays, pile items interpenetrating (piles are now z-separated pyramids). Kept the 2D renderer rather than moving to a 3D engine because the hand-drawn line comes from it.

Owner: "also speed up the animation by 2x (drop frame accordingly so same frame rate)". Every second frame kept: 15 s loop, 12 fps.

## 11. Layering root cause (owner feedback 2026-11-01)

Owner: "is the layering fixed? i still see many wrong parts", "the layering is also not stable, e.g. the stations' arms sometimes are correct and sometimes are not. The failed parts should be layered behind the assembling line at the front etc.", "some of the conveyer belt legs are also not stable in layering", "add one more production line at the bottom right, it should be visible partially."

Section 10 overstated this: `check_layers.py` only finds intersecting boxes, and zero intersections did not mean correct order. Root cause was in the sort: pairs were compared when their screen bounding rectangles overlapped. Those rectangles are huge for long diagonal belts, so pairs that do not overlap on screen added arbitrary order constraints, which formed cycles. The cycle breaker then drew parts wrongly, and moving parts changed the cycles frame to frame, hence the flicker.

Fix: compare only pairs whose projected hexagons overlap (ranges on x+y, y+z, x-z) and that a plane separates. `scripts/check_order.py` counts pairs drawn in the wrong order: 7804 over 30 frames before, 0 over the 180 GIF frames after. Parts in flight that pass through a machine add no constraint.

Extra line: owner corrected the position ("it should be the botton left along the same array"). It is the next row of the array, y = -140, partly visible at the bottom left.

Owner: "some of the failed CPO and failed system has red cross missing. Fix them." The always-present pile items were drawn unmarked. All four piles now mark every part as failed; system boxes also carry the cross on both visible walls so a box under another still shows it.

Owner: "zoomed out a bit too much, zoom in so that the top left and bottom right has near zero empty space. update aspect ratio if needed. Also add a final boxing and "shipping" (move box out of the frame) step." Frame now 1280x832, cut on the third row: feed box on the left edge, Known Good Die screen on the top edge, its system reject pile near the bottom edge. Slot 9 boxes the system; the closed box rides the belt out of the frame. The finished stack is gone.

Owner: "boxing animation should be improved, i.e., actually fold the box up and roll the tape over etc." Carton is now a real tube of hinged panels: opens from flat-packed in the air, drops over the system, short flaps then long flaps fold, tape head on a cross beam rolls along the seam.

Owner: "ok 33 MB is worth turning into a video. I will find a soundtrack." and "add an "outside five sigma" watermark on the final packaged box. No need to be very obvious, can make it grey etc." Output is now `outputs/cpo_line_loop.mp4`, H.264 yuv420p, faststart, 30 fps, no audio; GIF removed (rebuild with `--fps 12 --out outputs/cpo_line_loop.gif`). Watermark: grey hand lettering on the -Y wall of the closed box.

Open for discussion, not built: Known Good Die as a wafer-level test.

Owner decision on the wafer test: "B, wafer ride on belt, devent speed, arm/probe moves comically fast across the wafer, and another station right next to it slice wafer up comically fast while bad dies grow big and tossed, and good dies grow big to enter the next stage." and "keep the EIC PIC stage conveyer belt near same size as is right now." Built: EIC and PIC wafers alternate on the unchanged belt, probe raster marks each die, a DICING station right next to it saws 8 + 8 lines, dies pop to about 3x size in flight, bad ones onto the pile and good ones into the tray. Wafer and dies are small to fit the belt. Frame is now 1280x844 so the Known Good Die screen stays in frame.

## 12. Tempo and pace (owner, 2026-11-01)

Owner: "analyze the tempo of Mittsies - Vitality.m4a, and see if it matches or close to some of the parts in the assembly line... Ideally we crop out a duration that nearly loops, and has matching tempo to some of the stations." and "meanwhile dispatch subagents to work on pace of the animated stations too."

Track, measured only, not heard: 227.04 s, -11.7 LUFS, 150.00 BPM constant (grid contrast 5.24 versus about 1.1 at neighboring tempos), first downbeat 0.131 s. Onset autocorrelation in the loud sections: 0.2 s 0.84, 0.4 s 0.78, 0.6 s 0.71, 1.2 s 0.74, 0.3 s -0.12, so 0.3 s is off the grid.

Near-loop crops on bar lines: 8 bars 75.331-88.131 s (0.960); 16 bars 174.531-200.131 s (0.938). Scores are correlation of pitch and spectral pattern across the cut, by numbers only.

Beat clock in `lib/line_anim.js`: 16 bars, 25.6 s, 768 frames; beat = 12 frames at 30 fps. Provisional.

Owner: "give me 2 or 4 versions to choose from", then "starting at 0:13 of the track preferred, i.e., before the main strong theme kick in" and "or 0:19". Four previews in `outputs/pace_options/`, each playing its loop twice: on-beat and double-time paces, from 19.331 s for 8 bars (cut match 0.877, level jump 1.4 dB) and from 12.931 s for 12 bars (0.707, 1.7 dB). Beat clock only; per-station pacing from the two agents is not merged. Audio judged by numbers only.

Owner: "(out of the four options, the first one is the best and POR)": on-beat die and OE belts, track 0:19.331, 8 bars, 12.8 s. Owner: "two OEs getting placed around the ASIC per soundtrack beat", "update its duty cycle so that pitch/period is the same, but they move as fast as the OE station when they are active", "make them hit one beat per arm from left to right following each other", "do the same for various other components involved at the CPO stage, ASIC, enclosure frame, box etc., they should move fast when they are moving, and make idle time longer to match period", "be aware the fast arm (OE to substrate) should still be fast".

Built: package line one per bar, belt move in the first 8th; arm s hits on beat s % 4 + 1 with a 16th down and a 16th up, except arm 2 which taps on every OE landing; ASIC, ring, board, cover, modules, reject throws and the carton steps each move in about an 8th and land with their arm. Pacing agents' die and OE belt timing merged; their package-line timing, written for a 2-bar cycle, was not used. OE supply is 1 per beat and demand is 2 per beat, an illustrative mismatch.

## 13. Closeout

Owner: "this is great. Distill the experience and feedbacks thats potentially useful for future similar work into a skill, and register this project into the global video skill/memory." Global skill `iso-doodle-loop` written; project registered in the video library and the global memory index. Not committed.

Owner: "can we make the assembling/testing stations bodies sway left and right matching the tempo". Station bodies shear left and right in unison, full lean on every beat (2-beat period), base pinned; arm, head, lamp, screen, spool ride along; saw blade stays on its cut line. Saved as `outputs/cpo_line_loop_sway.mp4`; `cpo_line_loop.mp4` kept without sway, though `lib/line_anim.js` now has sway on (SWAY = 0 restores it).

Owner: "wrong swing, no curvature at all. the ladies from the game swings their hips one way and shoulders/heads the otherway (upperbody tilts a bit while head just translates), with legs fixed... They should have curvatures and stretches a bit. Give them some characters." Reference: `Helltaker mini-characters animation [720p60].mp4`, 2-beat cycle (48 frames at 60 fps), lower-body band moving about 14 px peak to peak against about 3 px in the head band, holds with a quick swing between, body rising in transit. Station bodies now have curved walls (`drawBent` in `lib/doodle.js`): pinned at the floor, hips at 38 % of height, shoulders at 82 %, head translating, 7 % stretch in transit, a fixed per-station amplitude factor. Replaces the rigid lean in `outputs/cpo_line_loop_sway.mp4`.
