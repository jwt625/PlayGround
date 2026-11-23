# DevLog-004: Wave 10: math rendering, presenter model, real simulations, literature visuals, popup pacing

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Delivered for review 2026-11-22; uncommitted since b726644 |
| Depends on | DevLog-003 (committed as b726644), `course/COURSE_SPEC.md` |

## 1. Instructions (verbatim, 2026-11-22)

```text
good. Cmt & push.
Then:
- render latex properly.
- build the directory model with higher res, either svg or blender similar to gary model.
- improve various interactive demo with more detailed model and more realistic simulations and results. e.g. Interactive: Rotate the x-cut die should do actual simulation with representative model profile. Check how 2d eigenn modes are solved in e.g. applied multiphysics repo or the luxretius repo, and implement proper anisotropy.
- for "what the literature reports", add proper visualization for each case to help with engagement. Crop figures if open access (CCBY) with citation, or reproduce the illustration and photos or figures from fresh with the reference using svg or blender or any other format you see fit.
- some time the ok/not ok popup is too fast, and comes up without any naarator explaining the scenario. e.g. "Module 3 | Respecting Boundaries: Mode Conversion in Bends
Scenario 1: The bigger ring". For similar slides,narator should describe and open with smt similar to "read/observe this scenario/meeting/conversation, and judge if gary's reaction is proper". Some of the examples should have gary react properly so that trainee dont default to always "not ok"
- note the PM fiber could launch elliptical polarization if input polarization is not aligned to the axis of the fiber.
```

```text
the "Module 4 | Photorefractive Conduct
Interactive: Resonance sweep" is the same for both direction at lower power, should this be the case?
```

```text
add a progress disk (circular ring that starts as full and opens & shifts clockwise to disappear when current narrative duration is done) to the cc box on the right most.
```

```text
whenever Pat Reynolds is on the slide, can we make them move similar to e.g. starcraft bottom control panel's character cam? Could be 3 simple animations prerecorded, and then play in random combinations during the narrative.
```

```text
3 short animations prerecorded, 3~5s?
```

```text
the RF probe modeling and renderings are horrible. Find actual probe pictures and drawings from GGB industries and MPI etc. and update the modeling and rendered pics.
```

```text
"Module 6 | Bias Point Retention
Incident video: The chip that was turned around" the animation should be rebuilt with better and closer to reality schematics/illustration for the chip (waveguide, electrodes layout should be topologically correct, and as close and consistent to real uscope images as possible). Similar quality requirements for other similar animation slides.
```

```text
the narrator added for ok/not ok slides should play as the page animation plays out. Currently the animations holds until narrator is finished.
```

```text
the fiber rendered in cover slide image of "Module 1 | Recognizing Unsolicited Feedback" is going thru the mounting plate as well as into and out of the breadboard. Fix it.
```

```text
simplify figure captions, e.g. "redrawn for this course. Regime order from the paper; spectra are illustrative sketches, not data." is too freaking long and unnecessary, just "redrawn" is enough. Apply similar simplification across the deck. Minimize distraction. Centralize disclaimers to the end. Trainee should enjoy the focus on the actual training content.
```

```text
"redrawn, illustrative" is enough
```

```text
for the incident video illustration on rods at 12 o clock TM polarization, the frame where it shows TM field direction and RF field direction, show actual simulated RF field lines and optical mode distribution, look up existing simulation configs and results (run it if needed) from EO modulator atlas or applied multiphysics repo.
```

```text
the landing page of each course should have some media and animations. Could be each module cover image fading in and out in order.
```

```text
once this is done, cmt and push, and then spin it off into a standalone repo and deploy as a subroute under my main page. (there should be a skill about such project spinoff and deployment).
```

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D31 | "directory model" read as the director, Pat Reynolds (host slides use a placeholder SVG). Build a Blender presenter in the same style as the Gary model, replace `hostSvg()` and the Pat avatar | Only "Director" character in the course; host frame is the low-res SVG |
| D32 | Math: KaTeX vendored into `engine/vendor/katex/` (works from file://). Markup `$...$` inline, `$$...$$` display, in displayed text only; narration stays plain speakable English (validator enforces no `$` in narration) | Static no-build, offline |
| D33 | x-cut die widget: offline full-vector anisotropic eigenmode sweep (Python, uv) over propagation angle on a representative x-cut TFLN rib, shipped as precomputed data (`.js`), widget shows real neff(theta) for the lowest modes and the field at the chosen angle | Real simulation without a heavy in-browser solver; data generated by code |
| D34 | Resonance sweep: steady-state model is correct that both directions coincide below the bistability threshold (a about 0.77 linewidth, power about 0.13 in widget units). Upgrade to a dynamic model with response time and sweep rate so slow photorefractive vs fast thermal history shows | His question; realism item |
| D35 | Literature figures: CC BY open-access figures cropped with on-screen citation and license (attribution required by the license); everything else reproduced from scratch (SVG / code plots / Blender) with "after Author, Journal Year, Fig. N" citation | His instruction; CC BY attribution must be adjacent |
| D36 | Popup pacing: every step with an OK / not OK popup opens with a narrator line telling the trainee to read or watch and judge; engine waits for narration and messages before the popup. About a third of judged scenarios get a proper Gary response (correct answer "OK") | His instruction |
| D38 | Caption progress ring: far right of the CC box; full at the start of each narration sequence (one run of lines including pauses), arc start moves clockwise so remaining arc = remaining time, gone at the end; driven by the audio clock (pause freezes, speed rescales). Assigned to lane E | His instruction |
| D39 | Presenter cam: 3 seamless loops of Pat from the same rig (idle: breathe and blink; talk: mouth and small nods; talk with an open-hand gesture), all starting and ending on the same neutral frame so any order joins cleanly. MP4 only (H.264 yuv420p, faststart, no audio), about 600x800. Engine plays talk loops in random order (no immediate repeat) while narration plays, idle when silent, still image as poster and for reduced motion. Empathy slides get idle and talk variants in the hand-on-heart pose | His instruction; random order needs a shared neutral pose |
| D40 | RF probe rebuild (lane RP): model from GGB Picoprobe / MPI TITAN / FormFactor product photos and dimensioned drawings (references kept in research/ref_probes/, never shipped); GSG probe with connector, body, coax, tapered three-needle tip; arm and manual XYZ positioner; probes land on pads from the long edges, fibers on the facets; fix every render with probes | His instruction; old probes were blocks with rods and collided with the fiber axis |
| D41 | All 24 code-drawn video steps (11 FE, 13 FB, about 97 keyframes) rebuilt on shared microscope-style drawing libraries `course/anims/fe_draw.js` and `fb_draw.js` (window.LMS_DRAW, loaded before the course files, also by validate.js and extract.js); topology rules: x-cut MZM arms in the two gaps of a G-S-G line, pads matching probe pitch, electrodes never crossing waveguides, correct facets, isolator direction, ring bus coupling; references from open-access micrographs, notes in research/ref_micrographs/ | His instruction, extended to all similar slides |
| D42 | Captions: citation plus a short tag only ("Redrawn, illustrative." for sketches, "Redrawn." for faithful reproductions, citation and figure number for CC crops); license links and all notes move to the final Disclaimer and Media Credits page; widget notes collapse to one line plus a closed "Model details" disclosure; disclaimer-style asides removed from bodies and figures; animation frames carry no disclaimer wording | His instruction: minimize distraction, centralize disclaimers |
| D43 | Rods-at-twelve TM frame: real simulated fields (lane SIMF): RF quasi-static G-S-G solve with anisotropic clamped LN permittivity plus TE0 / TM0 modes from the course solver, overlap factors; existing configs in his EO-modulator and multiphysics repos read first (never named in shipped files); exported as anims/fe_rods_fields_data.js + fe_rods_fields.js, hooked into the FE animation lane's keyframe | His instruction |
| D37 | Privacy: solver methods may be read from his other repos, but nothing in this public project names or links them | CLAUDE.md rule 13 |

## 3. Lanes

| Lane | Scope | Files |
|---|---|---|
| E Engine | KaTeX vendor + rendering everywhere text is shown; reveal-tab media (`media` field); popup gating after narration; validator rules | engine/*, player.html, index.html, tools/validate.js, COURSE_SPEC |
| S Mode solver | xcutIndex rebuilt on a real anisotropic mode sweep | tools/modesolver/, course/widgets/xcut_modes.js, data |
| W1 FE widgets | prSweep dynamics, eulerSign, pyroCalc, biasDrift realism | course/widgets/*.js |
| W2 FB widgets | fabryPerot, modeHop, doublet realism | course/widgets/*.js |
| H Host | Pat Reynolds Blender model, host frame stills, avatar | tools/blender/, media/render/, media/avatars/ |
| L1 FE literature | Visuals for each tab of the three literature reveal steps | media/lit/, ferroelectric.js (those steps only) |
| L2 FB literature | Visuals for literature-reporting tabs | media/lit/, feedback.js (those tabs only) |
| C1 FE content | Popup intros, balanced OK cases, PM fiber elliptical note, math markup | ferroelectric.js |
| C2 FB content | Popup intros, balanced OK cases, math markup | feedback.js |
| Coordinator | TTS catch-up, validator, browser checks, audits | |

## 4. TODO

- [x] E, S, W1, W2, H, L1, L2, C1, C2 lanes
- [x] Presenter cam (HC), RF probe rebuild (RP), fiber routing fix, caption simplification, FE / FB animation rebuilds (AF, AB), simulated rods fields (SIMF)
- [x] TTS catch-up for changed narration, orphan cleanup
- [x] Validator, full browser sweep of every step in both courses
- [x] Fresh-context audit (physics of solver and widgets, citations and licenses) and fixes
- [ ] Taxonomy drag-and-drop quizzes (proposal awaiting his go)
- [ ] Commit (needs `git add -f` for jpg / png / mp3 / mp4 / svg-adjacent ignored types)

## 5. Progress

- 2026-11-22: committed and pushed wave 7-9 as b726644. Plan written.
- 2026-11-22: coordinator added the widget override registry (`window.LMS_WIDGETS`, `window.LMS_WIDGET_SPEECH[courseId]`, collected by speech_lines.js and tools/tts/extract.js), placeholder files in `course/widgets/`, literature credit stubs `media/credits_lit_{fe,fb}.js`, and script tags in player.html. Nine lanes launched (E, S, W1, W2, H, L1, L2, C1, C2).
- 2026-11-22: answered his resonance sweep question (D34): steady-state model is correct below threshold; dynamic upgrade assigned to W1.
- 2026-11-22: lane E done. KaTeX 0.16.11 vendored (engine/vendor/katex, engine/math.js with MutationObserver pass, LMS_math / LMS_mathPlain); reveal-tab `media` with lightbox and credits; `course.host.img` / step `hostImg` with SVG fallback; popup pacing (chat with ask: narration, then messages, then popup after 800 ms of silence; bigger ring: narration end 8841 ms, last message 10162 ms, popup 11021 ms); caption progress ring (verified clockwise, hides at end); validator rules (no `$` in spoken strings, KaTeX parse, tab media registry, ask needs narration, widget speech registration). Both validators 0 errors. Open: ring jumps on unrecorded lines until TTS catch-up; woff/ttf fallbacks listed in KaTeX CSS not shipped (woff2 used).
- 2026-11-22: he asked whether each misconduct category is gone through individually and asked for drag-drop classification quizzes. Answered: categories are covered individually in both category chapters; classification sorts are partial (FE split 4+4, FB only categories 1-4) and other taxonomies (four hazards, simulation hazards, temperature places, five regimes, reflectors) have none. Proposal awaiting his go.
- 2026-11-22: lane C2 done (feedback.js). All 33 OK / not OK popups (all on chat steps) open with a medium-specific intro ("Read this email thread... Then judge whether Gary's response is proper"); post-processing throws if a judged chat lacks an intro. 10 of 33 scenarios flipped to correct answer OK (PROPER map): Isolator Placement, Certified Single-Mode, categories 3/5/6/8, The Datasheet, The Contrast Ratio, The Snap, The Calm Wavemeter. Math converted to $...$ on displayed text (etalon, Fresnel, Q extraction, FSR, Snell, cos^2). Left plain: exam pool options, canvas labels. 82 new lines to record. Validator OK.
- 2026-11-22: lane C1 done (ferroelectric.js). All 35 OK / not OK steps (11 videos, 24 chats) open with a "read / watch this ..., and judge ..." intro. Balance now 10 OK, 22 not OK, 3 "only if": flipped Animation, Notes to self, Polarization blame, Routing shortcut, Lock that ran away, Neighbor, Drift-free claim, Ignored volt, Template frame (Kitchen foil already OK). PM fiber: misaligned linear launch splits between the two eigenmodes and exits generally elliptical, varying with length, wavelength, temperature, stress; PER = 10 log10(Pmax/Pmin) added; rods video and Dana's message fixed. Math converted on displayed text. Narration +640 words (about 4 min); course about 98 min. About 75 lines to record. Validator OK.
- 2026-11-22: lane W2 done. fabryPerot: Fresnel from real indices (cited), lossy Airy, tunable-laser sweep with aliasing warning. modeHop: Lang-Kobayashi steady state, modes / anti-modes, up/down sweeps with hysteresis, regime map with C = 1 line and Tkach-Chraplyvy levels (default -55 dB, 30 cm fiber, alpha 4: C = 3.78). doublet: CW/CCW coupled-mode theory, through / reflection / drop, single-Lorentzian fit showing Q misreport; presets Ji et al. Optica 2017 and Bahadori et al. 2018. Sources moved from scratch to `course/tools/widgets_src/fb/` (build.sh rebuilds identical files). Unverified DOIs listed for the audit.
- 2026-11-22: he asked to keep subagent memory use low (machine shared); limits sent to all running lanes (one heavy process each, 2 solver threads, one Blender at 4 threads, one browser at a time). Saved as a memory rule.
- 2026-11-22: lane W1 done. prSweep: time-domain model (Sun et al. Opt. Express 2017 thermal 10 us; photorefractive four components 25.7 ms to 28.4 s weighted per Jiang et al. Opt. Lett. 2017), sweep time 10 us to 1000 s, ten-pass triangle scan, dark relax; verified below threshold: 10 ms sweep coincides, 1 s sweep differs. biasDrift: two-layer Maxwell-Wagner (Holzgrafe et al. 2024; tau 1.6 ms oxide-clad), constant V / range-limited controller / heater. pyroCalc: sigma = p dT, E = sigma/(eps0 eps33) with screening ODE; 100 K gives 16.2 kV/mm. eulerSign: full 3m r-tensor rotated about x vs transposed (wrong) rotation; 45 deg 39.8 vs 35.0 pm/V. Unverified from primary: Paschen 327 V (Wikipedia), Weis and Gaylord / Volk values, Popescu via Zhu 2021.
- 2026-11-22: lane H done. "Gary model" = chunky clay character kit v2 from the interconnect film; copied into `course/tools/blender/charkit/` (one import renamed). Pat Reynolds built with it at subdivision 2: silver bob, brown rectangular glasses, pearl necklace, navy blazer with Five Sigma lapel pin, office set with the bell-curve sign. Renders `media/render/host_pat_present.jpg` (all host slides via `course.host.img`) and `host_pat_empathy.jpg` (hand on heart; "How do you think Gary felt?" slides via `hostImg`), 1200x1600, about 36 s each on GPU; avatar `pat_reynolds.jpg` re-cropped from the render. Script `tools/blender/shot_host_pat.py` defaults to 4 threads. Caveat: rerunning `tools/avatars/draw.py` or `write_cast.py` would overwrite the new avatar and its credit (both tools predate the mail fields in cast.js).
- 2026-11-22: lane L2 done. 24 feedback-course literature tabs plus the Marinins slide have a figure: 10 crops of openly licensed figures (Xu 2021 Sci Rep, Franken x2, Lapointe x2, Raja 2019, Kramnik, Chen 2026, De Koninck: CC BY 4.0; Bao 2024 arXiv: CC BY-SA 4.0) and 16 redrawn SVGs (Optica / IEEE / Nature / IOP papers are not CC BY). Licenses checked via Crossref / OpenAlex then the article page. Captions "Adapted from ... CC BY 4.0" or "Schematic after ..." / "Model plot drawn for this course ..." with DOI links. Scripts `tools/lit/fb_crops.py`, `fb_draw.py`; raw CC sources in `research/lit_sources/fb/` (28 MB, README). Seven tabs skipped (no cited source). Seven new narration lines. Open: Bao is BY-SA (swap if strict CC BY wanted); preprint figure numbers / values may differ from published versions.
- 2026-11-22: lane S done. `tools/modesolver/fdmode.py`: Yee-grid FD, all three E components, full rotated tensor incl. eps_xz, quadratic eigenproblem in beta via companion linearization + shift-invert Arnoldi; sub-cell sidewall averaging; Zelmon 1997 / Malitson 1965. Validation: isotropic slab vs analytic 8e-6 (5 nm grid); anisotropic slab vs 4x4 transfer matrix within 1.1e-5 incl. hybridized pair; rib at 0/90 deg vs transverse solver 1e-7; crossing angle vs grid 30.96/29.89/30.25/30.21 deg. Shipped profile 800 nm film, 400 nm etch, 1.2 um top, 65 deg sidewalls, oxide below, air above, 1550 nm: TE/TM anticrossing at 30.3 deg (and 149.7), min splitting 0.0027. The suggested 600/300 nm rib does not cross (closest 0.033 at 0 deg), consistent with Wang et al. Nature 2024 (mixing above about 700 nm film); kept as comparison in the note. Data 127 angles, 1.16 MB. Widget: die/ring top view, neff vs angle colored by TE fraction with zoom, four field maps. Defaults now 1 worker, 2 threads (peak 2.4 GB on 15 nm grid).
- 2026-11-22: lane L1 done. 13 ferroelectric literature tabs have a figure: 8 CC BY 4.0 crops (Wang Nature 2024 Fig. 2g, Wang IEEE PJ 2020 Fig. 6bc, Liu LAM 2023 Fig. 3e, Xu arXiv 2020 Fig. 5, Shams-Ansari arXiv 2022 Fig. 2, Zhang arXiv 2026 Fig. 2de, Liu ACS Photonics 2026 Fig. 5a, Zheng arXiv 2024 Fig. 2) and 5 drawn (Xie AWG schematic, Liang stated levels, Valdez computed MZI, V pi electrical dB, Weis r33 ranges). Fact fixes: Liu 2026 used stepped voltage for constant electrical power (not constant voltage); the adiabatic bend picture is stated by Wang 2020 (credited, new ref R.wangPJ2020). Scripts `tools/lit/fe_crops.py`, `fe_figs.py`; sources `research/lit_sources/fe/` (PDFs git-ignored; one local copy carries a university download watermark, not committed, not in the crop).
- 2026-11-22: coordinator removed superseded in-course widget code (feedback.js fabryPerot / modeHop / doublet; ferroelectric.js the five widgets and the old WSAY speech list). Validators OK; credits page now lists 13 + 26 literature figures and both host renders. TTS catch-up running (188 lines, 2 threads).
- 2026-11-22: TTS catch-up: 188 clips (5 min wall, 2 threads), second pass 0; both courses fully recorded (FE 1474, FB 1528; 2978 clips). 58 superseded clips removed (none in manifest).
- 2026-11-22: his note "the narrator added for ok/not ok slides should play as the page animation plays out. Currently the animations holds until narrator is finished." Coordinator changed R.chat: for steps with ask, the intro narration starts together with the messages; the popup waits for both, then the 800 ms silence. Verified on "Scenario 1: The bigger ring" (dev mode): messages start at 21 ms with the intro caption, narration ends 7703 ms, popup 8561 ms. COURSE_SPEC updated. Video steps already play alongside narration.
- 2026-11-22: fresh-context audit done. No physics or formula errors found in widgets, mode solver (independent slab check matched to 1e-4; thickness crossover about 0.70 um air top consistent with Wang 2024), OK popups or displayed math; all 39 literature DOIs resolve and every CC claim holds for the version cropped. Findings: ignored file types need `git add -f` at commit (lit crops, host renders, new mp3, modesolver logs); citation fixes (SiN index is Luke 2015 not Ji 2017; Ren npj Nanophotonics 2025; Tkach numbers tab vs widget conflict; Ji 2017 does not support the Q misread sentence; Hausmann 2014 claim unverified); overclaims (Paschen "any gap", Shen, Lu 2019, Liu 2026, Liu 2023 sidewall symmetry, "grounded lithium niobate", r14/r41 are zero in 3m); missing CC license links; crop clipping; Xie SVG mismatch; arXiv TeX sources in research/lit_sources/fb to remove; nits. Fix lane launched.
- 2026-11-22: audit fix lane done. Citations: SiN index to Luke 2015; Ren 2025; Kramnik now Nature Electronics 8, 620 (2025); Hausmann claim verified and hedged to the paper's wording; Ji 2017 limited to the splitting. Overclaims fixed (Shen, Lu 2019, Liu 2026 contact resistance, Liu 2023 symmetry argument, grounded shields not "grounded LN", c14 / p14 / p41 instead of r14 / r41, Paschen for gaps above a few um, Franken run conditions, Lapointe projected loss). `licenseUrl` on 73 CC credit entries with "License" links in captions, lightbox and credits; six crops re-cut; Xie SVG fixed; arXiv TeX sources removed except the three figure PDFs the crop scripts read. Marinins volume not confirmable (kept 2022). About 15 narration lines need TTS.
- 2026-11-22: fiber fix lane launched for butterfly_isolator_fc_bench.jpg (pigtail through the mount plate, segments dipping below the breadboard) and sibling renders, with an explicit fiber-vs-solid / table clearance check in the script.
- 2026-11-22: lane RP first pass done: probes modeled on GGB 40A/67A/110H and MPI TITAN / MP60 drawings (references in research/ref_probes/, git-ignored, spec table with sources), die electrode bent to N/S pads, probes land from the long edges, six renders redone plus new tfln_probe_station_wide.jpg (Module 6 opener now uses it), hotspots re-projected. Coordinator review found the CPW reads as one gold band (no visible G-S-G gaps or waveguides), termination needles near the pad edge, a fiber through a holder bar and a mid-air fiber loop in the wide shot, "+Z" half hidden: sent back for a second pass. Die layout shared with the FE animation lane.
- 2026-11-22: caption simplification lane launched (D42); animation lanes told no disclaimer wording in frames. Saved memory rule on minimal captions.
- 2026-11-22: caption lane done. 39 captions cut to citation plus tag ("Redrawn.", "Redrawn, illustrative.", "Computed, illustrative.", or figure number for crops); License link removed from captions and lightbox (kept on credits page); 25 registry `note` fields carry the moved detail and print on the final page; one disclaimer paragraph explains the tags. 8 widgets: one-line note plus closed "Model details". Asides reworded (bend anticrossing, offset splice, Lapointe / De Koninck cites). 8 SVGs regenerated without "sketch / not to scale" labels. 4 narration lines changed.
- 2026-11-22: lane AB first pass done: 13 feedback videos (52 keyframes) on `course/anims/fb_draw.js` (23 scenes; flip-chip InP on SiN with section and grating inset, GSG MZM with reflector-pair ripple table, mode-hop staircase model, ring doublet from coupled-mode theory, isolator benches, OSA truncation); old helpers removed; references in research/ref_micrographs/README_fb.md. Coordinator review: "not to scale" / "scale expanded" labels left in frames, a waveguide drawn through a grating coupler in the flip-chip top view, overlapping labels: sent back for a fix pass.
- 2026-11-22: lane HC done. Five 600x800 24 fps H.264 loops (idle 4 s, talk 3 s, gesture 5 s, empathy idle 4 s, empathy talk 3 s; 160-300 KB each; 34.5 min total render, one Blender) start and end on the still's neutral pose (decoded join differences at codec-noise level). Engine `course.host.cam` / step `hostCam`: talk clips (gesture weight 0.5, never twice in a row) while narration is busy, idle otherwise, swap only at clip end on first decoded frame, pause-aware, reduced-motion and load-failure fall back to the still, works from file://. Teeth hidden (they poked through the lip). Commit needs `git add -f` for *.mp4.
- 2026-11-22: fiber lane done. Root cause in shot_butterfly.py: AUTO-handle Bezier through sparse points (chord through the plate corner, overshoot below the table up to 1.4 mm, bend radii down to 4.3 mm, splice sleeves floating). New `route()` (explicit handles, plan-view arc-straight-arc, flat height where entering components) and `check_routes()` (0.25 mm sampling vs table, exact solids, continuity, axis, 20 mm min bend): 26 violations to 0. Also fixed backwards-tapered boot, mismatched isolator boots, FC nuts buried 1.55 mm (`fc_rest()`), D-sub cables through the plate. Three butterfly renders redone (about 12 s each). Sent back to apply the same to fc_apc_vs_pc_ferrules.jpg (nuts 1.67 mm into the mat).
- 2026-11-22: fc_apc_vs_pc_ferrules.jpg fixed: connectors posed with fc_rest() (nuts on the mat, 7.4 deg tilt), jackets via route(), check_routes 0 violations, 10 s render; APC 8 deg face still reads.
- 2026-11-22: lane AB fix pass done: disclaimer wording removed from frames (FSR readout now DESIGN), flip-chip top view routing fixed (device coupler separate from the loopback pair on the 127 um pitch line), dashed thin leader lines, overlap fixes across 15 scenes at desktop and phone width; 52/52 keyframes draw with 0 errors; feedback.js validator OK.
- 2026-11-22: lane RP second pass: G-S-G now reads at bends, feed and pads (active 8 um gaps kept physical), needles verified on conductors numerically, clearance tool check_clearance.py (0 violations), closeup camera shows +Z / RF. Coordinator review: wide shot still has a jacket ending in mid-air and a fiber hanging from the top edge, bend radii 6-10 mm: sent back for a third pass on the wide.
- 2026-11-22: lane RP third pass done: wide shot fibers clamp -> along holder bar -> platen -> off frame; RF cables up and over the positioners; min bends coated fiber 15.6 mm, jacket 20.3 mm, RF cable 26.0 mm; 0 intersections; wide and closeup re-rendered. Coordinator looked at the wide: no floating or hanging fiber.
- 2026-11-22: lane AF done: 11 ferroelectric videos (46 keyframes) on `course/anims/fe_draw.js` (parametric die from tfln_die.py dimensions; arms leave the gaps straight and pass under the bent line through thick-oxide windows after Nelan 2022; termination resistor; light pulse and RF packet; traveling-wave EO response with -3 dB near 6 GHz and nulls at 13.5 / 27 / 40 GHz for counter-propagation). Fresh audit inside the lane: 19 issues, 18 fixed (open: "Many hands" props not drawn). Rods keyframe 3 calls LMS_DRAW.fe_rodsFields when present. 11 micrograph references in research/ref_micrographs/README.md.
- 2026-11-22: TTS catch-up 18 lines (57 s); 18 superseded clips removed; both courses fully recorded.
- 2026-11-22: full browser sweep (dev mode, every step of both courses, 201 each): 0 page errors, 0 failed requests, 0 unrendered math.
- 2026-11-22: lane SIMF done: anisotropic quasi-static RF solve of a G-S-G line (600 nm film, 5 um gaps, clamped eps 44.3 / 27.9) plus TE0 / TM0 from the course solver and a full-tensor Pockels perturbation; matches an independent finite-element reference case within 0.2-0.8 %. Gamma TE 0.505, TM 0.433; Vpi(TM)/Vpi(TE) 2.80 (RF) and 2.62 (low frequency), consistent with the narration's 'nearly three times'. Frame `anims/fe_rods_fields.js` + data (117 KB) hooked from fe_draw.js rods keyframe 3; method in tools/fields/README.md.
- 2026-11-22: start screen backdrop: each chapter's first step image cross-fades in order (1.6 s fade, 7 s push-in, 5.2 s per cover) under a radial veil, module label bottom left; reduced motion shows the first cover still; timer cleared on step change. Checked both courses at desktop and 390 px, 8 covers each, no errors.
