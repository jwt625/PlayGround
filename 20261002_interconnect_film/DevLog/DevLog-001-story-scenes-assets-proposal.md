# DevLog-001: Storyboard, scene timing, narration, asset plan

| Field | Value |
|---|---|
| Date | 2026-10-02 (rev 3, after Wentao's second review) |
| Status | Storyboard + plain-English narration v1 for review; nothing built; sound/speech deferred |
| Depends on | `DevLog-000-unhinged-interconnect-explainer-kickoff.md` (research, provenance; `$BLOG` = `../../jwt625.github.io`) |
| Authors | Claude, at Wentao Jiang's request |

All on-screen numbers are traceable to DevLog-000 Section 4 (blog) or 5.4 (standards/vendor). "(vendor)" = vendor claim. "Derived" = arithmetic from sourced figures. Anything not traceable is TBD or deliberately shown without a number. Numbers spoken in narration are jokes, not data claims (Section 5).

## 1. Decisions from Wentao (2026-10-02)

Rev 1 (scene content):
- Chapters approved. LPO skipped. 10 s per scene, super fast short-form pace. Mobile aspect, at least 1:1.
- Literal clay-form shotgun. No blood: a comical clean bullet-through-hole in Gary when shot.
- No detailed trace display; any scope view is quick, then the camera returns to human spatial and time scale consequences.
- Sound and speech come next. Now: storyboard and text per scene, timing as tight as possible.
- Retimer scene: retimer chips at every connector launch, more and more as the camera moves through trays, particles and glow; pan to scope, eyes open, then latency and energy per bit through the roof; Gary shot.
- NPO scene: Gary fights vendors over laser in-package vs disaggregated, connector form factors and die size, 2D vs 3D packaging (side by side to stacked) while lead time and cost blow up; Gary shot.
- CPO: heat to keep MRMs stable against GPU/ASIC power swings; NPO moves onto the organic substrate; zoom on GPU + HBM with violent workload/temperature swings and smoke; zoom slightly out to the hot optical engines; fried eggs on them; MRM yield via motion-blur shuffle back to the SiPho wafer under a wafer-level tester; Gary shot.
- Final scene: fiber mess; Gary whipped while tying fiber bundles in a pulled-out tray.

Rev 2 (answers to Section 8 of rev 2):
1. S1 and S4 also end with Gary shot (every scene except the last). S1 includes a shot of the NVL72 backplane copper cartridges; S4 includes the zoomed-in CPO shot.
2. Eggs are fried eggs, scaled to fit onto a hot optical engine; animate frying (egg white from transparent to white, edges turn brown, steam).
3. Thermal runaway scene is cut (60 s total, six scenes).
4. Parody logo "NARROWCOM" with the waveform tip flipped upside down on the retimer chips.
5. 4:5 aspect confirmed.
6. Narration rule: plain non-technical English; narrate what Gary is trying to do and what happens to him; technical terms appear only as brief labels or after effects. Examples given: "now the signal takes forever to get through and cranks up the electricity bill"; "now Gary has to wait 3 more months and pay his year end bonus to the vendor"; "now Gary is moving the optics even closer to the logic"; "only one out of ten rings work". Rev 2 lines judged cringe and replaced.

Rev 3 (answers to Section 8 of rev 3):
1. "Whipped" means the Manager literally whips Gary (not a snapping fiber bundle). S6 narration line "They fight back" is replaced accordingly (Section 4, S6).
2. Add a final 2 s ad-style disclaimer scene (S7): ultrafast narration, black screen with white disclaimer text, and the list of actual technical sources in smaller font.
3. Build a simplified crude Blender project now (key objects, camera moves, narrative text; no detailed modeling or VFX, with in-scene text labels marking where VFX will go) so Wentao can watch and give feedback. Build notes and results: `DevLog-002-crude-blender-build.md`.

Rev 4 (v0 feedback by timestamp, 2026-10-02; implemented in crude v0.1):
- 0:03 S1: the propagating pulse decays along the cable; at least a dozen cables between the racks.
- 0:06 S1: scope eye shakes like a measured eye and only nearly closes; BER label on the scope climbs as the eye shrinks; BER is a real Gaussian estimate from the simulated traces.
- 0:08 S1: longer frame of the Manager staring at the scope, framing scope and Manager together; Manager's face turns angry comically.
- 0:10 holes: they recover slowly across the film (can change at cuts) but never fully heal, so Gary keeps accumulating holes.
- 0:12 S2: set inside a rack, through the trays vertically near the backplane connectors (a row of about 4 connectors per tray), each getting 2-3 rows of retimers behind it; rows increase as the camera rises through multiple trays; after the camera exits the rack, Gary stares at the recovering eye and the Manager stares at a big whiteboard behind them with the latency and energy-per-bit curves climbing; then he gets shot.
- 0:23 S3: connector form factors use real different BGA or LGA packages; vendors wear shirts with real logos or slightly modified parody versions (Molex, Nubis/Ciena, TeraHop, etc.).
- 0:29 S3: this time Gary shoots himself in the head.
- 0:30 S4: proper mock-up of a DGX/HGX-style board with random regulators, MLCCs, small coils and transformers.
- 0:34 S4: zoomed-in PIC with rings on it, rings randomly changing color.
- 0:37 S4: one egg per OE; OEs change color and glow when heated; use the actual OE count, at least 8 + 8 on the sides of the XPU.
- 0:39 S4: Gary throws the eggs up as he is shot; they fly out of frame and back in and drop on the Manager's head.
- 0:41 S5: steps are reflow on substrate, FAU attach, EIC/PIC bonding, dicing and tape merged into one step, and lastly the earliest step, wafer-level test (WLT); use or model the FormFactor CM300 (reference image `../references/formfactor-CM300.png`).
- 0:46 S5: show actual Lorentzian ring lineshapes, with new measurement traces appearing here and there and a tight green spec/pass window.
- 0:59 S6: the Manager whips Gary with a bundle of active optical cables; customers slap the Manager (NVIDIA-shirt customer) while an OpenAI/Anthropic-shirt customer hugs the NVIDIA customer's leg.
- Naming: this version is "v0.1" (`scenes/crude_v0_1.blend`, `outputs/crude_v0_1_20261002.mp4`).

Rev 5 (answers and v0.1 feedback by timestamp, 2026-10-02; implemented in crude v0.2):
- Answers: 16 OEs total (8 + 8); TERAHOP confirmed; parody logos are my choice (drawn as text wordmarks and a vector valley mark); whiteboard curves need no y-axis units or tick labels, and the latency and energy labels ride next to the newest point and move right with it.
- 0:06 S1 (and all later scope shots): only one set of eye traces on a scope screen. (v0.1 showed an open "good" eye overlaid on the closing one, caused by the Gaussian-ISI model; v0.2 closes the eye by noise and jitter instead.)
- 0:12 S2: camera fully inside the rack, more zoomed in, no front side in frame; retimers appear slower and more dramatically, with move-into-place and fade-in; the 3rd row appears when the camera nears the top tray.
- 0:17 S2: the scope sits on a proper lab bench, not a pillar.
- 0:26 S3: vendors shout, shove and beat each other; two shoot the other two; the Manager shoots one of the remaining two while Gary shoots himself.
- 0:30 S4: interposer / package layout like Rubin Ultra (`../references/rubin-ultra.png`).
- 0:33 S4: do not jump to the PIC: zoom actually onto an OE from the previous frame, then fade in and out quickly to the PIC; ring colors come in waves from left to right, with a similar wave on the substrate, to imply global heat rather than random per-ring modulation.
- 0:44 S5: the wafer moves around on the stage instead of the probe head covering the wafer; wafers come in and out quickly.
- "Start adding more details" and "call this v0.2": see DevLog-002 for the detail upgrades.

## 2. Format and style (measured from the reference Short)

Reference file: `../references/JE3LSo1Rp54.mp4` (Fuji Beyblade Short). Measured: 2160x3840 (9:16), 29.97 fps, 1516 frames, 50.6 s, audio stream present (audio not analyzed). Frame review (1 fps contact sheet) and cut detection show:

- Practically no hard cuts (one detected above scene-change threshold 0.25, at 11.1 s). Transitions are continuous camera moves, motion-blur whips, and match-moves between locations.
- A caption chunk about every 1 s, 1-3 words each, white bold sans with dark drop shadow, placed about 75-80% down the frame. The opening 10 s carry 22 words (about 2.2 words/s).
- Semi-realistic glossy 3D with strong glow/particle bursts on key events; saturated sky-blue and earth palette; no characters in this Short.
- Style hybrid for this film: same continuous-camera, caption-every-second grammar; clay characters and clay props (Gary, Manager, shotgun, eggs), glossy PBR hardware with glow and particles.

Spec:

- 1080x1350 (4:5), 30 fps, 10 s = 300 frames per scene; all key action inside the centered 1080x1080 safe square so it crops cleanly to 1:1; captions in the lower third (~75-80% height) inside the safe area.
- Narration budget: 18-23 words per scene, about 2.5-2.9 words/s over an 8 s window (faster than the reference's 2.2, per "super fast pace"); the shot and whip moments carry no narration, only SFX.
- Hit cadence: one visual event every ~0.5-1.0 s; one camera move per beat; no static shot longer than 1.0 s; any scope view under 1 s.
- Gary-hole continuity: every shot leaves a clean see-through circular hole in the clay (no blood), at a new spot on his body. Holes persist across scenes (count per scene in the Section 3 table).

Scenes and runtime: S1 copper, S2 retimers, S3 NPO, S4 CPO heat, S5 CPO MRM wafer, S6 fiber mess = 60 s, plus S7 disclaimer 2 s = 62 s.

## 3. Cast and props

| Element | Spec |
|---|---|
| Gary | Clay technician, simple rig, one outfit; reaction faces (calm, sweat, dread, flat). Holes accumulate: 1 after S1, 2 after S2, a head hole after S3 (self-inflicted), 4 after S4, 5 after S5; each recovers slowly (shrinks in steps, floor 30%) but never heals; in S6 a fiber threads through one of them |
| Manager | Clay, loud tie; in-frame at human scale only (never inside hardware shots); owns the clay shotgun (S1-S5) and a whip (S6) |
| Clay shotgun | Single prop, clay texture with thumbprints; muzzle gives a clay-smoke ring plus "BANG" caption; recoil squash; the shot lands as an instant clean hole in Gary |
| Scope | Wall-mounted generic sampling scope; eye on screen for under 1 s per use |
| Pulse | Tiny clay signal courier (S1, S2) |
| Vendors (S3) | Four clay vendors in colored shirts with parody wordmarks: MOLEXX, NUBISS, TERAHOP, AYARR (not the real logos); they shout, shove and punch, then two shoot the other two |
| Customers (S6) | Clay person in a green NVYDIA shirt (slaps the Manager); smaller clay person in a dark OPENAY / ANTHROPY shirt (hugs the NVYDIA customer's leg) |
| Eggs (S4) | Clay fried eggs, scaled to fit onto an optical engine |
| Retimer chip | Black mold clay-like package; laser-etched gold wordmark "NARROWCOM" (parody: the waveform tip of the real logo flipped upside down); generic part-number text only |

Branding: no real logos on any model. The parody mark is drawn from scratch as vector (not traced from the supplied Broadcom chip photo; the photo is a material reference only: matte black mold compound, shallow gold laser-etched lettering, rows of marking text, surrounding blue PCB and passives). Not legal advice; parody is the intent.

## 4. Scene-by-scene storyboard (rev 5, implemented as crude v0.2)

Notation: t = seconds within the scene (frame = t x 30). "Caption" = lower-third text chunks (1-3 words). "Label" = brief technical term as an after effect or sticker. "Card" = small footnote data card with a source tag. Narration is plain English (Section 5); all jargon lives in captions/labels/cards. Rev 5 incorporates Wentao's v0.1 feedback (by timestamp) and is what `crude_v0_2` implements.

Cross-scene rules:
- Holes slowly recover but never fully heal: after each hit a hole shrinks in steps (x0.9 every 1.5 s, floor 30% of its original radius), so Gary keeps accumulating holes.
- Scopes sit on a proper lab bench (top, legs, shelf, back panel, bench scope with knobs, mug, multimeter, cable spool), not a pedestal.
- Each scope screen shows exactly ONE set of eye traces. The eye closes (or recovers) by noise and jitter growing (and a mild bandwidth change), so it stays one coherent fuzzy eye instead of an open eye overlaid with a closing one. The printed BER is the Gaussian estimate from the same simulated samples, Q = (mu1 - mu0) / (sigma1 + sigma0), BER = 0.5 erfc(Q / sqrt 2). Illustrative model, not a measurement.
- Brand marks on people and chips are parody marks, never real logos: shirt wordmarks MOLEXX, NUBISS, TERAHOP, AYARR (vendors), NVYDIA and OPENAY / ANTHROPY (customers); NARROWCOM chips and sign with a valley mark (the arch tip flipped upside down) over a wave line.
- The floor is a checker pattern (1 m tiles x2) for scale cues; muzzle smoke rings, ear steam, egg bubbles and Manager/Gary noses and ears are the first small detail upgrades.

### S1 Copper: the stretch (0.0-10.0)

Narration (22 words): "Gary wants faster data over copper. But faster means a shorter wire. So Gary tries stretching it one more meter. Bad idea."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-2.2 | Hook: fly along the NVL72 backplane copper cartridge wall (cartridges with ports and handles); Pulse sprints along a cable; Gary stands in front at the end | Fast FPV fly-along, then snap-stop onto Gary | "COPPER" ; card: "NVL72 backplane: about 5,000 copper cables, 2 miles (SemiAnalysis)" |
| 2.2-3.0 | Two racks 5 m apart joined by a bundle of 14 cables; Pulse decays along the way (about 0.2 of start amplitude) | Wide at about 6.5 m | "5 m" ; card: 802.3bj |
| 3.0-3.6 | Racks at 2 m; Pulse decays less (about 0.5) | Whip pan | "2 m" ; card: 802.3ck |
| 3.6-4.2 | Racks at 1 m; Pulse barely decays (about 0.7) | Fisheye push-in | "1 m" ; card: 802.3dj objective |
| 4.2-6.0 | Gary pulls the rack one more meter; the bundle goes taut | Handheld close | "STRETCH" |
| 6.0-6.8 | Bench scope close-up: one eye that shakes and fuzzes toward closure (never fully closed); BER climbs from about 1e-8 to a few 1e-2 | Hard push-in on the screen | lab: "EYE NEARLY CLOSED  BER RISING" |
| 6.8-8.5 | Wide framing bench, scope and Manager: Manager walks in, stares at the scope, face turns red, eyebrows steepen, mouth opens, steam puffs from his ears | Medium-wide at about 6 m, slow push | lab: "MANAGER READS THE SCOPE" |
| 8.5-9.2 | Manager turns on Gary and raises the clay shotgun | Push-in on barrel | none |
| 9.2-10.0 | BANG (smoke ring); first hole in Gary; he topples | Wide | "BANG" |

### S2 Retimers: inside the rack, row after row (10.0-20.0)

Narration (21 words): "Signal fading? Gary puts a chip on every connector. Now it takes forever to arrive and cranks up the electricity bill."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-0.8 | Flash cut: bench scope shows the closing eye with its BER; Gary watches | Quick cut-in | lab: "EYE CLOSING" |
| 0.8-1.6 | Camera fully inside the rack (side walls and backplane only, no front frame), at the first tray: a row of 4 backplane connectors; the first row of NARROWCOM retimers slides in behind them and fades in with a glow flash; a NARROWCOM sign with the parody valley mark is on the backplane | Inside the rack at tray level, slight push | lab: "RETIMER ROW ADDED BEHIND EACH CONNECTOR" |
| 1.6-5.4 | The camera rises vertically through 8 trays; each tray's chips slide in (move into place, fade in, glow flash settling to orange), slower and chip by chip; rows increase with height: 1 row on the lower trays, 2 on the middle, a 3rd row appears as the camera nears the top trays | Rising inside the rack, looking at the connector rows | lab: "TRAY AFTER TRAY: MORE ROWS" |
| 5.4-6.4 | Camera exits the top of the rack and cranes down into the room | High crane, descending | lab: "OUT OF THE RACK" |
| 6.4-8.6 | Gary stares at the bench scope: the eye recovers and its BER falls toward 1e-9 and below; Manager stares at a big whiteboard behind them where two curves draw themselves (no axis units or tick labels); the labels ENERGY / BIT and LATENCY ride next to the newest point of each curve, moving right with it | Wide with the whiteboard behind | lab: "EYE RECOVERS  BER FALLS" ; card: sources, curves have no units |
| 8.6-9.2 | Manager turns from the whiteboard and raises the shotgun | Push-in toward the barrel from the side | none |
| 9.2-10.0 | BANG; second hole in Gary | Wide | "BANG" |

### S3 NPO: closer, and everyone has a different idea (20.0-30.0)

Narration (23 words): "Gary moves the optics closer to the logic. Every vendor wants something different. Now he waits three more months and pays his bonus."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-1.8 | Conference table; the optical module slides next to the ASIC; four vendors in parody-logo shirts (MOLEXX, NUBISS, TERAHOP, AYARR) lean in | Snap push-in | lab: "NPO" ; card: Cheng 2025 |
| 1.8-3.0 | Laser fight: in-package engine vs external laser box; the vendors begin shouting (mouths flap, "@#$%!" balloons) and shoving | Whip pan A to B, orbit | lab: "LASER: IN PACKAGE OR EXTERNAL?" |
| 3.0-4.2 | Real different packages stand up with undersides to camera: BGA (ball array), LGA (land pads), fine-pitch BGA; the shouting and shoving continue | Close, slow push | lab: "BGA vs LGA vs PITCH vs SIZE" |
| 4.2-5.2 | Dies of mismatched sizes; vendors keep fighting | Quick cut-ins | lab: "DIE SIZE" |
| 5.2-7.0 | Dies side by side, then into a wobbling tower; calendar pages fly ("+3 MONTHS"); price tag inflates; the brawl escalates to punches | Dolly back and up | lab: "2D vs 3D  LEAD TIME  COST" |
| 7.0-8.3 | Gary hands a vendor the "YEAR-END BONUS" envelope; the Manager walks in and watches | Orbit snap | lab: "THE VENDORS LOSE IT" |
| 8.3-8.9 | Two vendors (MOLEXX and NUBISS) pull shotguns and shoot the other two (TERAHOP and AYARR), who fall with holes; Gary pulls out his own shotgun | Wide, tight on all seven people | "BANG BANG" |
| 8.9-10.0 | The Manager shoots NUBISS while Gary shoots himself in the head, both at once; the MOLEXX vendor is the only one left standing | Wide | "BANG" |

### S4 CPO: keep it scorching hot (30.0-40.0)

Narration (21 words): "Gary glues the optics onto the chip. The chip's heat swings wildly, but optics need it steady. So: scorching hot, always."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-1.4 | Top-down on the mocked-up DGX/HGX-style board. The XPU package follows the NVIDIA Rubin Ultra slide layout (`../references/rubin-ultra.png`): a gold lid frame; a wide central interposer band with 4 reticle-size GPU dies; 8 HBM stacks above and 8 below; around it regulators, MLCC clusters, inductors, transformers, coils, connectors and bolts. 8 + 8 optical engines (OEs) slide in onto the left and right edges | Locked top-down | lab: "NPO -> CPO (8 + 8 OEs)" |
| 1.4-2.6 | Dolly-zoom onto the XPU: power and temperature bars lurch; smoke puffs | Dolly-in with FOV compression, shake | lab: "XPU POWER  TEMPERATURE" |
| 2.6-3.45 | The camera keeps zooming, continuously, from the XPU shot onto one OE on the right edge (no cut) | Continuous zoom, lens 55 to 85 | lab: "ZOOM ON AN OE" |
| 3.1-3.8 | Quick fade to black and back in, landing on the zoomed-in PIC | Fade quad | none |
| 3.5-4.6 | PIC close-up: bus waveguides with 24 microring modulators on a substrate made of 28 slabs; a heat wave travels left to right across the rings and across the substrate (global heat, with the rings riding the same wave) | Slow oblique orbit over the chip | lab: "RING RESONANCES: HEAT WAVE" ; label: "HEAT WAVE ->" |
| 4.3-4.9 | Quick fade out of the PIC and back in on the board | Fade quad | none |
| 4.6-5.4 | Pull back to the board: every OE changes color cyan, orange, red and glows (staggered) | High pull-back | lab: "HEATERS ON" |
| 5.4-8.2 | One fried egg per OE (16 eggs, scaled to the engine): drop, white from glassy translucent to opaque, edges brown, bubbles, yolk sets, steam; camera runs along the left column of OEs, then the right | Low orbit along each OE column | big: "BREAKFAST" |
| 8.2-8.9 | Cut to human scale: Gary holds a plate of eggs; Manager aims at him | Wide | none |
| 8.9-10.0 | BANG; Gary tosses the eggs up as he is hit: they fly out of frame and come back down onto the Manager's head; Gary topples; fourth hole | Wide, tilted up to follow the eggs | "BANG" |

### S5 CPO yield: back through the factory to wafer test (40.0-50.0)

Narration (22 words): "Gary tests every optical chip at the factory. The rings all come out slightly different. Only one out of ten rings works."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-0.3 | The optical die on its engine | Locked | none |
| 0.3-3.3 | The die is yanked backwards through the line in reverse order: REFLOW ON SUBSTRATE, FAU ATTACH, EIC/PIC BONDING, DICING + TAPE (merged); motion-blur streaks fly past | Cut-per-station tracking shots with speed ramps | lab: "BACK THROUGH THE LINE" |
| 3.3-4.8 | Earliest step: wafer-level test on a CM300-style station. The probe head is fixed; wafers fly in one after another, the stage steps each wafer under the probe (die map lights up as dies are probed), and the wafer flies out; three wafers in 1.5 s | Over the station | lab: "WAFER-LEVEL TEST" ; label: "WAFERS IN / OUT, STAGE STEPS" |
| 4.8-7.0 | Screen with Lorentzian ring-transmission dips: new traces appear here and there (newest flashes white); a narrow green spec window; running pass/total count; 1 in 10 inside the window by construction | Close on the screen | lab: "RING RESONANCES vs SPEC" |
| 7.0-8.4 | Human scale: Manager holds the printout "1 / 10" next to Gary | Wide | big: "1 IN 10" |
| 8.4-10.0 | Manager raises the shotgun; BANG at 9.1; fifth hole in Gary | Wide | "BANG" |

### S6 Fiber: the tray (50.0-60.0)

Narration (20 words): "Last job: Gary pulls out the tray and ties up the fibers. Thousands of them. Manager wants it done yesterday."

| t | Action | Camera | Caption / label / card |
|---|---|---|---|
| 0.0-1.5 | A single glowing fiber flows serenely; end-face macro: a connector jams | Smooth dolly, then snap push-in | lab: "FIBER" ; card: SMF-28 loss and connector note |
| 1.5-3.2 | A tray is pulled out of the rack; fibers spill like noodles | Whip pan to the rack, tray slides toward camera | FX: fibers spill |
| 3.2-7.2 | The pile grows; Gary kneels and ties bundles; fibers squirm; a fiber threads through one of his shrunken, unhealed bullet holes | Handheld close, push-in, wide | footnote: "9,072 fibers per rack: hypothetical NVL576-style (own estimate)" |
| 7.2-8.0 | Manager walks in with a bundle of active optical cables (AOCs) as a whip and winds up | Push-in on the Manager | FX: AOC bundle whip |
| 8.0-8.7 | CRACK: the AOC bundle whips Gary | Close on Gary | big: "CRACK" |
| 8.7-10.0 | The NVYDIA-shirt customer walks in and slaps the Manager; the OPENAY / ANTHROPY-shirt customer hugs the NVYDIA customer's leg | Wide with all four | big: "SLAP" ; lab: "THE CUSTOMER" |

### S7 Disclaimer (60.0-62.0)

Ad-style closing card on pure black: large white disclaimer text, then a small-font list of the actual sources used (IEEE 802.3bj/ck/dj task-force documents; SemiAnalysis NVL72 piece; Arista and Ciena OFC 2026 slides and Broadcom 2023 slides; Cheng et al. Opt. Express 2025; Broadcom TH5 Bailly CPO deck and NVIDIA Developer Blog on CPO; Corning SMF-28 spec; own fiber-count estimate; package layout after NVIDIA's public Rubin Ultra slide; probe station after the FormFactor CM300 photo), plus a line stating that the eye, BER, ring-spectrum, heat-wave and decay animations are illustrative models. Ultrafast narration (about 9 words/s): "Gary is fictional. Results not typical. Figures vary by standard, vendor and mood. Void where copper is cheaper."

## 5. Narration script v1 (plain English)

Rules:
- Narrator describes what Gary wants and what happens to him; no jargon. Technical terms (retimer, NPO, CPO, latency, energy/bit, MRM, yield) appear only as short labels/after effects.
- Numbers spoken in VO are jokes ("three more months", "one out of ten"); sourced numbers appear only on cards.
- Word counts include everything the narrator says in the scene.

```text
S1 (22 words)
Gary wants faster data over copper. But faster means a shorter wire. So Gary tries stretching it one more meter. Bad idea.

S2 (21 words)
Signal fading? Gary puts a chip on every connector. Now it takes forever to arrive and cranks up the electricity bill.

S3 (23 words)
Gary moves the optics closer to the logic. Every vendor wants something different. Now he waits three more months and pays his bonus.

S4 (21 words)
Gary glues the optics onto the chip. The chip's heat swings wildly, but optics need it steady. So: scorching hot, always.

S5 (22 words)
Gary tests every optical chip at the factory. The rings all come out slightly different. Only one out of ten rings works.

S6 (20 words)
Last job: Gary pulls out the tray and ties up the fibers. Thousands of them. Manager wants it done yesterday.
```

Total 129 words over 60 s plus the 2 s disclaimer (S7, below). Narration windows (approx): S1 0.2-7.6, S2 0.2-8.0, S3 0.2-8.2, S4 0.2-8.2, S5 0.2-8.2, S6 0.2-6.0 (then action only). To sync tightly: S1 segments "Gary wants faster data over copper" 0.2-2.2, "But faster means a shorter wire" 2.2-4.2, "So Gary tries stretching it one more meter" 4.2-6.6, "Bad idea" 6.8-7.6 (lands on the scope slam).

## 6. Production plan

Tool decision: Blender (installed: 4.2.3 at `/Applications/Blender.app`, headless Cycles/Metal run on this machine before). Not three.js: clay shading, volumetric glow, particles/smoke/steam, DoF, motion blur and frame-exact offline renders are native in Blender EEVEE; three.js would only be lighter for interactive widgets. Blender 5.2 LTS optional.

- Procedural textures (implemented in `../scripts/tex_gen.py`, numpy + zlib, no extra dependencies): eye sequences for the S1 and S2 scopes (random NRZ bits through a Gaussian-limited channel + noise + jitter, new draw every frame so the eye shakes; BER = Gaussian estimate from the same samples, printed with a small bitmap font), the S2 whiteboard curves, and the S5 ring spectrum (Lorentzian dips, 1 in 10 inside the spec window by construction). They are written to `../assets/generated_textures/` (regenerated on each build) and shown as Blender image-sequence emission textures. The old HTML eye lab was not ported line by line; the model is similar in spirit (ISI from a limited-bandwidth channel, noise, jitter) but Gaussian-filtered rather than one-pole.
- Per-shot render: EEVEE, 1080x1350, 30 fps, PNG sequence to a scratch folder; encode each scene to H.264 and delete PNGs only after the scene is accepted (disk: 17 GiB free as of 2026-10-02; PNG sizes unverified until a 10-frame test).
- Captions/labels: render in Blender/Pillow (ffmpeg here has no drawtext); white bold sans, dark shadow, lower third inside the safe square.
- Assembly: ffmpeg concat of scene mp4s, H.264 yuv420p, AAC later, `+faststart`, fixed 30 fps.
- Layout: per DevLog-000 Section 7; scenes `scenes/s01_copper.blend` ... `s06_fiber.blend`; shared components linked from `assets/components/`.

### 6.1 Assets by scene

| Scene | Build list (DIY in Blender) |
|---|---|
| S1 | NVL72-style backplane copper cartridge wall (generic, procedural cable bundles), two racks on rails, twinax cable, Pulse, Gary rig, leash/tether, scope, Manager, clay shotgun |
| S2 | Rack frame with 10 trays, backplane, rows of 4 connectors and 1-3 rows of NARROWCOM retimer chips per tray, scope with eye texture, whiteboard with graph texture, Manager, Gary |
| S3 | Conference table, ASIC package + optical module, in-package laser engine vs external laser box, BGA / LGA / fine-pitch BGA packages with ball and pad arrays, mismatched dies, tower of dies, calendar, price tag, envelope, 4 vendors with parody-logo shirts, Gary's gun |
| S4 | HGX-style board mock-up (PCB, substrate, interposer, XPU, HBM, regulators, MLCCs, inductors, transformers, coils, connectors, bolts), 8 + 8 optical engines with per-engine heating colors, power/temperature bars, smoke, PIC with bus waveguides and 24 rings with random color changes, 16 fried eggs, plate, 3 tossed eggs |
| S5 | Optical engine and die, line stations (reflow oven, FAU attach, EIC/PIC bonder, dicing saw + tape roll), CM300-style probe station (cabinet, plate, column, 4 positioners, chuck, wafer, hopping needles, die map), Lorentzian spectrum screen, printout |
| S6 | Fiber aisle, tray slide, connector end-face macro, procedural fiber pile, AOC bundle whip, NVYDIA customer, OPENAY/ANTHROPY customer |

### 6.2 References and licenses

- In `../references/`: `formfactor-CM300.png` (FormFactor/Cascade CM300xi-style probe station, used as the shape reference for the crude S5 model: black cabinet, silver top plate with circular opening, central microscope column, four positioners, front drawer with wafer chuck; the real product marks are not reproduced); `JE3LSo1Rp54.mp4` (style); `OAM Spec v1.0.zip` (OCP Accelerator Module Design Specification v1p0 PDF, OAM pin map/list xlsx, OAM_0p90_ME.zip mechanical, Molex Mirror Mezz.7z); `Open-Compute-Specification-HGX-Baseboard-Contribution R1 V0.1.pdf`; user-supplied Broadcom chip photo (material reference only, in chat). OAM/HGX are relevant to the S4 GPU module look; read the OCP license terms before redistributing derived geometry.
- NVL72 rack: the NVIDIA DSX blueprint repo (https://github.com/NVIDIA-Omniverse-blueprints/omniverse-dsx-blueprint-for-ai-factories) contains no rack geometry; scene data (SimReady USD with GB200 and GB300 NVL72 designs) is a separate DSX Content Pack on NGC (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/dsx_dataset). The repo license is the NVIDIA Software License Agreement plus Omniverse product-specific terms (not an open license). Plan: reference only until the content-pack terms are read. The S1 hero shot is the backplane copper cartridge wall, built generically; public NVIDIA/press images of the NVL72 spine serve as visual reference (to download, Section 7).

## 7. Verification queue and downloads

- [ ] Re-read slides for S2: OFC2026 IMG_4150 (Ciena 25 W / 10 W / 2.5 W), IMG_4060 (Arista passive Cu 0 pJ/b); Broadcom 14 W/800G vs 5.5 W CPO from the Bailly deck.
- [ ] Read IEEE 802.3bj/ck/dj primary documents directly for the S1 reach figures (current values partly snippet-based; dj is an objective).
- [ ] S1 NVL72 spine card: read the SemiAnalysis piece (5,000 cables, 2 miles) and one NVIDIA primary source on the NVL72 backplane cable cartridges.
- [ ] S4: confirm the thermal-stability framing (resonance vs temperature, heater power) against a primary source or OFC2026 microring papers (Th2A.14 90 GHz Si MRM).
- [ ] Downloads (manual): public images of the NVL72 backplane cable cartridges; OSFP MSA and QSFP-DD spec PDFs (dimensions); egg-frying reference footage (for the animation timing; own footage is fine).
- [ ] Fresh-context audit of the final script; record corrections in Section 10.

## 8. Open questions (defaults proceed if unanswered)

Rev 4 open items:
- Resolved in rev 5: 16 OEs total; TERAHOP; parody logos at my discretion; whiteboard without axis units.
- Open: (1) the S3 shooting order (MOLEXX and NUBISS shoot TERAHOP and AYARR; the Manager shoots NUBISS) is my pick of which two shoot; say if you want different vendors. (2) The latency curve has no sourced numbers by design; the energy curve is drawn without ticks, so its 0 to about 15.6 pJ/b endpoints now appear only on the S2 card.

Earlier items:

1. (resolved rev 3: the Manager whips Gary)
2. S1 and S4 shot pattern: Gary shot in all scenes S1-S5, with the NVL72 copper-cartridge shot in S1 and the zoomed-in CPO shot in S4 as camera shots, per your reply (my reading: "shot" in your reply = gun shot, plus the two camera shots you named). Confirm. [as drafted]
3. Narrator uses "optics", "logic" and "rings" (your own words) in VO; everything else technical is label-only. OK? [yes]

## 9. TODO

- [x] v0.1 built from Wentao's timestamped feedback (see DevLog-002)
- [x] v0.2 built from the v0.1 feedback (see DevLog-002)
- [ ] Wentao reviews v0.2
- [ ] Verification queue (Section 7)
- [ ] Lock script v2 and caption/label/card text (words, timing, source footnotes)
- [ ] Project skeleton (README, `config/`, `assets/`, `scenes/`, `scripts/`) after approval
- [ ] 10-frame EEVEE test at 1080x1350 on this Mac to size render time and PNG disk use
- [x] Procedural eye/graph/spectrum textures (tex_gen.py)
- [ ] Build order: Gary/Manager/shotgun/hole -> S2 tray + NARROWCOM chip -> S1 -> S3 -> S4 (egg shader early, it is the riskiest look) -> S5 -> S6
- [ ] Sound and speech pass (next phase): VO recording plan (own voice vs TTS), SFX hit list on the frame-accurate beats above, music source

## 10. Audit corrections

(none yet)

## 11. Progress log

- 2026-10-02: Rev 1 proposal written from four research reports.
- 2026-10-02: Wentao approved chapters and gave scene content; references added. Reference Short measured; NVIDIA DSX repo checked. Rev 2 storyboard written.
- 2026-10-02: Wentao's second review: shot in every scene except the last, fried-egg spec, thermal runaway cut, NARROWCOM parody logo, 4:5 confirmed, narration rewritten in plain English. Rev 3 written (six scenes, 60 s).
- 2026-10-02: Wentao's v0 review by timestamp (14 notes); Section 4 rewritten as rev 4 and the crude film rebuilt as v0.1 (DevLog-002).
- 2026-10-02: Wentao's v0.1 review by timestamp (answers + 8 notes + 'add more details'); Section 4 rewritten as rev 5 and the crude film rebuilt as v0.2 (DevLog-002).
