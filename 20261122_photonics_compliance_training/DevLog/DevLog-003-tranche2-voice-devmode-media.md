# DevLog-003: Tranche 2: recorded voice, dev mode flag, real media, incident scenarios

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Delivered for review 2026-11-22; uncommitted |
| Depends on | DevLog-002 (prototype), `course/COURSE_SPEC.md` (section "Tranche 2 additions" is the contract for this tranche) |

## 1. Instructions (verbatim, 2026-11-22)

```text
1. light is good.
2. sounds good.
3. swap the voice next and make it more natural.

Add a flag for dev mode so i could quickly inspect. And the dev mode should be off for actual deployed page.

Also, time to start geenrating/rendering various assets. Real training courses have real world media, some from actual scenarios, some dummy/generic workspace pictures, items, images etc. Each module should have at least two or three with full screen images or people or tools/equipments involved, or actual lab and fab scenarios and scenes. Photorealistic as possible. Various UI and interactive elements should also be styled in two or three different ways with real media, such as cards to choose, points to expand placed on actual scene etc. This is the focus of next tranche of work, and go ahead start implementing them. Notify me noce the above as well as media are generated/drawn/rendered, and incorporated into the training courses.
```

```text
come up with actual scenarios e.g. chips being handled with 90 deg off or 180 deg off wrong orientation, and device performance got damaged, and gary felt shamed. Add typical training course bs like question on what the tester would do in such scenario, with options to hide the mistake etc., or as a bystander to point or not point out the correction etc. Another example could be pyroelectric effect frying electrical circuits etc.
```

```text
for the ethical questions, take into account actual workflow and environment and tools, e.g. could also blame simulation tools, cleanroom equipments wafer vacuum chuck rotation / spin coater chuck rotation messing up orientation etc.
```

```text
also come up with more complex scenario, e..g the sexual harassment training has various classification of where the aggressor is from, internal, external, domestic, quid pro quo etc., for photics/photonics some of the mistakes could be with tools, with the equipments, with vendors and suppliers, and with internal (same lab/group) and external (across universities etc.) collaborators etc. Make up similar category names and build module to go over these scenarios etc.
```

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D9 | Light theme stays default; 25-30 min narration per course stays | His answers 1 and 2 |
| D10 | Dev mode = `?dev=1`, honored only on localhost / 127.0.0.1 / file://; `?review=1` dropped | "off for actual deployed page" without a build step |
| D11 | Voice: Kokoro-82M (kokoro-onnx, local) per-sentence clips keyed by a text hash, browser speech only as fallback; default voice picked from A/B samples (samples kept in `research/voice_samples/`) | Same local stack as the pigeon and interconnect films |
| D12 | Media = real photos (public domain / CC with attribution, mostly Wikimedia Commons and US government sources) + Blender Cycles product renders. No image-generation models | His rule: no image generation for technical visuals |
| D13 | No identifiable real person is cast as Gary or anyone at fault; cast appears in bunny suits, from behind, gloved hands | Parody of misconduct must not attach to real people. Confirmed by him 2026-11-22: "gary or bunnysuits sounds good." |
| D14 | New step types `photo`, `hotspots` (3 styles), `cards` (grid / flip, choose / explore), `spot` (find the hazard), image on `slide` / `check` / video keyframes | "styled in two or three different ways with real media" |
| D16 | Wrong options in ethics questions = realistic excuses from the actual workflow (simulation axis conventions, spin coater, aligner or probe-station chuck, prealigner, dicing, pick-and-place, wafer flat conventions; OSA resolution, averaging, patch cords for the feedback course), each explained as "the tool did what it was told; here is where the reference was lost" | His addendum |
| D17 | One new taxonomy module per course parodying harassment-training aggressor categories: equipment-, software-, supplier-originated, intra-group, cross-institutional, domestic, quid pro quo, hostile environment (feedback course adds internal/external feedback and retaliation); each category gets a policy definition, vignette, warning signs, a classification check and what-would-you-do / bystander checks | His taxonomy request |
| D15 | Incident scenarios with "What would you do?" and bystander questions (hide it, blame the crystal, speak up), Gary's shame handled in HR voice | His scenario message |

## 3. Lanes

| Lane | Owns | Phase |
|---|---|---|
| Engine | `engine/*`, `player.html`, `index.html`, `tools/validate.js` | 1 |
| Voice | `tools/tts/*`, `engine/speech_hash.js`, `audio/*`, `research/voice_samples/` | 1 pilot, 3 full run |
| Photos FE / FB | `media/fe/`, `media/fb/`, `media/shared/`, `media/credits_*.js`, `research/media_plan_*.md` | 1 |
| Blender | `media/render/`, `media/credits_render.js`, `tools/blender/` | 1 |
| Content FE / FB | `courses/ferroelectric.js`, `courses/feedback.js` | 1 scenarios, 2 media insertion |

## 4. TODO

- [x] Engine: dev mode, clip playback, new step types, credits, validator
- [x] Voice: samples, pronunciation check, generator, pilot, full run (2633 clips, 231 min, 113 MB, af_heart; Whisper spot check WER 3.3%, 0 missing clips in dev walks)
- [x] Photos sourced and placed (2-3 full-screen per module): 37 FE+shared, 26 FB, all PD/CC with credits
- [x] Blender renders: 15 (plus 3 card crops), two critique passes (`research/render_critique_v1.md`, engineering-audit table in `research/media_plan_render.md`)
- [x] Incident scenarios in both courses + one taxonomy module each
- [x] Media inserted; headless walk light/dark/phone 0 errors; fresh-context audit `research/audit_courses_v1.md` applied
- [x] Notify him

## 5. Progress log

- 2026-11-22: contract appended to `course/COURSE_SPEC.md`; lanes started.
- 2026-11-22: fresh-context content audit `research/audit_courses_v1.md` (2 must-fix, 11 should-fix, 17 nits) dispatched to content lanes. His correction to its thin-film item (verbatim): "TE and TM could still be nearly degenerate in thin film depending on waveguide profile and orientation. They might actually cross each other during a bend." -> claim made conditional on profile and orientation, not "thin film is always birefringent".
- 2026-11-22: his render feedback (verbatim): "some of the rendering is bs, e.g. .../butterfly_isolator_fc_bench.jpg has the mount / holder mounted not in actual breadboard holes, but still has a bolt thru its four mounting holes. This is nonsense. Review and fix similar nonsense." -> benches rebuilt on the 25 mm grid; fresh-eyes critique `research/render_critique_v1.md`; Blender lane fixing all must-fixes. Engine: shot notes dev-only; credit source links only for web URLs; spot tags numbered; phone callouts show only the open label; feedback panels join spelled acronyms.
- 2026-11-22: his FC connector feedback (verbatim): "the rendered FC connectors .../fc_apc_vs_pc_ferrules.jpg are also wrong metal housing. Find actual images and rawings to compare. It should have a few layers, inner one has a notch gap instead of a protrusion, And outer one can rotate and screw onto th receptacle, which also has grooves outside for better friction/grip." -> rebuilt from Commons photos and a vendor exploded drawing: ferrule, plug body with key tab (references show the key as a tab on the plug and the keyway on the adapter; the notch gap on the plug is the split retaining ring), separate knurled coupling nut, crimp sleeve, short tapered boot. His wafer note (verbatim): "the wafer rendering also depends on if it is single sdie polished or double side polished (DSP)" -> carrier wafer DSP (transparent), hot-plate wafer SSP (milky, lapped back); stated in alt text.
- 2026-11-22: final sizes: courses 174 (FE, about 94 min narration) and 176 (FB, about 77 min) steps; media 23 MB; Kokoro clips in `course/audio/`.
- 2026-11-22: final check (coordinator): learner mode via http://127.0.0.1:47293 plays recorded clips (HTTP 200), Next locked; dev mode badge and unlocked Next; dev gate false on outside5sigma.com and github.io, true on 127.0.0.1 and file://; validator OK both; no emoji, absolute paths or private terms in shipped files. 8 orphan clips (superseded text) removed.
- Open: course length (FE about 94 min, FB about 77 min narration) vs his 25-30 min answer; listen check for "T E light" and "in-plane"; dicing-saw render not made; DSP wafer tint reads warm grey.
