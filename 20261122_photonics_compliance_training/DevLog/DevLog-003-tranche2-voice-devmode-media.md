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

## 6. Wave: immersion and chat UI (2026-11-22, after commit ed0dd80)

His feedback (verbatim):

```text
remove "rendered in blender..." etc labels. They can be centralized to a final disclaimer, no need to break the immersion
```

```text
the bystander should be given an actual name instead of calling it a bystander. The chat window display/UI should be better, e.g., using a proper iphone frame, and render a few kinds of msg platform style throughout the courses, such as imessage, signal, slack etc.
```

```text
also remove banners like "SCENARIO | HYPOTHETICAL. ANY RESEMBLANCE TO YOUR LAB IS INTENTIONAL.", all these should be pretty obvious, and lump it into the central final disclaimer.
```

| # | Decision | Why |
|---|---|---|
| D18 | No credit text on content screens (photos included); all media credits plus the dramatization disclaimer live in one generated final step "Disclaimer and Media Credits" after the certificate; dev overlay still shows path + credit | Immersion; CC BY / BY-SA attribution satisfied by a credits list in the work |
| D19 | Animatic "BLENDER RENDER PENDING" label and shot notes dev-only; "rendered" / "drawn for this course" wording removed from captions and narration | Same |
| D20 | Bystander becomes a named recurring coworker in both courses | His request |
| D21 | Chat `skin` field: phone imessage / signal / whatsapp in a realistic iPhone frame; thread slack / teams; call facetime / meeting; email mail. Real platform visual language, no real logos or app names; rounded corners allowed on device and bubbles only | His request; sharp-corner rule kept for LMS chrome |
| D22 | In-content "scenario / hypothetical / fictional / dramatization" banners and lines removed; wording moved into the final disclaimer | His request |

```text
if a page has multiple evolutions for the narrator, e.g., trainee read the scenario, and click to see the results, then implement it that way. Maximize interactivity as possible (which is what this kind of training course do) to keep the trainee engaged. Add additional play button or interaction UI as needed.
```

| D23 | Engine: `stages` on any step (click-to-reveal stages, each with narration), chat `tapToAdvance`, new `reflect` (ungraded response then reveal) and `sort` (drag into buckets, click fallback, forgiving spoken whys), video Replay. Content lanes restructure multi-evolution pages into stages and add sort / reflect exercises, mostly reusing existing narration | His interactivity request |

```text
for various message apps UI, the various characters should also have proper corresponding profile pictures.
```

| D24 | `media/cast.js` (LMS_CAST: name -> avatar, aliases) used by every chat skin; avatars mix code-drawn illustrated portraits, non-face photo avatars (pets, landscapes, lab gear; PD/CC), code-drawn group emblems; Gary as an unidentifiable bunny-suit crop; credits in `credits_avatars.js` and the final credits list | His request; D12/D13 still hold (no image generation, no real identifiable person in the cast) |

```text
can you update the company name to Five Sigma Photonics. Should be rare enough and fits my brand of outside five sigma
```

| D25 | Fictional employer renamed "Intracavity Photonics, Inc." -> "Five Sigma Photonics, Inc." in courses (incl. narration), engine, dashboard, chat skins, cast, tools; company emblem redrawn as a five-sigma mark; supersedes D8's company name. The physics category "Intracavity (Domestic) Feedback" is unchanged | His request |
- 2026-11-22 18:50: wave 6 done. FE 181 steps (about 95 min narration), FB 181 steps (about 79 min); every spoken line recorded (FE 1394/1394, FB 1381/1381; manifest 2753 clips, 116 MB; 154 catch-up clips; 34 superseded clips removed). Coordinator check: learner mode plays clips and gates Next, dev mode unlocks; no "Intracavity Photonics", emoji, absolute paths in shipped files; real-course chat skins with avatars inspected (iMessage, Slack). Uncommitted since ed0dd80.

```text
some of gary's actions are too extreme, make them less obviously wrong, like "Gary: I can load the next lot correctly and we never talk about lot 7." and the manager's "we will talk about lot 7". Should be more ambigious, such as gary saying delete lot 7 data instead of mark it as mistake with failure analysis. Also the layout should be more closer to an actual email with sender and title and date time etc.
```

| D26 | Tone pass in both courses: Gary's and the manager's lines and the wrong options become plausible rationalizations (exclude as non-representative, clean up data, follow up next lot, note verbally); wrongness = what is missing (failure analysis, log, root cause, marking); retaliation subtle | His request |
| D27 | Mail skin becomes a real email client view: subject title, per-message From (name + address), To, Cc, ISO date-time, attachments, signature, stacked replies, message list; optional per-message `to`, `cc`, `date`, `subject`, `attachments`; addresses at the fictional domain "fivesigma.photonics" via `email` in cast.js | His request |

```text
sometimes the narrative repeats the same phrase immediately and sounds like a glitch, e.g. "know your axis". Also should pause a bit when narrator answer her own question. Or popup interaction box for user to choose ok vs not ok.
```

| D28 | Back-to-back identical sentences removed (scan: 6 FE "Know your axes." doubles, 1 FB "Every interface reflects." double, 3 FE correctSay "That's right" prefixes); engine pauses about 1.2 s after spoken questions; new `ask` popup on any step (question spoken, OK / Not OK style buttons, short spoken response, then auto-reveal of the first stage); chats' "Is this OK?" self-answers become ask popups | His request |

```text
each module should have an opening deck/media like module 0 has. Either stock image or svg/code generated/blender rendered and relevant materials.
```

| D29 | Every module opens with a hero title slide (kicker "Module N", title, subtitle, one or two narration lines, full-bleed relevant image with push-in) before its objectives; images from existing photos/renders or new code-drawn title-card art with correct physics (credited "Drawn for this course" in the final credits). FB Module 0 title slide gains an image | His request |
- 2026-11-22 19:07: session restart interrupted the engine lane (mail layout, ask, question pause mostly landed) and the FE lane (tone/email/ask pass partial); both resumed.

```text
i think we should also add stock corporate music at the intro and outro slides to keep the user engaged. Maybe also add minor sound effects around the slide deck to bump awareness and focus of the tranee. Do not over-do the sound effect.
Pick a few meme commonly used ones, and i might already have a few ready.
```

| D30 | Sound layer: corporate music bed on course start / Module 0 opening slides and an outro on certificate + disclaimer, ducked under narration; subtle SFX on correct / wrong / reveal / incoming message / popup / sort place / module complete / certificate / perfect score, rate-limited; meme SFX (boom, sad trombone) only via explicit per-step `sfx`, sparingly; learner toggle. Sources: public-domain / CC0 music or code-synthesized; functional SFX synthesized. Meme SFX: proposal only, he supplies the files (his correction, verbatim: "i did not ask you to search. Just propose."); manifest keys stay null until then | His request |
- 2026-11-22 20:10: waves 7-9 done (tone pass, real email layout, ask popups and question pauses, module openers, sound layer). FE 188 steps (about 95 min), FB 188 steps (about 79 min); all spoken lines recorded (FE 1415, FB 1457; 2848 clips, 118 MB; 288 superseded clips removed). Music: Kevin MacLeod "Life of Riley" (intro loop) and "Cheery Monday" (outro), CC BY 4.0, credited on the disclaimer page; 8 synthesized functional SFX; meme SFX slots null pending his files. Coordinator check: clips and music load (HTTP 200), learner Next gated, dev unlocks, no old company name, emoji or local paths in shipped files. Uncommitted since ed0dd80.
