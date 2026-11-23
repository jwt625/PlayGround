# DevLog-000: Photonics compliance-training parody - kickoff, topic and course proposal

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Proposal for review. Nothing built, no research verification pass yet |
| Scope | Meme corporate-compliance training course on optics/photonics pitfalls. Blender-rendered vignettes (no AI video generation), generated narration, an interactive course player |
| Sibling projects | `../20261002_interconnect_film/` (Gary, manager, NPC rigs, lab/office/interconnect asset library, Kokoro VO pipeline), `../20261009_pigeon_pluggable/` (DR8 research, fast-tier render pipeline, post captions, VO rev process, OSFP hero) |
| Authors | Claude, at Wentao Jiang's request |

Path convention: paths are relative to this folder. Public side project: no startup or day-job material.

## 1. Instructions (verbatim from Wentao, 2026-11-22)

```text
are you familiar with company training video style on topics such as sexual harassment and cybersecurity and workplace violence etc.? Im not asking about actual content, im asking about the style and visual presentation and interactivity of these kinds of training materials. Are you familiar with their style and theme and storytelling methods etc.
```

```text
You got a lot of them right, good job. note it also has the following
1. overly repetitive talk points and excessive examples / hypothetic scenarios (including videos of 3rd person camera recording as well as first person interaction such as a phone screen with messaging or video call etc.).
2. for the quiz and exams, it is very forgiving and has video and/or audio feedbacks, including voice saying "thats right" when user choose the correct option, and when wrong long duration to explain whats wrong about it, and give user chance to choose again with the wrong options showing or marked as wrong.
3. many of the elements and interactivities are enforced, e.g., video and audio pause when browser tab lose focus, audio and video play cannot be accelerated or skipped.
Now here is the project I want to do: a meme-like training course but for optics/photonics/optical interconnects, picking a handful of pitfalls around a given topic. This is a great project for the following reasons:
1. actual photonic experts are familiar with the topics and traps, and would get the joke and meme aspects immediately
2. out of the field audience could actually learn things.
3. the video and audio elements should be much more intuitive than usual blogs or textbook or papers.

Find and read relevant projects, esp. the OSFP pogeon video, gary the scale-up interconnects video, and we could make the video elements based on blender modeled 3d scene and recording instead of actual ai video gen, and use similar generated narrative.

create a new project folder, and propose a topic and a storytelling training course chapters, and ask me for any feedback or clarifications you need.

Is this project clear?
```

## 2. Genre style bible (from the chat discussion, his additions marked W)

| # | Trait | How the parody uses it |
|---|---|---|
| G1 | Slide-based course inside an LMS (learning management system) shell: progress bar, chapter menu with locks, "Next" button | The whole deliverable is a fake LMS player |
| G2 | Flat corporate palette, rounded sans, stock-office look; earnest host narrator | Host narrator over clean slides; Blender office/lab sets in a slightly too-clean "training video" lighting |
| G3 | Ensemble of archetypes: careless employee, target, bystander, well-meaning manager, HR / safety officer | Gary = careless employee; manager; a bystander NPC; a "Fiber Safety Officer" host |
| G4 | Gray-area scenarios that escalate; "Is this OK?" pauses; rewind and replay done right | Each pitfall plays wrong, freezes, then replays correctly |
| G5 | W: overly repetitive talking points, excessive examples and hypotheticals | One mantra repeated in every chapter; 3+ scenarios per pitfall where 1 would do |
| G6 | W: mixed camera modes: 3rd-person staged vignettes and 1st-person phone screen (messaging, video call) | Blender vignettes for 3rd person; HTML phone-screen chat and video-call scenes for 1st person |
| G7 | W: forgiving quizzes with audio/video feedback: "That's right!" on correct; long explanation on wrong, then retry with the wrong option struck out | Implemented literally; wrong-answer explanations deliberately long |
| G8 | W: enforcement: pause on tab blur, no seek, no playback speed, locked Next until media ends | Implemented literally in the player; a "Are you still there?" modal |
| G9 | Knowledge check pool, 80% pass, attestation checkbox, PDF certificate | Final assessment + certificate with a course code |
| G10 | Tonal whiplash: upbeat stock music over grim content | Synthesized "corporate ukulele" bed under laser-eye-damage content |

## 3. Topic candidates

| Option | Topic | Why it works | Risk |
|---|---|---|---|
| **A (recommended)** | **Fiber connector handling and laser safety** ("Inspect before you connect") | Every photonics and data-center person has done or seen every one of these mistakes; outsiders can learn it in minutes; very visual (end-face microscope views, green vs blue connectors, light leaking at a bend, invisible IR). Laser safety is already a real mandatory corporate training, so the parody frame is almost literal | Lab-hygiene topic, less "deep" than architecture topics |
| B | Fiber-to-chip coupling in the lab (from his blog `fiber2chip` and `how-to-hold-fibers`): polarization, angle, epoxy shift, index matching, grating-coupler bandwidth | Strong expert in-jokes | Narrow audience; harder for outsiders |
| C | Data-center link bring-up: SMF vs MMF mix-up, wrong reach class, receiver overload on a short loopback, breakout mapping, FEC (forward error correction) blame | Reuses pluggable and pigeon assets | Overlaps with the Gary and pigeon films |

Recommendation: A for course 1. B and C fit as later courses in the same player ("Course catalog" screen can show them as "Coming soon" locked tiles).

## 4. Proposed course (option A)

Working title: **"Photon Safety and Fiber Conduct in the Workplace - Annual Refresher (OPT-1310)"**. Fictional employer name TBD (Q4).

Mantra repeated in every chapter (G5): **"Inspect. Clean. Inspect. Connect."**

Recurring cast (reuse rigs from `../20261002_interconnect_film/assets/components/characters/`):
- Gary (gary_v2): the employee who does everything wrong, every chapter.
- Manager (manager_v2): approves, escalates, is in every video call.
- Bystander NPC (npc_v2): the "Speak up!" coworker.
- Host: the Fiber Safety Officer, on-camera presenter for chapter intros and outros (new rig, or npc_v2 variant with a lanyard and safety glasses).

Chapter titles parody compliance-course naming; content is real.

| Ch | Title (on screen) | Pitfall(s), real content | 3rd-person vignette (Blender) | 1st-person scene (HTML) | Interaction |
|---|---|---|---|---|---|
| 0 | Welcome and Course Navigation | How to use the course; "This course takes approximately 45 minutes"; enforcement rules stated | Host at a desk, wave | N/A | Locked Next, "Click the speaker icon to begin", tab-blur pause demo |
| 1 | Recognizing Contamination | Dirty end faces cause link loss and back reflection; dust caps are not clean; mating dirty to clean contaminates both; inspection zones on the end face | Gary unplugs, sets the connector face-down on the bench, re-plugs. Hard zoom into the ferrule: a dust-boulder landscape on the core | Chat: "link flapping again lol" / "did u inspect" / "it was new out of the bag" | Hotspot: "Click all 5 defects on this end face" (code-generated end-face image with zones) |
| 2 | Appropriate Cleaning Practices | Dry click-cleaner first, then wet-to-dry if needed, then re-inspect; never shirt, breath, or canned air on the face; IPA residue | Gary wipes the ferrule on his polo, then blows on it. Replay done right with a click cleaner | Video call: manager shares a "best practice" slide that shows the shirt method | Drag-and-drop: sort 8 cleaning actions into Acceptable / Unacceptable |
| 3 | Know Your Type: APC vs UPC | Green = APC (8 degree angled polish), blue = UPC; mating them leaves an angled air gap: high loss, strong reflection, can damage both faces; "it clicked" is not a test | Gary forces green into blue: slow-motion cross-section of the two ferrule tips meeting at an angle, light bouncing back | Chat: "it clicked so its fine" | "Is this OK?" for 6 connector pairs (photos-as-renders) |
| 4 | Respecting Boundaries: Minimum Bend Radius | Macrobend loss; cable ties overtightened; door or rack rail pinching; bend-insensitive fiber is not a license | Gary zip-ties a bundle tight and closes the rack door on it; red visual fault locator glow leaks out at every bend | Ticket thread: "Intermittent errors on port 17" reopened 4 times | Slider: bend a fiber on screen, live loss meter (code-driven) |
| 5 | Who Transmits, Who Receives: MPO Polarity and Gender | Pinned vs unpinned (male/female) ferrules; trunk types A/B/C; TX must land on RX; MPO-16 APC on 800G DR8 (from pigeon research) | Gary mates two pinned MPOs: pins hit pins. Then "flip it" spiral through every adapter in the rack | Video call: 9 people, nobody knows which polarity the trunk is | Trace-the-lane puzzle: drag fiber 1 to the right position across a trunk |
| 6 | Recognizing Invisible Threats: Laser Safety | 1310/1550 nm is invisible; never look into a fiber or port to check if it is on; use a power meter or an inspection scope with a filter; laser classes (IEC 60825-1) | Gary peers into a live MPO with a loupe "to see if it's on". Nothing visible. Cut to the host, very serious music; Gary's eye glows in an IR camera view | Phone: Gary texting "its not on, i checked" | "What should Gary do?" branching, 4 options, 3 wrong with long explanations |
| 7 | Final Assessment | 10 questions from a pool, 80% to pass, unlimited retries | N/A | N/A | Quiz, attestation checkbox, certificate (PDF) with course code |

Optional chapters if he wants a longer course: SMF vs MMF jackets (yellow vs aqua), receiver overload on short loopbacks without an attenuator. Both overlap with option C, so they are held back.

### 4.1 Chapter template (each chapter follows it, G4/G5)

1. Title card + learning objectives slide (3 bullets, read aloud in full).
2. Cold-open vignette (3rd person, Blender), 15-30 s: Gary does it wrong. Freeze frame, red "X" stamp.
3. Host: "Let's take a closer look." Concept slide with click-to-reveal tabs; mantra repeated.
4. Hypothetical scenarios, 2-3 more than needed: 1st-person phone chat or video call; each ends "Is this OK?".
5. Replay done right (same vignette, green check).
6. Knowledge check (G7): voice "That's right!" or a long wrong-answer explanation, retry with the wrong option struck out.
7. Key takeaways (the mantra again) + "Click Next to continue".

### 4.2 Production plan (draft)

| Element | Approach | Reuse |
|---|---|---|
| Course player | Static no-build HTML/JS LMS shell: chapter lock, progress bar, media gating, tab-blur pause, no seek/speed, quiz engine, certificate | baby-mode `frame(t)` / autoplay ideas from `lithium-niobate` |
| 3rd-person vignettes | Blender, fast tier first (pigeon lesson: under 5 min per capture), clips rendered per chapter | Gary, manager, NPC rigs and 132 motion actions; `lab_bench`, `lab_clutter`, `office_cubicle_set`, `conference_room`, `fiber_connectors`, `mpo_connector_and_adapter`, `fiber_cables_and_trays`, `osfp_module`, `datahall_environment`; `bslot.sh` (max 3 Blender processes) |
| Microscope / cross-section views | Code-generated (end-face zones, contamination, APC/UPC gap, bend loss) for accuracy and interactivity | N/A |
| 1st-person phone and video-call scenes | HTML/CSS mock UI, generic (no real app branding), recorded or played live in the player | N/A |
| Narration | Kokoro TTS via `make_vo.py`, narration JSON with `say`/`show`; host voice distinct from the films' `am_liam`; Whisper round-trip check | `../20261009_pigeon_pluggable/scripts/audio/` |
| Feedback voice | Short "That's right!" / "Not quite." clips + long explanation lines | Same pipeline |
| Music | Synthesized upbeat corporate bed (no licensed track issue, unlike the pigeon film) | `scripts/synth_audio.py` from `advanced-packaging-atlas` hybrid-bonding short |
| Trailer | 60 s screen capture of the player for social, 4:5 1080x1350 | `recipes.md` |

### 4.3 Facts to verify before scripting (no claim is used until sourced)

| Claim | Planned source |
|---|---|
| End-face inspection zones and pass/fail criteria | IEC 61300-3-35 (summaries from scope vendors if the standard is paywalled) |
| APC angle (8 degrees), APC green / UPC blue color code, typical return loss | Connector vendor datasheets, TIA/IEC connector standards |
| "Contamination is the leading cause of link failure" | Vendor studies (inspection-scope and test-equipment makers); label as vendor claim |
| Cleaning procedure (dry, wet-to-dry, re-inspect) | IEC 61300-3-35 / IEC TR 62627-01, vendor cleaning guides |
| Minimum bend radius rules of thumb; bend-insensitive fiber classes | ITU-T G.652 / G.657, cable datasheets |
| MPO pinned/unpinned, polarity methods and trunk types | TIA-568 series summaries, MPO vendor guides |
| 800G DR8 uses MPO-16 APC | `../20261009_pigeon_pluggable/research/hero_800g_osfp_dr8.md` (already sourced) |
| Laser classes and eye-safety statements for 1310/1550 nm transceivers | IEC 60825-1 / -2 summaries, transceiver datasheets (class label) |

## 5. Open questions for Wentao

See section 6 for status. Asked in chat 2026-11-22.

### 5.1 Feedback on the topic (verbatim, 2026-11-22)

```text
the topics are too common, pick some very niche and deeply technical ones, such as laser mode hop or polarization management on a PIC. It should not overlap with actual safety training such as fiber connector handling and laser safety, those actually exist. I want topics that there is no chance or broad audience for it to be actually worth making such a training course, but it is for the love of the game and for helping actual experts avoid pitfalls they might not have thought about. Maybe a dozen ish people around the world actually understand such topic etc.
```

Result: options A, B, C in section 3 rejected (too common; A overlaps real safety training). Section 4 course plan is superseded by `DevLog-001-niche-topic-shortlist.md`. Genre style bible (section 2) and production plan (4.2) still apply.

## 6. TODO

- [x] Read sibling projects (interconnect film DevLog-000/001, pigeon DevLog-000, narration rev 5, asset library)
- [x] Folder + this DevLog with proposal
- [ ] Wentao picks topic, format, length, tone (questions in chat)
- [ ] Research pass on section 4.3 (one subagent per claim group), cache sources under `references/`
- [ ] Chapter scripts + narration JSON v1 (DevLog-001)
- [ ] Course player skeleton with enforcement and quiz engine
- [ ] Crude Blender block-out of one chapter (pilot) for timing and look
- [ ] Fresh-context audit of technical claims

## 7. Progress log

- 2026-11-22: genre discussion in chat; folder created; sibling projects read; proposal (section 3-4) written.
