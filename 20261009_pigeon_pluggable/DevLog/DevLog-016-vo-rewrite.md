# DevLog-016: VO rewrite in the owner's writing style (narration rev 3, rev 4) - workstream W

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Rev 3 voiced by A (DevLog-014 section 11; coordinator picked the s4d / s6d alternatives). Rev 4 draft delivered (section 12, DevLog-018 V1/V2); A voices it |
| Item | F3 in `DevLog-015-interim-feedback-v2.md`; V1/V2 in `DevLog-018-feedback-wave-v3.md` |
| Inputs | DevLog-015 (owner feedback), DevLog-010 (P5-P7, section 9 v2 work order), DevLog-006 (storyboard rev 3), DevLog-002 (rev 2 narration, backronyms, disclaimer), DevLog-000 section 3 (parody names, say-spellings), DevLog-014 (voice, pacing, Whisper fixes), `scripts/audio/narration.json` (rev 2 measured timing), `research/*.md` (facts) |
| Style source | Owner's blog repo, read-only: `$BLOG/_posts/` (`$BLOG` = `../../jwt625.github.io` from this folder) |
| Outputs | this file; `scripts/audio/narration_rev3_draft.json`, `scripts/audio/narration_rev4_draft.json` (narration.json is NOT modified by W) |

## 1. Owner request (verbatim, 2026-10-10)

```text
the VO needs to be improved, have a subagent read and sample my writing styles from my blog and use similar style. Do not repeat same info like left eyes send right eyes receive got repeated twice.
```

## 2. Plan

1. Read the blog posts (tutorial, teardown, OFC, weekly OFS 2025-2026) and write a style guide with short quoted examples.
2. Rewrite each scene with the same story beats, one fact per scene and nothing said twice; keep the conspiracy-meme voice.
3. Make each scene fit its 10 s window at the rev 2 read speed (Kokoro `am_liam`, speed 1.10) and estimate line durations from the measured rev 2 takes.
4. Check every technical claim against `research/*.md`, and follow the naming rules: never "pluggable", parody names only, OSFP backronym once, DFB / TIA / DSP spoken as letters, "Mach Zehnder" spelled as before.
5. Deliver the draft JSON (same schema as narration.json) and this DevLog.

## 3. Style guide (sampled from the blog)

Posts read: `2026-08-20-silicon-photonics-and-co-packaged-optics.md` (tutorial), `2026-06-13-100G-200G-400G-800G.md` (teardown), `2026-04-04-OFC2026.md` (conference notes), weekly OFS #64 (2025-09-15), #88 (2026-03-02), #110 (2026-08-03), #118 (2026-09-27), plus grep passes over OFS #70-#118 for vendor and optics lines (#91 for the OFC Q&A quotes). Quotes are kept short.

| Trait | What he does | Short examples (post) |
|---|---|---|
| Plain-word definitions | Gives the jargon a plain meaning in a few words, often with "fancy name" or "fancy way of saying" | "circuits for light" (SiPh/CPO); "fancy name for periodic structure" (OFS-64); "a fancy way of saying we shove more wavelengths into the same fiber" (SiPh/CPO) |
| Physics first, in order | Walks the chain source, modulation, transmission, detection, one sentence each; says "real quick" | "Now we go thru each one of them real quick." (SiPh/CPO) |
| Function in one line | What a part does, in everyday terms | "continuous optical power, which carries no information"; "as an optical switch"; a photodetector "converts optical power into electrical current" (SiPh/CPO) |
| Analogies from software/everyday | Eye diagrams as dashboards, carriers as Wi-Fi | "extremely fast observability plots" (SiPh/CPO) |
| Blunt line, then the correction | Meme first, then the real reason | "copper bad bad, but copper is cheap. Sorry, it is that..." (SiPh/CPO recap) |
| One-word or very short answers | Deadpan punchlines, sometimes a whole section | "How to make your own" -> "Don't." (SiPh/CPO); "Industry best practice." (OFS-91, OFC Q&A) |
| Question hooks, quizzes | Asks the reader to look or think | "Can you find the isolator?" (teardown); "quiz: ..." (SiPh/CPO) |
| Self-aware asides | Admits limits, laughs at himself | "These are not my expertise, I just know these words." (SiPh/CPO); "lol", "ngl", "hmm" (OFS) |
| Riffs on known lines | Twists a famous phrase into photonics | "if you gaze long enough into a photodiode, the photodiode will gaze back into you" (OFS-64) |
| Vendor tone | Respects the engineering, mocks the hype and marketing; skeptical question instead of a rant | "commercial CPO? are you sure about that?" (OFS-118); "co-packaged obliterator (CPO)" (OFS-91); "imposters" for organic-substrate CPO (SiPh/CPO) |
| Caps for real enthusiasm | Rare, short | "FOUR BUCKS WOULD YOU BELIEVE IT!" (SiPh/CPO) |
| Numbers as products | Breaks big numbers into lanes x rate x fibers | "expand it back into lane rate, wavelengths, fibers, and directions" (SiPh/CPO) |
| Sentence length | Weekly posts: lowercase headline quips of 5-15 words plus terse bullets; tutorials: 10-25 word sentences with no padding | - |

Rules taken from this for the VO:
- One idea per line, 2-8 words, and at most one clause after a colon.
- Name the part, then say what it does in a few plain words ("light in, current out").
- Use deadpan, not shouting. Put the joke in the last two or three words.
- Point at what is on screen; do not narrate what the picture already shows.
- Use one blog signature per scene at most ("fancy name for", "real quick", the gaze riff). Do not stack them.
- Spoken lines cannot use "lol"/"ngl"; the casual register comes from rhythm and word choice instead.

## 4. Narration rev 3

Timing method: lines whose text is unchanged keep their measured rev 2 duration (`narration.json`). New lines are estimated with a linear syllable model fitted on the 25 measured rev 2 lines (S7 excluded): dur = 0.159 s x syllables + 0.038 s x pauses + 0.53 s, RMS error 0.17 s, max 0.56 s. Lines are laid out like `make_vo.py`: t0 = max(scene start + anchor, previous end + 0.22 s). These are target windows only; A's measured takes replace them.

### S1 Intro photos -> street -> flap -> dock (0-10)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s1a | 0.25-2.04 | 4 | Birds evolved from dinosaurs. | same | Separate intro: real bird photos with simple models of their matching form factors |
| s1b | 2.26-3.62 | 5 | Or so they told you. | same | Intro fades into the night street |
| s1c | 3.84-7.24 | 7 | Meet the O-S-F-P: Overhead Surveillance Feathered Platform. | Meet the OSFP: Overhead Surveillance Feathered Platform. | Hero rises past the tree, through the bird flap, onto the pad |
| s1d | 7.90-9.55 | 5 | Fancy name for a pigeon. | same | Hero docks; every head tracks Gary; settle on the hero |

Scene: 21 words, about 8.2 s of speech, about 0.45 s of air at the end.

### S2 Hop out, latch, sides, mating order (10-20)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s2a | 10.25-11.49 | 2 | Feathers? Heatsink. | same | Hero hops out forward onto Gary's hand; fins read |
| s2b | 11.80-13.62 | 7 | Beak: a latch that bites the cage. | same | Beak latch close-up |
| s2c | 14.60-15.77 | 3 | Right side transmits. | same | Right pads + right eye rim pulse red (14.6) |
| s2d | 16.30-17.47 | 3 | Left side receives. | same | Left pads + left eye rim pulse cyan (16.4) |
| s2e | 17.69-19.00 | 3 | Ground lands first. | same | Re-plug cutaway: GND claws, then power, then signal pads; beak bites the lip |

Scene: 18 words, about 6.7 s, about 1.0 s of air. **The only place TX/RX sides are said.**

### S3 Inside the TX eye (20-30)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s3a | 20.25-21.46 | 3 | Inside, real quick. | same | Head turn, push into the right eye, red fill |
| s3b | 22.00-24.99 | 7 | Two D-F-B lasers: steady light, zero information. | Two DFB lasers: steady light, zero information. | 2 CW DFB lasers, Si lenses, micro isolator (F2 label) |
| s3c | 25.21-27.33 | 5 | Eight Mach Zehnder modulators fix that. | Eight Mach-Zehnder modulators fix that. | Splitter tree -> 8 MZMs blinking fast (F7) |
| s3d | 27.55-29.39 | 6 | Four levels, two bits a symbol. | same | Prism, 2x4 lenslets, fast-blinking beams launch; zoom out of the eye |

Scene: 21 words, about 8.2 s, about 0.6 s of air. "Eight" is said once here; the eight lenses and eight beams are shown, not said.

### S4 Inside the RX eye, bird POV (30-40)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s4a | 30.25-31.26 | 1 | Incoming. | same | Orbit during the head turn; the sender bird across the aisle beams into the hero's other eye; cyan fill |
| s4b | 32.00-33.88 | 5 | Photodiodes: light in, current out. | same | Lenslets -> interleaver -> 8 PIN PDs flashing (32.6-34.6) |
| s4c | 34.20-36.80 | 7 | A T-I-A turns tiny current into voltage. | A TIA turns tiny current into voltage. | Octal TIA, spiral inductors, bonds (hold to 36.5) |
| s4d | 37.40-39.24 | 7 | Gaze at a bird, it gazes back. | same | Bird POV, battlefield-intel HUD (F4); lock on Gary at "gazes back" (about 38.5) |

Scene: 20 words, about 7.3 s, about 0.8 s of air. The eye is not named; the cyan fill and "Incoming" carry RX.

### S5 Exploded view and hatchery (40-50)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s5a | 41.30-43.91 | 8 | The brain: a D. S. P. from Narrow-com or Mer-vall. | The brain: a DSP from NARROWCOM or MERVALL. | Explode; DSP die stamp ticks NARROWCOM / MERVALL (about 42.6 / 43.4); the other parody call-outs (F5) are on screen only |
| s5b | 44.20-47.71 | 8 | A microcontroller snitches: temperature, laser current, light levels. | same | Push in on the MCU; I2C telemetry HUD, ending "SUBJECT: HEART RATE ELEVATED" (shown, not said) |
| s5c | 48.20-49.56 | 3 | Hatched by Out no light. | Hatched by OUTNOLIGHT. | Snap back, leg band close-up, hatchery flashback |

Scene: 19 words, about 7.5 s, about 0.45 s of air (same as rev 2; anchors unchanged).

### S6 Street: sweep, hop, satellites, cameras (50-60)

| id | t0-t1 (est) | Words | Say | Show | On screen |
|---|---|---|---|---|---|
| s6a | 50.25-51.68 | 4 | They follow you outside. | same | Gary exits the door (at or before 50.25); hero + two birds follow |
| s6b | 52.40-54.24 | 6 | Bird to bird. No fiber required. | same | Eye beams sweep the street and hop between perched birds (F6) |
| s6c | 54.80-56.84 | 5 | Satellite handoff. Again. Again. | same | Beam tracks one LEO satellite, hands off to the next one entering view; "Again" x2 lands on handoffs 2 and 3 |
| s6d | 57.30-59.26 | 5 | Every camera on the street. | same | CCTV links; heads turn to camera; hero blinks its TX eye after the line |

Scene: 19 words, about 7.3 s, about 0.75 s of air.

### S7 Disclaimer (60-62)

| id | t0-t1 | Words | Say | Show |
|---|---|---|---|---|
| s7a | 60.10-61.57 (measured, fast) | 6 | Parodee. Typical build. Not real birds. | Parody. Typical build. Not real birds. |

Unchanged (tuned in DevLog-014).

Totals: 124 words (rev 2: 127), about 45 s of speech in 60 s (rev 2: 43.9 s). Speech rate per scene 2.5-2.7 words/s of speech, against 2.5-3.5 in rev 2.

## 5. No-repeat audit

| Information | Said in | Elsewhere only shown |
|---|---|---|
| Right = transmit, left = receive | s2c, s2d (once) | S3 red fill / S4 cyan fill, "Incoming", HUD |
| Birds are OSFPs / surveillance | s1c (backronym once), s1d | heads tracking (S1), HUD (S4), telemetry gag (S5) |
| "Watching / sees you" | s4d only (gaze riff) | rev 2 said it twice (s1d, s4d); s1d is now the "fancy name" joke |
| Lane count eight | s3c only | 8 lenses, beams, PDs on screen; DR8 label |
| Lenses | not said | rev 2 said "eight lenses" in S3 and S4 |
| Mating order | s2e | cutaway contact glow order |
| Light -> current -> voltage | s4b, s4c | - |
| Telemetry fields | s5b | HUD |
| Free-space links | s6b | beams in S3/S4/S6 |

## 6. Fact check (every technical claim -> research)

| Line | Claim | Source |
|---|---|---|
| s1c | OSFP backronym (parody) | DevLog-002 backronyms |
| s2a | Feathers = heatsink (OSFP integrated finned heatsink) | `research/hero_800g_osfp_dr8.md` claim 7 (heatsink) |
| s2b | Beak = retention latch (stylized) | DevLog-002 R2 |
| s2c/d | TX and RX on opposite lateral halves (card edge and eyes) | hero research claim 7 (TX/RX on card edge, orientation); DevLog-002 R4 |
| s2e | GND mates first (then power, then signal) | hero research claim 7 (mating order, MSA 3.5 / 15.1) |
| s3b | 2 CW DFB lasers; CW light carries no information | hero research claim 1 + canonical internals; blog SiPh/CPO ("carries no information") |
| s3c | 8 MZMs encode the data | hero research claim 2 |
| s3d | PAM4: four levels, two bits per symbol | hero research claim 5 (8 x 106.25G PAM4 host lanes); blog SiPh/CPO (R_bit = R_symbol log2 M) |
| s4b | PIN photodiodes convert light to current | hero research claim 4; `research/rx_optics_refs.md` S1 (PIN array) |
| s4c | TIA converts the small current to a voltage | hero research claim 4 (octal TIA); blog SiPh/CPO |
| s5a | DSP brain; Broadcom / Marvell make PAM4 DSPs (as NARROWCOM / MERVALL) | `research/form_factors_and_companies.md` C; DevLog-000 section 3 |
| s5b | MCU reports temperature, laser bias current, TX / RX optical power (DDM) | hero research claim 6 |
| s5c | Innolight = module maker (OUTNOLIGHT) | form factors research C |
| s6b | No fiber: film device; disclaimer card says real DR8 uses 16 fibers via MPO | DevLog-002 R4, S7 card |
| s6c | Satellite handoff = the on-screen LEO train (generic) | DevLog-015 F6 (staging, not a claim) |
| alt s3d-a | 8 lanes x 100G = 800G | hero research (800G DR8, 8 lanes; host 8 x 106.25G) |

No numbers were added except "eight" (s3c) and those in the alternatives. "Pluggable" does not appear. Parody names only.

## 7. Alternatives for the three weakest beats

| Beat | Primary | Alt A | Alt B | Why weak |
|---|---|---|---|---|
| s3d (TX launch) | Four levels, two bits a symbol. | Eight lanes, a hundred gig each. (about 1.7 s; says "eight" a second time, but new information) | Not on and off. Four levels. (about 1.7 s) | Densest jargon in the film; may not land in 1.8 s |
| s4d (POV lock) | Gaze at a bird, it gazes back. | It sees you. (rev 2, measured 0.94 s, Whisper-clean, hud_lock already planned on it) | Target acquired. Hi, Gary. (about 1.7 s; fits the battlefield HUD) | "It" is a little ambiguous right after the TIA line |
| s6d (ending) | Every camera on the street. | Down to every camera on the street. (rev 2, measured 1.72 s) | Smile. The cameras are in on it. (about 2.2 s, ends about 59.5) | Flat ending before the blink |

Optional extra (only if the F2 label "MICRO ISOLATOR (FARADAY)" is on screen during s3b): s3a -> "Can you find the isolator?" (about 1.8 s; the owner's own teardown question). The research flags the isolator as optional (none documented for DR8; `research/tx_optics_refs.md` fact 6), so the line only asks and does not claim one is there.

Other signature lines considered and left out: "Copper bad bad." (there is no copper path on screen to point at); "Industry best practice." after the DSP vendors (S5 has no room; possible replacement for s5c if the hatchery shot is cut).

All alternatives are also in the JSON under `alts` (s3a, s3d, s4d, s6d).

## 8. Change log vs rev 2

| id | Rev 2 | Rev 3 | Reason |
|---|---|---|---|
| s1a-s1c | - | unchanged | Hook and backronym work; measured and Whisper-clean |
| s1d | It is watching you. | Fancy name for a pigeon. | Removes the duplicate of s4d; blog "fancy name for" |
| s2a | Feathers: heatsink. | Feathers? Heatsink. | Question-answer deadpan; same fact |
| s2b | - | unchanged | |
| s2c | Right foot and eye transmit. | Right side transmits. | Shorter; pads and rim show foot and eye |
| s2d | Left foot and eye receive. | Left side receives. | Same; anchor 6.0 -> 6.3 so it lands on the 16.4 cyan pulse |
| s2e | - | unchanged | |
| s3a | Right eye: transmit. | Inside, real quick. | Repeat removed (owner); blog "real quick" |
| s3b | Two D-F-B lasers glow deep in its skull. | Two D-F-B lasers: steady light, zero information. | Says what CW light does (blog explanation); "skull" moved out |
| s3c | Eight Mach Zehnder modulators. | Eight Mach Zehnder modulators fix that. | Adds what the MZMs do |
| s3d, s3e | Eight lenses. / Eight beams. | Four levels, two bits a symbol. | Lenses/beams shown, not said; adds PAM4 |
| s4a | Left eye: receive. | Incoming. | Repeat removed (owner) |
| s4b | Eight lenses, eight photodiodes. | Photodiodes: light in, current out. | Repeat of "eight lenses" removed; says what the part does |
| s4c | A T-I-A, a transimpedance amplifier, turns tiny currents into voltage. | A T-I-A turns tiny current into voltage. | Shorter (about 2.6 s vs 3.96 s); expansion stays on the S4 label |
| s4d | It sees you. | Gaze at a bird, it gazes back. | Owner's own photodiode-gaze riff; no repeat of s1d |
| s5a | - | unchanged | Stamp tick timing kept |
| s5b | A microcontroller reports temperature, laser current and light levels. | A microcontroller snitches: temperature, laser current, light levels. | Conspiracy verb; same fields; the heart-rate gag stays on screen |
| s5c | - | unchanged | |
| s6a | - | unchanged | |
| s6b | Eye to eye, bird to bird. | Bird to bird. No fiber required. | "Eye to eye" already shown; adds the no-fiber gag |
| s6c | Up to the satellites. | Satellite handoff. Again. Again. | Matches the new LEO tracking/handoff staging (F6) |
| s6d | Down to every camera on the street. | Every camera on the street. | Shorter; leaves air for the blink |
| s7a | - | unchanged | |

## 9. Notes for A (voice) and S (beats)

- Whisper-check the new words: "Heatsink" with "?" before it (keep `tail_fix`), "transmits", "receives", "Photodiodes", "T-I-A", "snitches", "handoff", "Again. Again." (Kokoro may read the repeat flatly; if so, try "Again, and again."), "Gaze ... gazes".
- If "Feathers? Heatsink." comes out with odd question intonation, fall back to the rev 2 text "Feathers: heatsink." (measured 1.03 s).
- Re-target events: `hud_lock` to the word "back" in s4d (about 38.6-39.0); satellite handoffs on "handoff" / "Again" / "Again" (about 55.0 / 56.0 / 56.5); hero TX blink after s6d (about 59.3-59.8); the door exit at or before s6a (critic #14).
- If a scene overflows, shorten s3b first ("Two D-F-B lasers. Steady light.") and s5b second ("A microcontroller snitches: temperature, current, light."). Do not time-stretch.

## 10. TODO

- [x] Read the blog posts and write the style guide
- [x] Rev 3 script per scene with word counts, estimated windows, beat mapping
- [x] No-repeat audit, fact check against research
- [x] Alternatives for the 3 weakest beats (plus the optional isolator line)
- [x] `scripts/audio/narration_rev3_draft.json` (narration.json untouched)
- [x] A: voice rev 3, Whisper check, measured timing into narration.json (DevLog-014 section 11)
- [x] Rev 4 draft per DevLog-018 V1/V2 (section 12)
- [ ] A: voice rev 4 (OUNOLIGHT, DIMENTUM, INCOHERENT, SEMTREK, MACOMBO say-spellings), Whisper check, timing
- [ ] S: re-align beats and events to A's rev 4 timing
- [ ] Owner: listening pass

## 11. Progress log

- 2026-10-10: DevLog-000/002/006/010/014/015, narration.json and research read; blog sampled (SiPh/CPO tutorial, 100G-800G teardown, OFC 2026, OFS #64/#88/#110/#118, grep over OFS #70-#118); style guide written; duration model fitted on the 25 measured rev 2 lines (RMS 0.17 s); rev 3 drafted, trimmed to 18-21 words per scene, audited for repeats and facts; draft JSON written.
- 2026-10-10: rev 4 (owner feedback DevLog-018 V1/V2): base = narration.json rev 3 as voiced (s4d = "Target acquired. Hi, Gary.", s6d = "Smile. The cameras are in on it.", say "Mach Zender"); S1/S2/S3/S5/S6 lines replaced; duration model refitted on 41 measured rev 2 + rev 3 takes (RMS 0.18 s); vendor roles checked against research; `narration_rev4_draft.json` written; section 12 added.

## 12. Narration rev 4 (2026-10-10, DevLog-018 V1/V2)

### 12.1 Owner input (verbatim excerpts from DevLog-018 section 1, VO items only)

```text
- remove "fancy name for a pigeon". Replace it with workhorse of modern data centers.
- "feathers heat sink" replace with the more technical names of the heat sink, like Integrated heat sink (IHS)
- dont use "real quick" etc. Directly start describe what is inside, e.g. DFB lasers and where they are from (source photonics, lumentum / coherent or their parody names)
- 0:44: no need to talk abt dumb ucontroller, talk about DFB, PD and TIA etc. (semtech, macom and their parody names). Dont zoom in and out and back in too much, just slow orbit the cam.
- 0:50: ... Use ounolight instead of outnolight, too weird.
- dont say they follow you outside, just say they observe you outside. ...
- your last line is bs / too plain. Come up something that is more grand, like IoT, distributed intelligence etc.
- gary should be unaware and goes abt his day instead of getting scared.
```

Base: `scripts/audio/narration.json` rev 3 as voiced by A (s4d = "Target acquired. Hi, Gary.", s6d = "Smile. The cameras are in on it.", say "Mach Zender"). Timing: unchanged lines keep A's measured rev 3 durations; new lines use the syllable model refitted on 41 measured rev 2 + rev 3 takes (dur = 0.161 s x syllables + 0.026 s x pauses + 0.52 s; RMS 0.18 s, max 0.57 s), laid out like `make_vo.py`.

### 12.2 Rev 4 script (target windows, film s)

| id | t0-t1 | Dur | Say | Show | On screen |
|---|---|---|---|---|---|
| s1a | 0.25-2.04 | 1.79 m | Birds evolved from dinosaurs. | same | Intro cards: bird photo, bird name, form-factor name |
| s1b | 2.26-3.62 | 1.36 m | Or so they told you. | same | Fade into the street |
| s1c | 3.84-7.24 | 3.40 m | Meet the O-S-F-P: Overhead Surveillance Feathered Platform. | Meet the OSFP: ... | Hero rises, flap, pad |
| **s1d** | 7.46-9.43 | 1.97 e | Workhorse of modern data centers. | same | Hero docks; heads track Gary; settle on the hero (anchor 7.9 -> 7.45) |
| **s2a** | 10.25-12.11 | 1.86 e | Feathers? Integrated heat sink. | same | Hero hops onto Gary's hand; fins read |
| s2b | 12.33-14.15 | 1.82 m | Beak: a latch that bites the cage. | same | Beak latch close-up (now about 0.5 s later than rev 3) |
| s2c | 14.60-15.98 | 1.38 m | Right side transmits. | same | Right pads + rim red |
| s2d | 16.30-17.62 | 1.32 m | Left side receives. | same | Left pads + rim cyan |
| s2e | 17.84-19.15 | 1.31 m | Ground lands first. | same | Re-plug cutaway |
| **s3a** | 20.25-22.40 | 2.15 e | Two D-F-B lasers, by Dim-en-tum. | Two DFB lasers, by DIMENTUM. | Push into the right eye; DFBs switch on as the camera arrives (T light-up) |
| **s3b** | 22.70-24.69 | 1.99 e | Steady light, zero information. | same | CW light through lenses, isolator, couplers, MMI tree, following the pan (anchor 2.0 -> 2.7) |
| s3c | 25.00-27.45 | 2.45 m | Eight Mach Zender modulators: fix that. | Eight Mach-Zehnder modulators fix that. | Gold electrical waves along the TW electrodes; modulated light out |
| s3d | 27.67-29.39 | 1.72 m | Four levels, two bits a symbol. | same | Lenslets, fast beams launch |
| s4a-s4d | as rev 3 | m | Incoming. / Photodiodes: light in, current out. / A TIA turns tiny current into voltage. / Target acquired. Hi, Gary. | | unchanged |
| s5a | 41.30-43.91 | 2.61 m | The brain: a D. S. P. from Narrow-com or Mer-vall. | The brain: a DSP from NARROWCOM or MERVALL. | Exploded view, slow orbit; DSP stamp |
| **s5b** | 44.20-47.64 | 3.44 e | Photodiodes by In-co-here-ent, T-I-As by Sem-trek or May-combo. | Photodiodes by INCOHERENT, TIAs by SEMTREK or MACOMBO. | Slow orbit past the PD array and TIA call-outs |
| **s5c** | 48.20-49.52 | 1.32 e | Hatched by Oo-no-light. | Hatched by OUNOLIGHT. | Snap back, leg band, hatchery v3 (M) |
| **s6a** | 50.25-51.73 | 1.48 e | They observe you outside. | same | Gary walks out, unaware (coffee / phone) |
| s6b | 52.40-54.38 | 1.98 m | Bird to bird. No fiber required. | same | 3-4 sweeping beams, 2-3 switching point-to-point links |
| s6c | 54.80-56.65 | 1.85 m | Satellite handoff. Again. Again. | same | Satellite array, MIMO links re-configuring |
| **s6d** | 57.10-59.28 | 2.18 e | One flock, one mind: the Internet of Birds. | same | Wide on the SF skyline and the whole network; heads turn to camera; hero TX blink after the line (anchor 7.3 -> 7.1) |
| s7a | 60.10-61.57 | 1.47 m | Parodee. Typical build. Not real birds. | Parody. Typical build. Not real birds. | Card |

m = measured (rev 2/3 take), e = estimate. Bold = changed in rev 4.

| Scene | Words | Speech (s) | Air at end (s) |
|---|---|---|---|
| S1 | 21 | 8.5 | 0.57 |
| S2 | 20 | 7.7 | 0.85 |
| S3 | 20 | 8.3 | 0.61 |
| S4 | 17 | 6.8 | 1.04 |
| S5 | 19 | 7.4 | 0.48 |
| S6 | 22 | 7.5 | 0.72 |

Total 125 words (S7 included), about 46.2 s of speech in S1-S6 (rev 3 as voiced: about 45 s), plus 1.5 s on the S7 card.

### 12.3 Design notes

- S3 opens directly on the lasers and their maker (owner: no filler). The laser vendor is said only in S3, and S5 names only the PD and TIA vendors, so no vendor is said twice. INCOHERENT appears once, for the photodiodes.
- S4 (what a PD and a TIA do) and S5 (who makes them) both say "Photodiodes" and "TIA", but the information is different, so this is not a repeat. If the echo bothers the owner, use S5 alt B, which leads with the vendor name.
- No line suggests Gary is afraid. "They observe you outside." and the s6d network line describe the birds; Gary is only addressed in s4d ("Hi, Gary") and as the viewer's "you".
- Not said: "I-H-S" (it adds 0.5 s; see the s2a alt); "Source Photonics" (the owner named it, but in `research/form_factors_and_companies.md` it is a module maker with HQ/owner unclear and no laser role, and it has no parody name).

### 12.4 Fact check (new claims)

| Line | Claim | Source |
|---|---|---|
| s1d | OSFP is the workhorse of AI data centers (owner's text) | `research/form_factors_and_companies.md` A: OSFP "mainstream for AI" (agent assessment) ; DR8 volume, hero claim 10 |
| s2a | OSFP uses an integrated heatsink | `research/hero_800g_osfp_dr8.md` claim 7 (Heatsink: "OSFP: integrated heatsink", MSA sec 1) |
| s3a | Lumentum makes CW DFB lasers (DIMENTUM) | hero claim 1 (Lumentum 800G 2xDR4 OSFP: 1311 nm CW DFB); form factors C (Lumentum: EML, CW lasers) |
| s5b | Coherent sells 8-ch PIN photodiode arrays (INCOHERENT) | `research/rx_optics_refs.md` S1 (Coherent 1x8 500 um pitch PIN array, 100G PAM4) |
| s5b | Semtech makes octal TIAs for 800G DR8 (SEMTREK) | hero claim 4 (Semtech GN1825); `research/rx_optics_refs.md` S2 |
| s5b | MACOM makes TIAs (MACOMBO) | form factors C (MACOM: linear drivers, TIAs) |
| s5c | Innolight = module maker (OUNOLIGHT) | form factors C |
| s6d | "Internet of Birds" (parody of IoT; no claim) | - |

MACOMBO is named for TIAs only, not drivers: the hero's modulator driver is integrated in the DSP (hero claim 2, `research/tx_optics_refs.md` 3.3 Driver).

### 12.5 Say-spellings for A to verify (Whisper)

| Show | Say (primary) | Fallbacks |
|---|---|---|
| OUNOLIGHT | Oo-no-light | Ooh-no-light; Ou-no-light |
| DIMENTUM | Dim-en-tum | (DevLog-000 section 3) |
| INCOHERENT | In-co-here-ent | Incoherent (plain word; Whisper will spell it in lowercase, which is acceptable) |
| SEMTREK | Sem-trek | Sem trek |
| MACOMBO | May-combo (MACOM is said "MAY-com") | Mack-combo |
| Integrated heat sink | Integrated heat sink | keep `tail_fix` on "sink" |

### 12.6 Alternatives

New S5 vendor line (s5b):

| | Say | Show | Est. |
|---|---|---|---|
| Primary | Photodiodes by In-co-here-ent, T-I-As by Sem-trek or May-combo. | Photodiodes by INCOHERENT, TIAs by SEMTREK or MACOMBO. | 3.44 s |
| Alt A | Photodiodes by In-co-here-ent, T-I-As by Sem-trek. | Photodiodes by INCOHERENT, TIAs by SEMTREK. | 2.80 s (MACOMBO on the label only; 0.6 s more air before the hatchery) |
| Alt B | In-co-here-ent photodiodes, May-combo T-I-As. | INCOHERENT photodiodes, MACOMBO TIAs. | 2.64 s (leads with the vendor; avoids the "Photodiodes" echo of s4b; SEMTREK on the label only) |

New S6 final line (s6d):

| | Say / show | Est. |
|---|---|---|
| Primary | One flock, one mind: the Internet of Birds. | 2.18 s |
| Alt A | Welcome to the Internet of Birds. | 1.97 s |
| Alt B | The flock is the computer. (riff on "the network is the computer") | 1.65 s |

Optional, not requested: s6c alt "Many links, many satellites. Always reconfiguring." (about 3.0 s) matches the new MIMO staging (V4), but it only fits if s6d is Alt B. All alternatives are also in the JSON under `alts` / `say_fallbacks`.

### 12.7 Change log vs rev 3 (as voiced)

| id | Rev 3 | Rev 4 | Reason |
|---|---|---|---|
| s1d | Fancy name for a pigeon. | Workhorse of modern data centers. | Owner text (V2); anchor 7.9 -> 7.45 |
| s2a | Feathers? Heatsink. | Feathers? Integrated heat sink. | Technical name (V2); OSFP MSA term |
| s3a | Inside, real quick. | Two D-F-B lasers, by Dim-en-tum. | No filler; lasers + maker first (V2) |
| s3b | Two D-F-B lasers: steady light, zero information. | Steady light, zero information. | Laser count moved to s3a; anchor 2.0 -> 2.7 |
| s5b | A microcontroller snitches: temperature, laser current, light levels. | Photodiodes by In-co-here-ent, T-I-As by Sem-trek or May-combo. | Microcontroller line dropped; PD / TIA vendors (V2) |
| s5c | Hatched by Out no light. (OUTNOLIGHT) | Hatched by Oo-no-light. (OUNOLIGHT) | Rename (V1) |
| s6a | They follow you outside. | They observe you outside. | Owner text (V2); Gary unaware |
| s6d | Smile. The cameras are in on it. | One flock, one mind: the Internet of Birds. | Grander ending: IoT / distributed intelligence (V2); anchor 7.3 -> 7.1 |
| others | - | unchanged | |

### 12.8 Notes for A and S

- If S5 overflows after voicing, use s5b Alt A first. If S1 overflows, the s1d anchor can move to 7.40.
- S timing changes: the beak close-up moves to about 12.3-14.2. In S3 the DFB switch-on lands on "lasers" (about 20.9) and the MMI-tree light-up runs under s3b (22.7-24.7). The PD/TIA call-outs go under s5b (44.2-47.6). The wide network shot goes under s6d (57.1-59.3) with the TX blink after it.
- The S5 telemetry HUD (heart-rate gag) no longer has a VO line. Keeping it on screen is S's call.

## 13. Narration rev 6: assembly process lines (2026-10-10, DevLog-020 W17)

Brief (coordinator): the owner doubled the OUNOLIGHT assembly sequence (slot 92 -> 184 film frames, about 47.9-54.0 s; film about 67.07 s) and asked for narration that describes the process. Write 3-4 short lines (14-18 words) in his style, one per step; keep or fold in the hatching line; no repeats; facts only from `research/assembly_refs.md`. Base: `narration.json` rev 5. Output: `scripts/audio/narration_rev6_draft.json` (only S5 changes; S6/S7 targets shifted +3.067 s; `total_s` / `scene_bounds` set for 67.067 s, which A owns).

Timing: a short-line model fitted on 39 measured takes of 2.2 s or less (dur = 0.129 s x syllables + 0.011 s x pauses + 0.69 s; RMS 0.17 s, max 0.39 s). The four lines estimate 5.54 s plus 3 breaths, about 0.2 s longer than the slot. Laid out like `make_vo.py`, they are pulled back to end at 53.95 (0.12 s before the S5 end), so the first line starts at 47.75, during the fade-in. Step times are estimates: M's v4 events doubled (A 0-1.8 s into the slot with DFB on at about 1.3 s; B about 1.8-3.0 s; C lid about 3.1-3.7 s; D about 4.2 s to the end). A should snap to M's v4.2 event sidecar.

| id | Target | Est. | Say | Show | Step (anchor) |
|---|---|---|---|---|---|
| s5c | 47.75-49.24 | 1.49 | Lens, isolator, lens. | same | A: Si lens, micro isolator, Si lens placed, DFB dark (slot start) |
| s5d | 49.46-50.82 | 1.36 | Laser on, peak, cure. | same | A: DFB on after the tool leaves, coupled power peaks, UV cure locks it (S2_dfb_on, about 49.2) |
| s5e | 51.04-52.26 | 1.22 | Bond it. Seal it. | same | B: RF + DC wire bonds; C: dispense, lid pressed, sealed (lid about 51.0-51.6) |
| s5f | 52.48-53.95 | 1.47 | Passed. Hatched by Oo-no-light. | Passed. Hatched by OUNOLIGHT. | D: MCB final test -> PASS (heads docked about 52.1; PASS near the slot end) |

15 words. The hatching line is kept, with "Passed." (the test verdict) folded in front of it.

Facts (`research/assembly_refs.md`):
- Lens + isolator + lens laser train: fact 16.
- Active alignment (laser on, coupled power maximized while the lens moves, then epoxy, UV cure with position tracking): facts 2-4, 6.
- Ball-wedge wire bonding: fact 8.
- Dispense and housing close: section 1 step 4.
- PASS on the final-test screens: section 6.2.

The step list is per the coordinator brief. The test details (CMIS/DDM, TX eye per lane, RX sweep, 8-lane BER) are shown on screen, not spoken. Saying them would cost about 3 s, and "eight lanes" was already said in S3.

Not used:
- FABRINOT. The set is the OUNOLIGHT hatchery, so "assembled by FABRINOT" would claim that a contract manufacturer built an OUNOLIGHT module. `research/form_factors_and_companies.md` does not support that.
- Test jargon (TDECQ / OMA / ER), for the same time reason as above.

No repeats: lens, isolator, bonding, sealing and test have not been said before. "Laser" was said in S3 (who makes it); here it is said for the switch-on.

Alternates:

| id | Alt A | Alt B |
|---|---|---|
| s5c | Two lenses, one isolator. (1.73 s; per-laser count, fact 16) | Lens. Isolator. Lens. (about 1.5 s; staccato) |
| s5d | Laser on. Peak. Freeze. (1.36 s; "freeze" = the UV cure locks the position) | Laser on, peak the power, U-V cure. (2.0 s; only together with s5f alt A) |
| s5e | Wire bonds. Lid on. (1.22 s) | Bond the wires, close the lid. (1.47 s) |
| s5f | Hatched by Oo-no-light. (rev 5 take, measured 1.22 s; safest fit) | Tested, passed, hatched by Oo-no-light. (about 1.6 s) |

Notes for A:
- If S5 overflows after voicing, swap s5f to alt A first (-0.25 s).
- "U-V" is spelled as letters only in the s5d alt B.
- Keep `tail_fix` on s5f (final "light", as in rev 5).

- 2026-10-10: rev 6 draft written (section 13); assembly_refs.md read; narration_rev6_draft.json from rev 5 with S6/S7 +3.067 s.
