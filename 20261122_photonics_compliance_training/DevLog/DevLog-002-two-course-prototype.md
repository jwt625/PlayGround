# DevLog-002: Two-course interactive prototype (UI/UX, style, tone review)

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Prototype built for his review after a 1 h absence. Video = keyframe animatics (Blender pending). Narration = browser speech synthesis (Kokoro pending) |
| Depends on | DevLog-000 (genre bible), DevLog-001 (first shortlist, superseded), `research/mine_*.md` (six corpus surveys) |
| Authors | Claude, at Wentao Jiang's request |

## 1. Instructions (verbatim, 2026-11-22)

```text
i have plenty of topics and materials from the photonics atlas project and my blog. Dig deeper and come up with a better proposal.
```

```text
I'm heading out for 1 h, and I want you to sample some of our past conversations (read local claude code cache and log etc.) and choose 2 topics that i would choose to build such interactive training course. html. Use best practice and my preference on the tech stack if possible. I want them to be ready to review when im back in ~1h ish. Ok to put placeholder or keyframes for videos since blender vid would take more iterations and rendering time. Goal is to get overall UI/UX, style and theme and tone etc. reviewed. Turn the parody knob to the max, the talkpoints should be so thorough the rules and precautions should get burnt into the viewer's subconsciousness.
```

```text
the sampling is for your understanding of my high level preferences and requirements to build a mental model of me and to guide you to pick after you get reports back from the surveying agents
```

## 2. Research surveys (six read-only subagents, files in `research/`)

| File | Corpus | Clusters / pitfalls |
|---|---|---|
| `mine_blog_longform.md` | Blog long-form posts + photonics PlayGround projects | 8 / about 100 |
| `mine_ofs_01_60.md` | Weekly reading posts 1-60 | 7 / 65 |
| `mine_ofs_61_plus_ofc.md` | Weekly reading posts 61-124 + OFC 2026 notes | 8 / about 98 |
| `mine_photonics_atlas.md` | photonics-atlas paper extracts | 6 / about 75 |
| `mine_eo_ln.md` | eo-modulator-atlas + lithium-niobate | 7 / 77 |
| `mine_packaging_atlas.md` | advanced-packaging-atlas | 8 / about 125 |

No fresh-context audit has been run on these files yet; each says which rows are its own inference.

## 3. Mental model used to pick (from his memory files plus sampled sessions)

Sampled: lithium-niobate baby-mode sessions (2026-10-28/29) for how he reviews interactive pages; the interconnect and pigeon film DevLogs for tone.
- He goes for his own first-hand material (LN lab and simulation stories) and topics he named (mode hops).
- Interactive pages: immersive, auto-advancing, chapter end flows straight into the next chapter (he called the return-to-menu "bs"), obvious affordances (animated hand hints), physically correct animation direction.
- Code-generated visuals, no image generation for technical content; dense, sharp corners.
- Humor: literal and escalating, parody names close to real ones.

## 4. Decisions (Claude, for his review)

| # | Decision | Why |
|---|---|---|
| D1 | Course 1: "Ferroelectric Workplace Conduct" (lithium niobate: crystal-axis bookkeeping, x-cut bend conversion, photorefraction, pyroelectric charge, DC versus RF V pi, simulation integrity) | Densest first-person material across four surveys; his PhD field; audience of order 100 |
| D2 | Course 2: "Feedback, Reflection and Zero Tolerance for Retaliation" (back-reflection, parasitic cavities, mode hops, reciprocity, backscatter in high-Q rings) | He named mode hops; atlas + weekly posts give sourced items; the anti-retaliation pun maps onto back-reflection |
| D3 | Catalog dashboard with 6 more locked "Coming soon" courses taken from the other strong clusters (adhesive index cliff, warpage, DC versus RF, metric reporting, simulation integrity, ultrastable cavities) | Shows the series; cheap |
| D4 | Static no-build HTML + classic scripts (no ES modules) so `player.html` also opens from `file://` | His "static no-build HTML for small tools" default; reviewable without a server |
| D5 | Light corporate LMS palette by default with a dark toggle; sharp corners everywhere | Genre fidelity at max parody vs his dark default; corners stay sharp per his rule. Open for his call |
| D6 | Narration: browser speech synthesis now; Kokoro clips later | Zero generation time for a UI review |
| D7 | Reviewer mode `?review=1`: Next always enabled, modules unlocked, idle timeout off, voice-off switch with 4x captions | He must be able to review in minutes; learner mode keeps every enforcement |
| D8 | Fictional employer "Intracavity Photonics, Inc."; host "Pat Reynolds, Director, Office of Photonic Conduct" | Neutral, not a real company name as far as checked (not verified against registries) |

## 5. Enforcement features implemented (genre bible G7, G8)

- Next locked until narration ends and the activity is done; "Please listen to the narration in full".
- Window blur or hidden tab pauses everything; on resume the interrupted sentence repeats in full; blurred time does not count toward seat time.
- Seat-time counter versus "required 01:34:00".
- Speed menu: only 1.0x, others "(disabled)"; video scrubber refuses seeking; full screen refused; minimum volume enforced.
- Idle 3 min: "Are you still there?" 60 s countdown, then the slide restarts.
- Module interstitial: mantra reminder plus objectives, Continue locked for 6 s.
- Forgiving checks: "That's right!" voice; wrong answer struck out, long explanation read in full, options locked until it finishes, unlimited retries.
- Final assessment from a randomized pool, 80% on first attempts, unlimited retakes, attestation with full legal name, printable certificate ("valid 12 months, auto re-enrolled in 11").
- Ticker of the mantra across the bottom of every screen.

## 6. TODO

- [x] Surveys, topic pick, spec (`course/COURSE_SPEC.md`), engine, dashboard, validator, headless smoke test
- [x] Content: `course/courses/ferroelectric.js` (7 modules, 82 steps, 15-question pool, 5 widgets, about 31 min narration), `course/courses/feedback.js` (7 modules, 84 steps, 15-question pool, 3 widgets, about 25 min narration); validator OK; headless walk of all steps in light and dark, 0 console errors; learner-mode gating and blur pause checked
- [ ] His review of UI/UX, style, tone
- [x] Fresh-context audit (`research/audit_courses_v0.md`): 52 DOIs + 3 arXiv ids resolve and match; formulas re-derived OK; 10 must-fix items applied (section 8). Authors flagged from-memory items: Tkach and Chraplyvy DOI, IEEE 176 DOI, Myers 1995 coercive field, Weis and Gaylord pyroelectric range, Jalas reciprocity list, Luo 2018 / Weigel 2018 titles
- [ ] Kokoro narration + "That's right!" clips; Blender shot list from keyframe `shot` fields
- [ ] Commit when asked

## 7. Progress log

- 2026-11-22: six surveys returned; picked D1/D2; engine + dashboard built; stub course passes headless walk (18 steps, 0 console errors).
- 2026-11-22 13:10: both courses in; Blender shot notes moved under the video player; served at http://127.0.0.1:47293/ (tmux session `compliance-training`, `python3 -m http.server`). `file://` also works.

## 8. Audit corrections (2026-11-22)

| Course | Item | Fix |
|---|---|---|
| Feedback | Lapointe 2025 numbers | > 25 dB isolation, < 1.5 dB insertion loss across the C-band (was "1 dB") |
| Feedback | Reciprocity test mixed time reversal with Lorentz reciprocity | Backward launch into the forward output mode(s), all ports and modes, compare with forward transmission |
| Feedback | Nonlinearity listed as an equal isolation route | Caveat: power window only, limited by dynamic reciprocity (Shi, Yu, Fan, Nat. Photon. 2015, doi 10.1038/nphoton.2015.79) |
| Feedback | Marinins 2022 called a "silicon nitride chip" | InP DFB on a silicon photonics chip with SiN waveguides and grating couplers |
| Feedback | Unused Voloshin reference | Removed |
| Ferroelectric | Pyroelectric 9.5e-5 attributed to Weis and Gaylord | Weis and Gaylord give 4e-5; 9.5e-5 from Popescu et al., J. Appl. Phys. 2013 (doi 10.1063/1.4788696) |
| Ferroelectric | Coercive field 21 kV/mm cited to Myers 1995 | Volk and Wohlecke 2008, Table 7.1 |
| Ferroelectric | Drift time constant "microseconds to hours" | Thin film microseconds to milliseconds; hours only for bulk LN |
| Ferroelectric | Slow passage through the TE/TM crossing stated as fact | Labeled a plausible picture; quiz answers no longer depend on it |
| Ferroelectric | Spelled acronyms ("D C") on screen | Joined on screen; engine caption normalizer also joins spelled acronyms in captions |

Re-check after fixes: validator OK for both; headless walk of all 94 + 96 steps, 0 console errors.
