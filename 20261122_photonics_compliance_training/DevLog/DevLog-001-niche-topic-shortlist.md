# DevLog-001: Niche topic shortlist and course 1 proposal

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Proposal for review. Nothing built. Physics below is from domain knowledge; every claim and citation is marked to verify before scripting |
| Depends on | `DevLog-000-kickoff-topic-and-course-proposal.md` (genre style bible section 2, production plan 4.2, his feedback 5.1) |
| Authors | Claude, at Wentao Jiang's request |

## 1. Selection criteria (from his feedback)

- Deeply technical; audience of order a dozen to a few hundred people worldwide.
- Pitfalls that experts in the area might not have thought about, not textbook basics.
- No overlap with any training that really exists (connector hygiene, laser safety, ESD (electrostatic discharge), cleanroom gowning).
- Visual: the physics should read better as animation than as a paper figure.

## 2. Meta frame: course catalog

The player opens on a fake LMS (learning management system) dashboard: "Your organization has assigned you 6 required trainings. 0 of 6 complete. Due: 2026-12-31." Each tile is a niche course. Course 1 is built first; the rest show as locked "Coming soon" tiles until made. This lets the series grow without re-architecting.

## 3. Shortlist

| # | Course | Who really understands it | Visual hook | Verdict |
|---|---|---|---|---|
| T1 | Polarization management on a PIC (SOI, SiN, TFLN (thin-film lithium niobate)) | PIC designers who have been burned by mode hybridization; x-cut LN ring people | Mode profiles morphing through an anticrossing; field arrows rotating; a Poincare sphere drifting while someone touches the fiber | **Recommended course 1** |
| T2 | Laser mode hops and optical feedback (DFB, DBR, hybrid III-V/Si external cavity, self-injection locking) | Tunable and narrow-linewidth laser people | Longitudinal-mode comb sliding under a gain/filter envelope; hysteresis loop; a wavemeter calmly averaging over a hop | Recommended course 2 |
| T3 | TFLN modulator bias drift and low-frequency Vpi | A few dozen LN modulator groups | Bias point walking off quadrature over hours; Vpi changing with frequency | Strong, his home field |
| T4 | Laser noise in cavity-optomechanical cooling and thermometry | Order a dozen groups | Phase-noise sidebands posing as phonons; fake sideband asymmetry | Most niche; hardest to visualize |
| T5 | Microring-bank wavelength locking in DWDM (dense wavelength-division multiplexing) links | CPO and optical I/O ring teams | Rings locking onto the wrong comb line or wrong FSR (free spectral range); thermal triangle | Overlaps Gary film CPO scene; hold |
| T6 | Optical-pump heating in microwave-to-optical transducers | Order a dozen groups | Quasiparticles flooding a qubit when the pump turns on | Very niche, ties to his PhD; hold |

## 4. Course 1 (T1): "Polarization Awareness and Mode Purity in the Integrated Workplace (PIC-TE0)"

Mantra repeated in every chapter: **"Know your mode. Check your symmetry. Trust no fiber."**

Cast: Gary (the PIC designer who tapes out without checking), manager, bystander NPC, host = "Polarization Compliance Officer".

| Ch | Title (on screen) | Pitfall (to verify) | 3rd-person vignette (Blender) | 1st-person scene (HTML) | Interaction |
|---|---|---|---|---|---|
| 0 | Welcome: This Training Is Mandatory for All Personnel Handling TM Light | Course rules; enforcement demo | Host at desk | N/A | Locked Next, tab-blur pause |
| 1 | Unconscious Conversion: Mode Hybridization in Tapers | In a vertically asymmetric cross section (air or different top cladding vs oxide below), TM0 and TE1 anticross at some width; an adiabatic taper through that width converts TM0 into TE1. This is the working principle of taper-based polarization splitter-rotators, and it also happens by accident: a sensing window that removes top oxide, a SiN overlay, a different cladding in a test structure | Gary's harmless taper; the light enters TM and leaves as TE1, then gets dumped in the next single-mode section. Gary: "But the taper is adiabatic." | Chat: "we opened the cladding window for the sensor, why did the loss jump 3 dB" | Slider: sweep waveguide width, live effective-index curves with the anticrossing and morphing field plots (code-generated) |
| 2 | Hidden Coupling: Slanted Sidewalls in Bends | Non-vertical sidewalls (typical for etched TFLN and some SiN) break the symmetry that keeps TE and TM uncoupled; bends then couple TE and TM | Gary's rib waveguide with 60-something degree sidewalls goes round a bend; a TM ghost appears at the output | Video call: "the simulation used vertical sidewalls" (everyone nods) | Drag the sidewall angle and bend radius; conversion meter |
| 3 | Respecting Crystal Orientation: x-cut Anisotropy | In x-cut LN, a TE mode's index changes with propagation direction (ne along y-propagation geometry, toward no as the bend turns); at some angles TE and TM become phase matched and convert. Rings and racetracks then show extra dips or distorted resonances that depend on where the bends point | Gary's racetrack rotated 90 degrees on the die "to fit the layout"; the ring now converts in its bends | Phone: photo of a weird ring spectrum, "is this a higher-order mode?" | Rotate a racetrack on an x-cut die; resonance spectrum updates |
| 4 | Mistaken Identity: Which Resonance Is That? | TE, TM and higher-order mode families all show up in a ring spectrum; Q is reported on the wrong family; polarization paddles optimized at one wavelength drift across a broadband sweep; check with group index from FSR | Gary reports a record Q; replay shows it was the TM family | Slack-like thread: "new record Q" then "which polarization?" then silence | Label the mode families in a spectrum by FSR (forgiving quiz with "That's right!") |
| 5 | Trust No Fiber: Off-chip Polarization Drift | SMF (single-mode fiber) polarization drifts with temperature and handling; a polarization-selective grating coupler converts that into power drift that gets blamed on the device; PM (polarization-maintaining) fiber launched off-axis acts as a temperature-dependent waveplate; key misalignment limits extinction | Gary leans on the fiber and the "device drift" follows his elbow; HVAC (heating, ventilation and air conditioning) cycles in the background | Video call: "the device is unstable" with the fiber visibly swaying on camera | Poincare sphere: move the fiber, watch the state wander and the grating-coupler power track it |
| 6 | Diversity and Inclusion: Polarization-Diversity Circuits | Splitting into two paths and rotating one fixes polarization dependence only if both arms match: unequal loss gives PDL (polarization-dependent loss); unequal length gives skew; the splitter-rotator's extinction ratio sets the floor | Two identical-looking arms; one quietly longer | Chat: "PDL spec failed, but both arms are identical in the schematic" | Match the two arms (length and loss sliders) until PDL meets spec |
| 7 | Final Assessment | 10 questions from a pool, 80% to pass, unlimited retries, attestation, certificate | N/A | N/A | Quiz |

Chapter 6 title is a pun on a real corporate training category. Swap it if it reads as off-topic or edgy: alternative "Equal Treatment of All Polarizations".

### 4.1 Course 2 (T2) chapter sketch: "Mode Hop Prevention and Feedback Awareness (LAS-1550)"

| Ch | Title | Pitfall (to verify) |
|---|---|---|
| 1 | Unwanted Feedback | Small back-reflections (connector, facet, grating coupler, ring backscatter) push a semiconductor laser through feedback regimes up to coherence collapse; a return loss that is fine for a datacom link is not fine for a narrow-linewidth laser; isolator-free integration makes this the default |
| 2 | Direction Matters: Hysteresis | Hop points differ for tuning up vs down in current or temperature; spectroscopy sweeps in one direction only see one branch |
| 3 | Hidden Hops: Your Instruments Are Averaging | Wavemeter and OSA (optical spectrum analyzer) update rates and resolution bandwidth hide fast hops or two-mode operation; check with a heterodyne beat, RIN (relative intensity noise), or a reference cavity |
| 4 | Synchronized Tuning | Grating angle and cavity length must track for mode-hop-free tuning (pivot-point geometry in Littman-Metcalf); in hybrid Vernier-ring lasers gain current, phase section and ring heaters all move the cavity phase and cross-talk thermally |
| 5 | Spatial Hole Burning | Quarter-wave-shifted DFB lasers become multimode at high current through longitudinal spatial hole burning |
| 6 | Self-Injection Locking: Phase Is Not Optional | Locking range and linewidth reduction depend on the feedback phase from the microresonator; the wrong phase gives hops or instability |

## 5. Facts and citations to verify (candidate primary sources)

| Claim | Candidate source (to verify) |
|---|---|
| Taper-based PSR via TM0-TE1 hybridization in vertically asymmetric SOI | Dai and Bowers, Opt. Express 2011 (PSR on silicon nanowires) |
| TE-TM coupling in bends from slanted sidewalls | To find (SOI and TFLN bend-conversion papers) |
| x-cut LN bend-angle-dependent TE-TM conversion in rings | To find (x-cut TFLN microring anisotropy papers) |
| Feedback regimes and coherence collapse | Tkach and Chraplyvy, J. Lightwave Technol. 1986 |
| Mode-hop-free pivot geometry | McNicholl and Metcalf, Appl. Opt. 1985 |
| Spatial hole burning in quarter-wave-shifted DFB | To find (DFB laser literature) |
| Laser noise in optomechanical thermometry (T4) | Safavi-Naeini et al., New J. Phys. 2013; Jayich et al., New J. Phys. 2012 |

## 6. Questions for Wentao (in chat 2026-11-22)

1. Course 1 topic: T1 polarization (recommended), T2 mode hop, or another from section 3?
2. Catalog frame: one course built, others shown as locked tiles. OK?
3. Format: interactive web course plus 60 s trailer, or linear video only?
4. Length per course: about 10-15 min total, 1.5-2 min per chapter?
5. Tone: deadpan earnest throughout, or escalating Gary-film consequences?
6. Public at outside5sigma.com/<name>; parody vendor names carry over?

## 7. TODO

- [x] Shortlist and course 1 + 2 chapter plans
- [ ] Wentao picks topic and answers section 6
- [ ] Research pass on section 5 (one subagent per claim group), cache under `references/`
- [ ] Chapter scripts + narration JSON v1
- [ ] Course player skeleton (enforcement, quiz engine, catalog dashboard)
- [ ] Pilot one chapter end to end (Blender vignette fast tier + code-generated interaction)
- [ ] Fresh-context audit of technical claims

## 8. Progress log

- 2026-11-22: his feedback rejected common topics; shortlist T1-T6 written; T1 and T2 chapter plans drafted.
