# Gameplay decisions and frontier experiment

Date: 2026-09-11

This augments [the original design](000-design-doc.md). It records the discussion following that document without replacing the long-term vision. Confirmed decisions below take precedence over conflicting interpretations of the original; proposed details are implementation hypotheses to playtest.

## Confirmed direction

- Audience: factory-game players without wave-physics background, plus engineering enthusiasts.
- Main activities: expanding production, debugging field networks, and commissioning reusable modules.
- Early reward: use field-engineered defenses to open a resource frontier. Later reward: improve fabrication yield with precision fields.
- Failures may scrap products and destroy equipment; failure is not restricted to temporary throughput reduction.
- Commissioning includes a short automatic acceptance test after manual tuning.
- Immediate deliverable: a focused playable experiment, not a complete factory game or a 5–10 hour vertical slice.
- Simulation direction: a circuit-level wave model with slowly varying envelopes and simple stray-light/interference modeling. Local 2D FDTD is an upper-end option for a specific need, not a required starting point. Physical scale and representation need further discussion.
- Potential defense demands include pulsed delivery and coherent focusing. The exact first weapon implementation remains a prototype choice.

## Core reward hypothesis

“I built something powerful, made it reliable, and can now stamp out more of it.”

Production gives steady progress. Debugging creates a visible step change in useful output. Commissioning turns a successful experiment into repeatable industrial capacity. That capacity opens territory and resources, funding the next expansion.

The proposed early loop is:

```text
extract resources → manufacture equipment → expand installed power
→ diagnose field delivery → concentrate power on a target
→ clear frontier → access richer resources
→ commission and replicate the installation
```

The later fabrication loop substitutes good products/min and access to advanced materials for damage and territory. Both reward controlled useful output rather than power generation alone.

### Numbers and visible rewards

| Measure | Purpose | Presentation |
| --- | --- | --- |
| Components/min and inventory | Ability to build more | Production panel and material movement |
| Installed power | Industrial investment | Emitter/network summary |
| Power on target | Useful delivery at a specified target and distance | Target meter plus field visualization |
| Qualified capacity | Output sustained under a declared commissioning test | Module rating and pass event |
| Accessible deposits | Concrete expansion reward | Newly reachable map area |
| Operational modules | Scale through reuse | Factory capacity summary |
| Good products/min, later | Precision fabrication reward | Accepted output and scrap counts |

Efficiency is a diagnostic and economic consideration, not the sole victory metric. Always distinguish installed, delivered, rejected, absorbed, and stray power. Show the visual effect and world consequence alongside a number increase.

Tech progression should unlock new actions and architectures. Proposed: combine production-funded research with first-time capability milestones; do not demand repeated manual demonstrations for every research node.

## Proposed first scenario

Target playtime: 30–45 minutes, subject to playtesting.

Objective: establish a small production line, commission a multi-emitter installation, and clear access to a richer deposit.

1. Extract starter resources and manufacture replacement emitter/network parts.
2. Operate one emitter against a nearby weak target.
3. Add emitters to challenge a tougher, more distant stationary armored organism.
4. Discover that installed power does not equal useful target power.
5. Inspect useful/rejected power and relative phase; adjust routing or tuning.
6. Concentrate enough power on the target to clear the frontier.
7. Install a controller and run an acceptance test with a declared small disturbance.
8. Replicate the design at a second site; resolve a different external routing constraint.

Proposed first mechanic: coherent focusing. Pulsed delivery, moving targets, fabrication yield, multiple species, and advanced communications follow only if the first loop is enjoyable.

Initial content: starter extractor, simple assembler, material transport, power supply, emitter, reference source, transport segment, splitter/combiner, phase tuner, monitor, controller, dump/cooling structure, one armored organism, and two resource grades. This is a scope list, not a requirement to build a full logistics engine before testing field interaction.

### Playtest questions

- Can a player explain why more emitters did not automatically solve the target?
- Can they locate and fix a field-delivery problem without knowing optical terminology?
- Can they predict the direction of a tuning change and verify it visually?
- Is the successful correction visibly satisfying and economically useful?
- Does commissioning feel like finishing a design?
- Does the second installation feel like applying knowledge rather than repeating chores?
- Can they recover from damaged equipment using retained production capacity?

## Network services and demand

Keep service requirements distinct even when shared hardware carries them.

| Service | Consumer | Consequence of insufficient service |
| --- | --- | --- |
| Electrical power | All active machinery | Reduced operation or shutdown |
| Process field | Weapon or later fabrication cell | Insufficient target delivery or poor yield |
| Reference | Coordinated emitter sectors | Poor combining/focusing |
| Communication, later | Tracking, inspection, remote coordination | Stale control, buffering, lower throughput |

Pulse spectral bandwidth and communication data rate are different quantities. A short-pulse weapon supplies a reason to model pulse dispersion, but does not by itself establish demand for a high-data-rate network. Moving-target sensing and distributed inspection are candidate data consumers.

## Diagnosis, damage, and mastery

Effect-before-name must not mean failure-before-diagnostics. Introduce each consequential failure mode with an affordable symptom-level instrument and at least one understandable mitigation.

Proposed damage chain: rejected power → heating → shrinking safety margin → component damage. Distinct causes may need distinct chains; damage should follow modeled stress rather than a random failure roll.

Buildable protection includes dumps, cooling, trips, branch isolation, and eventually redundancy. Protection has cost and operating limits. Preserve enough starter production to recover from a learning failure in the first scenario.

The mastery loop is encounter → measure → fix → automate → reuse. The player should eventually manage proven subsystems instead of continuously retuning every element.

### Commissioning contract

A reusable module declares supply requirements, reference requirements, rated useful output, tuning range, cooling demand, allowed return power, and protection settings.

Proposed state machine:

```text
uncommissioned → tuning → testing → qualified
                       ↘ failed test → tuning
qualified → operating-limit violation → degraded or tripped → recommission
```

An acceptance test checks output and stress limits for a specified duration under a declared disturbance. Store the test conditions with the rating; passing is not a guarantee under arbitrary future conditions. A blueprint retains topology and settings, while each placed instance verifies local services and conditions.

## Simulation boundaries and unresolved choices

Start with narrowband scattering networks, slow thermal/control dynamics, and a simple aperture-field calculation. Extend to envelopes when pulse behavior is actually part of the playable loop. Environmental coupling starts with a bounded set of identifiable paths.

Separate map distance from fine phase adjustment: routing determines length, attenuation, delay, and exposure; tuners supply fine adjustment; environmental changes introduce drift; controllers maintain operation within range and bandwidth limits. Exact unit mapping is unresolved. A Factorio-like map scale alone does not determine wavelength, phase sensitivity, time compression, or control rates.

Do not change existing physical behavior merely because a technology is researched. Research exposes instruments and new operating regimes. Reduced models must remain consistent where representations overlap.

Technical clarifications to the original document:

- Incoherent inputs still obey component port routing. A balanced lossless two-input/two-output combiner does not send all incoherent input power to one useful port.
- The −2 dB versus −31 dB path example gives a direct interference cross-term of approximately 7.1% of primary-path power when both numbers share a power reference. Large throughput swings need a sensitive load or feedback mechanism; sustained oscillation also needs dynamics.
- Coherence identity should support shared-origin branches and eventual reference locking between sources. A source ID alone is not the complete endgame model.
- Coherent path strengths do not generally form additive percentages of detected power. Show isolated strength, relative phase, and/or the predicted effect of blocking a path.
- Path pruning needs a stated error budget and stable thresholds. Many weak coherent contributions cannot automatically be treated as incoherent background.
- Keep temporal and spatial coherence distinct when implementing decorrelation costs; spectral broadening does not automatically destroy spatial focusing capability.
- A steady-state linear solver alone does not model source instability, saturation, or damage. Add only the slow dynamics needed for a selected interaction.

Further discussion: map units and game-time scaling; source/reference locking model; drift amplitude and controller timescales; damage and repair economy; environmental coupling limits; whether any specific local interaction justifies 2D FDTD.

## Initial asset direction

Provisional visual direction: oblique top-down industrial machinery, readable silhouettes, muted metal/ceramic materials, restrained field/status color, upper-left lighting. This is an art experiment, not a locked production specification.

First assets: an intact field emitter and an intact stationary armored frontier organism. Keep field overlays and beams separate from object artwork so they can reflect simulation state. Generated images are initial static artwork; footprint, anchor, rotation coverage, damage states, and final engine integration require follow-up.

See [asset notes](../assets/README.md) for files and generation provenance.
