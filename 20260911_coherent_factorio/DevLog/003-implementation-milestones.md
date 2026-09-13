# First playable outpost — implementation milestones

Started: 2026-09-11. First playable complete; ports/routing continuation tracked in [005](005-ports-routing-and-presentation.md).

Scope: desktop-browser prototype using TypeScript, a canvas world, and DOM UI. Implement a small complete resource-frontier loop with a renderer-independent simulation. The original artwork supported the first loop; the world-interaction milestone adds terrain and dedicated orientation sheets. This is the first short scenario, not the full 30–45 minute balanced slice.

## M1 — Foundation and physical model

- [x] Scaffold development server, TypeScript build, unit tests, browser tests.
- [x] Complex arithmetic and pivoted small dense linear solver.
- [x] Port-based scattering networks with bidirectional links and propagation loss/phase.
- [x] Independent source groups solve separately; powers add between groups.
- [x] Explicit source, passive component, link-loss, and terminal power accounting.
- [x] Tests: conservation, interference redistribution, reflection, independent sources, singular network handling.

## M2 — Factory world and persistence

- [x] Fixed-step world simulation, stable entity IDs, footprints, placement/removal.
- [x] Starter scenario, extraction, inventories, recipes, material transport, build costs.
- [x] Power supply and availability; no power means no production or field generation.
- [x] Save/load with schema validation; reset; pause and speed controls.
- [x] Tests for economy, invalid placement, topology changes, and save round-trip.

## M3 — Field engineering and consequences

- [x] Connect/disconnect individual field ports; inspect incident/outgoing power.
- [x] Tuning, slow thermal drift, field overlay and visible target delivery.
- [x] Target damage from useful delivered power; resource frontier unlock.
- [x] Dump heating, protection trips, equipment damage and repair/rebuild path.
- [x] Tests for phase-sensitive target delivery, damage, protection, and frontier access.

## M4 — Commissioning and reuse

- [x] Bounded automatic phase controller with explicit enabled state.
- [x] Acceptance test under declared drift, pass/fail/cancel and stored rating.
- [x] Blueprint capture/place with internal topology, costs, and local requalification.
- [x] Tests for test invalidation, control behavior, blueprint independence.

## M5 — Playable presentation and validation

- [x] Integrate sprites, resource counters, build palette, inspector, objectives, event feedback.
- [x] Usable camera, port hit targets, route selection, placement preview and keyboard shortcuts.
- [x] Browser checks: start, build, tune, connect, production, commissioning, save/load, reset.
- [x] Visually inspect browser screenshots; production build passes.
- [x] Record limitations, verified behavior, run instructions, and next TODOs.

## First-build choices

- A partially established expedition outpost teaches the basic links; players expand production and add/tune a second emitter branch.
- Simplified mixed ore converts into assemblies. Crystal is the unlocked frontier resource; these are prototype recipes.
- One narrowband field regime. Tile phase is explicitly a compressed effective path model, not an optical wavelength claim.
- Small dense network solving is sufficient for this map. No full-map wave grid, FDTD, dispersion, or stochastic multipath in this milestone.
- Beam intensity is a diagnostic visualization. The target model accounts for focused versus off-target radiation without claiming a computed full diffraction field.
- Broad roadmap mechanics remain deferred unless explicitly marked implemented.

See [implementation journal](004-implementation-journal.md) for progress, decisions, tests, and follow-up work.


## M6 — World interaction and onboarding (2026-09-12)

Follow [006](006-world-interaction-and-onboarding.md) for the accepted feedback, implementation notes, art limitations and human playtest checklist. Resource selection, moving investigative wildlife, basic defense, textures/dressing, separately redrawn orientations, direct-camera/minimap interactions, categorized construction, contextual tutorials and actionable diagnostics are implemented. Final expanded validation is recorded there.


## M7 — Map, animation and technology asset continuation

[007](007-map-animation-and-technology-assets.md) tracks the next art batch after commit `5c65d26`: four map textures, stateful debris/depletion, production and crawler animation, and a proposed technology-art graph. The new art is mapped independently of the simulation; no research economy or unlock gates are implied.
