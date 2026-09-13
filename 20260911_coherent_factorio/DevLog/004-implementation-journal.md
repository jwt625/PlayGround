# First outpost — implementation journal

## 2026-09-11 — Starting implementation

Repository contains design documents, 11 transparent sprites, and a vector UI kit; no game runtime exists yet. Implementing milestones in 003 in dependency order. Preserve simulation/rendering separation and test energy/resource accounting independently of the UI.

Current work: foundation and scattering-network model.

## Outstanding risks

- Phase sensitivity must be understandable at tile scale; use a documented effective phase-per-tile model plus fine tuning.
- Sprite projection and anchors remain approximate; gameplay footprints and ports are defined in world coordinates.
- Small-map balancing needs human playtesting even if the loop passes automated tests.

## Simulation and interaction implementation

- Implemented complex Gaussian elimination and reciprocal port networks. Matched active sources, lossless hybrids, phase tuners, weak emitter reflection, matched dumps, and bidirectional lossy links are represented explicitly. Independent source groups solve separately.
- 16 unit/integration tests pass: power conservation, phase redistribution, reflections, source independence, singular-network failure, economy accounting, build validation, controller/frontier/commissioning, blueprint placement, heat damage/repair, persistence, occupied ports.
- Local power bus has a 16-tile radius and 240-unit supply per generator. It abstracts wiring/fuel. Over-demand shuts active loads down rather than silently granting power.
- Target projection uses a normalized coherent sum with a two-mode minimum and distance-dependent coupling. It bounds useful target power by radiated power; remaining radiation is accounted as off-target. This is a gameplay approximation, not a far-field diffraction solver.
- Controller uses a bounded two-degree trial adjustment each tick. Qualification tests 20 simulated seconds with a declared additional thermal disturbance, requiring at least 32 target units.
- Found and corrected a protection-model issue: a tripped passive input must reflect rather than continue absorbing a live incident field. Bypassed cooling can still cause damage.
- UI and canvas renderer implemented; entering browser validation.

Current limitations: blueprint captures the entire outpost; loading restores physical installation but clears qualification and stored blueprint (re-record after testing). Material transport is one directed route per extractor/assembler pair with shared assembly output stock. No individual belt tiles, avatars, dispersion, ecology growth, or FDTD yet.


## 2026-09-12 — Session recovery and transport continuation

Recovered the prior project conversation from the local Codex rollout ending in `01a09422-253b-7911-96fd-eea0e41fd004`. It confirmed the frontier-first gameplay, commissioning contract, provisional single-view assets, and the request to continue into typed ports/grid transports. The earlier journal stopped before M5 validation was recorded; current implementation and browser checks now cover that milestone.

Implemented the plan in [005 — Ports, routing and presentation](005-ports-routing-and-presentation.md): independent art mappings, rotated footprints and port normals, stored grid paths with optional diagonals/waypoints, real material transit/backpressure, explicit generator wires, and geometric phase/loss with a separate bend-radiation ledger. Save v2 retains routes/packets/orientation; legacy layouts migrate when routable. Blueprint capture includes external routes and placement validates before committing.

The range power bus and instant material transfers described above are historical behavior and are superseded. Shared assembly/crystal output, effective optical units, dense solver, and whole-outpost blueprints remain prototype choices. Rotated artwork currently uses canvas rotation; dedicated oblique views are still TODO.

Validation: 30 headless tests and three browser scenarios pass; production build passes. Reviewed starter and rotated-routing screenshots. Added regression coverage for diagonal paths cutting through footprint corners and blueprint route failures leaving the world unchanged. See 005 for current controls, effective bend equations, limitations and next priorities.


## 2026-09-12 — PHIELDWORKS world interaction and onboarding

Applied the user's decisions: direct camera control; tutorial around the established outpost; initially investigating creatures, with aggression gated behind a powered defense and sustained field exposure. Introduced a wired perimeter sentry and a 30-second grace period. Physics and simulation never import presentation/UI modules.

Resource and creature inspection, categorized construction with explicit requirements, minimap/pan/zoom, backdrop dismissal, eleven contextual tutorial lessons, and object-focused diagnostics are implemented. Save v3 restores ecology and supports v1/v2 migration. Main UI, gallery and docs use PHIELDWORKS; the legacy local-storage key remains for compatibility.

Generated the terrain/prop/crawler assets and nine equipment turnaround sheets. Dedicated fixed-camera source rectangles supersede the canvas-rotation fallback recorded above. Opaque output from the image tool required dark-matte assets and runtime compositing; final alpha and animation are art-polish follow-ups. Exact prompts, source provenance and the workspace deliverables are linked from `assets/world-generation-notes.md`.

Final validation: all 35 simulation tests, eight browser scenarios (including the full tutorial with defense and recovery), and the production build pass. See [006](006-world-interaction-and-onboarding.md) for final results and next steps.


## 2026-09-12 — Repo recheck and asset continuation

Reviewed committed progress at `5c65d26` with a clean working tree. Continued the requested map/animation/technology art pass in [007](007-map-animation-and-technology-assets.md). Eleven generated source files add four terrain variants, depletion/debris, 48 motion frames, eighteen technology illustrations and eight industrial science packages. Runtime animation follows production/movement state; a cached terrain layer and source-frame metadata keep presentation separate from the core. Technology dependency/status review is an asset-design surface, not a research unlock implementation. Exact prompts and source paths are preserved under assets.

Final asset-pass verification: production build and all 11 browser scenarios pass. Source-frame registration corrected clipped mechanisms without modifying generated PNGs. See 007 and the linked review pages for the delivered assets, prototype limitations and next priorities.
