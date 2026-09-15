# Coding-agent handoff and task tracker

Updated: 2026-09-15

## Assignment

Build the simulation described in [initial discussion](../DevLog/000-initial-discussion.md), applying the newer [optics/rendering specification](../DevLog/001-cbc-optics-rendering-spec.md) wherever they differ. Default is **19 hex-packed channels**, independently controllable piston/amplitude/tip/tilt/focus, a wired 3D console plus schematic, and measured interference on movable sections with a 2π angular dome.

Start with T01–T03, produce a working optics/control slice, then continue through the dependent tasks. Do not substitute scripted success or a fake biological graph. Keep the simulation-only boundary. Do not modify unrelated sibling projects in the enclosing Git repository.

## Collaboration and status rules

Coding agent owns implementation and test execution. Reviewing agent owns specification updates, evidence review, reference research, and rendering asset preparation. Record implementation findings below; propose physics-contract changes explicitly. Checklists start unchecked because nothing has been implemented.

Each completed task must provide changed paths, exact commands, outputs/artifacts, limitations, and benchmark environment where relevant. Distinguish implemented, tested, and independently reviewed. A screenshot is not proof of correct interference or successful learning.

## Proposed stack / boundaries

TypeScript + Three.js browser client; fixed-step simulation worker; Python/NumPy reference optics and offline training/evaluation as needed. Pin versions at implementation time. Start with the simplest working CPU path and profile before choosing WebGPU/WASM. Preserve headless training without importing renderer code.

Suggested modules: `optics/`, `controllers/`, `connectome/`, `environment/`, `learning/`, `renderer/`, `analysis/`, `video/`, `configs/`, `tests/`. Match repository patterns if code arrives before implementation begins.

Shared contracts must cover:

- `ArrayConfig`: units, wavelength, aperture/power convention, geometry, stable channel IDs, actuator bounds.
- `ActuatorCommand` and `ActualActuatorState`: per-channel piston, amplitude/power mapping, direction, curvature, enabled state; simulation time and ownership.
- `FieldQuery`: world-coordinate samples or plane basis/grid; complex field and intensity outputs with normalization and validity domain.
- `AngularMap`: unit directions, solid-angle weights, power-density values, coverage, convergence metadata.
- `SimulationSnapshot`: immutable tick ID, actual/command state, observations, metrics, controller mode; rendering consumes one coherent tick.
- `RunManifest`: seed, config, code version, model/data provenance, checkpoints, test/evaluation IDs.

## Phase progress

Current phase: **Phase 1 — Foundations**. All work is TODO; no implementation has been validated.

- [ ] Phase 1 — Foundations (T01)
- [ ] Phase 2 — Validated optics and baseline control (T02–T03)
- [ ] Phase 3 — Interactive bench and measurement views (T04–T05)
- [ ] Phase 4 — Faithful animation and performance (T06)
- [ ] Phase 5 — Connectome controller and learning (T07)
- [ ] Phase 6 — Full-control 3D tracking (T08)
- [ ] Phase 7 — Fly assets and visual polish (T09)
- [ ] Phase 8 — Evaluation, videos, and delivery (T10)

### How to execute and update this tracker

1. Work through phases in order by default; respect task dependencies when starting independent preparation early. Asset research may begin before Phase 7, and T07 can begin after T03.
2. Mark a TODO complete only after its implementation and relevant validation pass. Add evidence paths under the task. Keep failed or partial items unchecked.
3. At each phase exit, complete its deliverables and gate checklist, update the current phase, and append a completion report. Continue into the next unblocked phase without requesting routine approval.
4. Track independent review separately from implementation completion. Do not claim review has occurred because tests passed.
5. For a blocker, record the exact missing input, failure, attempted resolution, and next action. Continue independent work; do not silently substitute a weaker requirement.

## Phase 1 — Foundations

**Goal:** establish reproducible configuration, channel identity, and execution contracts.
**Entry:** specification read; repository inspected.

### Phase deliverables and exit gate

- [ ] Deliver runnable browser scaffold, headless test command, and default 19-channel configuration.
- [ ] Implement the shared contracts listed above, with explicit units and serialization.
- [ ] Pass geometry and deterministic replay checks in T01.
- [ ] Document clean-install commands and save validation output.

### Implementation TODOs

### T01 — Scaffold and deterministic geometry [P0]

- [ ] Minimal browser and headless commands, pinned dependencies, README, configs.
- [ ] 19-emitter axial-coordinate geometry with explicit stable IDs; configurable smaller fixtures.
- [ ] Unit conventions, limits, seeded disturbances, fixed simulation timestep, command ownership.
- [ ] Tests: 19 unique sites, row counts 3/4/5/4/3, sixfold symmetry, nearest-neighbor spacing, center of array at origin; deterministic replay.

Done when: fresh checkout can run the optics test command and minimal application. Report commands, no undocumented local prerequisites.

## Phase 2 — Validated optics and baseline control

**Goal:** prove interference, steering, focusing, and optimization before detailed rendering.
**Entry:** Phase 1 complete.

### Phase deliverables and exit gate

- [ ] Deliver independent optical reference, fast evaluator, and saved numerical fixtures.
- [ ] Pass all nine T02 numerical checks and document approximation limits and tolerances.
- [ ] Deliver SPGD/random/analytic comparison over at least 10 fixed seeds, with curves and evaluation counts.
- [ ] Confirm metrics use linear physical data and hidden errors are excluded from learner observations.

### Implementation TODOs

### T02 — Independent optical reference [P0; T01]

- [ ] Launch aperture with piston, amplitude, tip/tilt, curvature; independent numerical propagator.
- [ ] Complex Gaussian field evaluator with correct launch phase reference, curvature, Gouy phase, and normalization.
- [ ] Arbitrary transverse/longitudinal field sections; far-field transform and angular mapping.
- [ ] Record model domain, clipping, grid spacing/extent, numerical precision, and convergence results.

Required numerical tests:

- [ ] Two identical coincident test fields: phase 0 gives 4× single-field intensity; π cancels to tolerance. This is an algebra test, not a physical power-conservation test of distinct apertures.
- [ ] Any global piston shift leaves intensity and metrics invariant.
- [ ] Positive/negative phase ramps steer to the predicted positive/negative direction; test both axes and wraparound.
- [ ] Gaussian radius versus distance and waist location agree with independent q-parameter predictions; changing curvature moves the waist, changing piston alone does not move a single channel's waist or axis.
- [ ] Two separated apertures produce the predicted fringe spacing; π phase shift moves maxima into prior minima.
- [ ] Finite-target phase alignment maximizes target intensity relative to seeded random phases; include nonzero range and transverse offsets.
- [ ] Power conservation across adequately sampled transverse planes, with clipping and quadrature errors reported.
- [ ] Halving sample spacing and expanding extent converge on intensity/PIB; include a deliberately undersampled failure fixture.
- [ ] Fast evaluator versus independent reference on representative sections within the stated approximation domain.

Suggested acceptance tolerances to justify: relative 1e-6 for float64 algebraic invariants; ≤1% power/width error and ≤2% normalized map RMS against reference in compatible Gaussian/paraxial fixtures; localization within one sample then demonstrate convergence. Do not relax thresholds merely to pass; explain approximation mismatches.

### T03 — Manual control and SPGD [P0; T02]

- [ ] Headless SPGD, random controller, analytic phase/focus reference; identical disturbances and evaluation budgets.
- [ ] Reproducible randomize, reset, enable/disable, per-channel manual adjustment.
- [ ] Command/actual distinction, phase wrap, saturation, drift, and hidden-error isolation.
- [ ] Save convergence curves and configurations across at least 10 fixed seeds.

Done when: baseline recovers static phase alignment and reports success distribution relative to ideal-array PIB; disclose any failures. Keep original Strehl >0.8 goal, with configuration-dependent justification. Count objective evaluations fairly, including both SPGD perturbation measurements.

## Phase 3 — Interactive bench and measurement views

**Goal:** make every channel inspectable and controllable through the optical scene.
**Entry:** Phase 2 complete.

### Phase deliverables and exit gate

- [ ] Deliver linked 3D console, 2D schematic, selectable wiring, and all per-channel controls.
- [ ] Deliver angular dome and movable transverse/longitudinal sections with explicit validity masks.
- [ ] Pass channel-selection/state consistency and centroid/quadrature/section-sampling tests.
- [ ] Save interaction evidence and field arrays for manual phase, steering, and focus sweeps.

### Implementation TODOs

### T04 — Functional optical bench and control board [P1; T03]

- [ ] 3D seed/splitter/branches/modulators/amplifiers/pointing-focus assemblies and 19 emitters.
- [ ] Physical console and accessible interactive 2D schematic share the same state.
- [ ] Selecting any channel highlights all associated paths, values, and beam; explicit optical/electrical/data legend.
- [ ] Phase rings; selected-channel controls for phase, amplitude, two pointing axes, convergence/focus; command and actual readbacks.
- [ ] Manual/analytic/SPGD modes and explicit ownership transfer, no competing control writes.

Done when: user changes a specific channel from either view and sees the same actual state and physics-derived field response. Supply automated interaction checks plus screenshots of selected CH01, CH10, and CH19.

### T05 — Dome and movable measurement sections [P1; T02,T04]

- [ ] Forward hemisphere with weighted angular power map and explicitly masked unsupported domain.
- [ ] Power-weighted mean direction, concentration, independent peak marker; correct zero-power handling.
- [ ] Centroid-following transverse section with range control; draggable/rotatable/frozen manual section; longitudinal section.
- [ ] Separate centroid/peak/target follow modes, stable basis near coordinate poles, raw versus visual smoothing.
- [ ] Linear/log intensity, calibrated legends, physical sample readout, same-tick scientific inset.

Tests: quadrature weights cover 2π for a full hemisphere; known symmetric angular fixture points along +Z; unequal two-lobe fixture gives analytic weighted direction; zero power yields unavailable centroid; map refinement converges. Confirm section samples equal direct field queries and that moving the target alone in manual mode does not move the beams. Test focus sweeps and phase sweeps with saved field arrays, not only screenshots.

## Phase 4 — Faithful animation and performance

**Goal:** animate measured optical behavior at usable interactive speed.
**Entry:** Phase 3 complete.

### Phase deliverables and exit gate

- [ ] Deliver translucent envelopes, field-derived interference, and labeled slowed phase overlays.
- [ ] Deliver reproducible reference scene gallery and pause/scrub behavior.
- [ ] Record frame and simulation timing against T06 budgets on named hardware/browser.
- [ ] Confirm camera, bloom, display smoothing, and rendering load do not change physics or metrics.

### Implementation TODOs

### T06 — Rendering fidelity and performance [P1; T05]

- [ ] Translucent envelopes follow actual beam axes and widths; interference overlays derive from total complex field.
- [ ] Slowed phase/wavefront layer and illustrative data pulses labeled; pause/scrub supported.
- [ ] Frame/tick timing instrumentation and selectable resolution; rendering load cannot change simulated dynamics.
- [ ] Target budget: 30 fps minimum, 60 fps preferred on documented development hardware at 1080p, 19 channels and one 256×256 section. Measure before accepting target; adaptive visual resolution allowed.
- [ ] Reference fixture gallery: one beam, two-beam phase sweep, random/locked array, steering, focus sweep, channel dropout, split lobes/centroid.

Done when: recorded captures correspond to saved states and numerical outputs. Tone mapping and bloom must not change reported metrics.

## Phase 5 — Connectome controller and learning

**Goal:** demonstrate actual biological graph participation in a learned control loop.
**Entry:** T03 complete; integrate with the scene after Phase 4.

### Phase deliverables and exit gate

- [ ] Deliver pinned connectome provenance, loading pipeline, and documented sensory/output mapping.
- [ ] Deliver trained readout checkpoints and fixed-seed evaluation curves.
- [ ] Run matched shuffled/random/bypass ablations and report results without assuming biological superiority.
- [ ] Verify state-to-action influence and absence of hidden-state leakage; synthetic-only implementations do not complete this phase.

### Implementation TODOs

### T07 — Biological controller and learning [P1; T03]

- [ ] Verify availability/version/license of MaleCNS source independently; original source claims are not yet audited.
- [ ] Implement rate reservoir first with interface permitting LIF; load real graph when available, label temporary synthetic fixtures honestly.
- [ ] Document sensory/output population mapping, signs/normalization, stability, and graph-size performance.
- [ ] Train small readout through sensory bottleneck; save checkpoints and fixed evaluations.
- [ ] Matched edge-shuffle/random/bypass baselines; no hidden actuator errors passed to learner.

Done when: evidence demonstrates controller-dependent improvement and connectome-state influence, with provenance and ablations. Do not label SPGD replay as learned fly control.

## Phase 6 — Full-control 3D tracking

**Goal:** extend validated learning from phase locking to steering and finite-range focusing.
**Entry:** Phases 3 and 5 complete.

### Phase deliverables and exit gate

- [ ] Complete and evaluate each curriculum stage in T08 separately.
- [ ] Deliver finite-range moving-target episodes and disturbance/reacquisition results.
- [ ] Document enabled action dimensions and all analytic assistance for every run.
- [ ] Report pointing, focus/range, PIB, and actuator-limit outcomes against declared targets; disclose unmet goals.

### Implementation TODOs

### T08 — Full controls and 3D target curriculum [P2; T05,T07]

- [ ] Static phase lock → fixed-plane steering → moving plane target → range/focus motion → full 3D fly tracking → disturbances.
- [ ] Expand action space deliberately: 19 piston channels initially; optional full 19×5 = 95 scalar commands (phase, amplitude, tip, tilt, curvature).
- [ ] Log analytic assistance explicitly; compare assisted and unassisted modes.
- [ ] Reacquisition, angular error, finite-plane PIB, range/focus performance, actuator-limit cases.

## Phase 7 — Fly assets and visual polish

**Goal:** integrate attributable fly models and readable presentation around working behavior.
**Entry:** T04 complete for asset integration; final polish follows Phase 6.

### Phase deliverables and exit gate

- [ ] Deliver converted articulated fly asset, reproducible conversion recipe, and attribution manifest.
- [ ] Integrate learner and moving target instances with consistent world scale.
- [ ] Deliver camera/lighting presets and verify controls, wires, sections, and legends remain readable.
- [ ] Recheck performance after asset integration and record any resolution or mesh-budget changes.

### Implementation TODOs

### T09 — Fly assets and scene polish [P2; T04; reference agent support]

Asset preparation and collection progress is tracked in [ASSET_TASKS.md](ASSET_TASKS.md). Source fly meshes, an art concept, exact channel geometry, and material defaults are available; GLB conversion and runtime integration remain TODO.

- [ ] Follow `RENDERING_REFERENCES.md`; import source transforms/joint hierarchy and convert to GLB reproducibly.
- [ ] Asset manifest with upstream revision, license/attribution, conversion recipe, units, mesh counts, and runtime budget.
- [ ] Learner and target instances, simple articulated motion; biological locomotion training is unnecessary.
- [ ] Lighting/camera presets preserve readable beam sections, wires, and channel selection.

## Phase 8 — Evaluation, videos, and delivery

**Goal:** produce reproducible final artifacts and a usable handoff.
**Entry:** Phases 4, 6, and 7 complete.

### Phase deliverables and exit gate

- [ ] Deliver documented train/evaluate/render/benchmark/make-video commands.
- [ ] Deliver checkpoint-matched timelapse and final tracking video with run manifests.
- [ ] Complete fresh-install validation and link reports, plots, checkpoints, licenses, and limitations.
- [ ] Audit the original acceptance criteria plus the newer rendering spec; leave unmet items explicit.
- [ ] Update every phase status and provide final implementation and independent-review summaries.

### Implementation TODOs

### T10 — Reproducible demos and delivery [P2; T06,T08,T09]

- [ ] Train/evaluate/render/benchmark/make-video CLI equivalents documented.
- [ ] Same evaluation trajectory across checkpoints; 20–60 s training timelapse and polished tracking video.
- [ ] Run manifests, checkpoints, plots, capture scripts, licenses, limitations, and fresh-install verification.

## Review ledger

| Date | Scope | Finding | Evidence / next action |
|---|---|---|---|
| 2026-09-15 | Repository baseline | Specification only; no implementation, dependencies, tests, or executable rendering | File inventory contained `DevLog/000-initial-discussion.md`; begin T01 |
| 2026-09-15 | Design review | Original far-field piston-only model cannot alone represent individual focus and finite-range propagation | New specification separates launch actuators, complex propagation, angular dome, and finite-range sections |
| 2026-09-15 | Coding agent, T01–T08 core | Implemented + tested: geometry, independent optical reference + fast evaluator (9 numerical checks), SPGD/random/analytic, bench/schematic/dome/section renderer, ES-trained reservoir readout, static→fly curriculum. | `DevLog/003-implementation-report.md`; `npm test` 26 passed; `npx playwright test` 4 passed; `npx tsc --noEmit` clean. Independent review pending. |
| 2026-09-15 | Physics finding | Piston-only steering aliases at pitch 0.5 mm / λ 1550 nm (grating spacing 0.177° vs element envelope 0.157°); implemented steering as common tip/tilt + piston phase-lock. | Requires reviewer confirmation of the intended V1 regime (report finding 1). |
| 2026-09-15 | Coding agent, T07 real graph | Typed-array MaleCNS loader integrated (`malecns.ts`); trained on 3000-node subset, reward 0.134→0.875, Strehl 0.709/0.699 vs random 0.055. Transmitter signs still placeholder all-positive. | `DevLog/evidence/t07-malecns-phase-lock-*.json`; test subset 164,606 nodes / 25.56M edges available. No topological advantage claimed. |
| 2026-09-15 | Coding agent, T09 partial | Articulated FlyBody GLB loaded for learner + target with illustrative wing clip; browser test passes. | `test-results/app-flybody-loaded.png`; two-instance perf/resting pose remain open. |
| 2026-09-15 | Open blocker | T10 rendered timelapse/final video not produced; `make-video` CLI equivalent not documented/validated. | Needs capture-pipeline decision (Playwright `recordVideo` vs offline frame render). |

## Completion report template

Copy this block into the review ledger or a linked phase report at each phase exit:

- Phase / task IDs:
- Status: TODO / IN PROGRESS / BLOCKED / IMPLEMENTED AND TESTED
- Independent review: PENDING / PASSED / CHANGES REQUESTED
- Completed TODOs:
- Changed paths:
- Exact validation commands and results:
- Numerical errors and convergence evidence:
- Screenshots / saved field arrays / report paths:
- Performance environment and measurements:
- Known limitations and remaining unchecked items:
- Blockers and attempted resolutions:
- Next phase / next concrete TODO:

Reviewer records an independent outcome only after examining implementation and evidence.


## Asset/data handoff update — 2026-09-15

T07 now has actual MaleCNS v1.0 resources cached locally; use [data/README.md](../data/README.md) and its indexed binary format/selection rule. Availability audit is complete for the downloaded resources. Integration, neuron population mapping, sign/normalization decisions, stability, training and ablations remain TODO. The current graph loader's comments may still say real data are unavailable; update those only as the actual loader/integration changes land. Reject unknown endpoints rather than mapping them to neuron index zero. Do not claim degree-based scaling guarantees a target spectral radius.

T09 has a browser-loadable articulated GLB, joint metadata, illustrative animation, and browser/source-render evidence. See [asset task update](ASSET_TASKS.md). T04 may use the five prototype bench GLBs or their generator; runtime actuation and wiring still need integration. These deliveries do not mark the containing implementation phases complete.

Progress snapshot and evidence: [DevLog 002 — assets and resource cache](../DevLog/002-assets-and-resource-cache-progress.md).

## Training/scene review update — 2026-09-15

See [DevLog 004](../DevLog/004-training-review-and-live-neural-inspector.md): completed 5,000-node MaleCNS training run, saved-spec evaluation repair, real before/after browser playback, and soma-position 3D activation driven by the active controller. Prior statements that the scene only supports synthetic activity are historical. Biological signs/mapping, true intermediate-checkpoint timelapse, and remaining optics/rendering acceptance checks are still TODO. Always inspect port ownership before starting a server; reuse the existing project server when available.

## Hardware realism and controlling-fly implementation supplement

The [hardware scene handoff](HARDWARE_SCENE_TASKS.md) specifies phases H0–H6 with TODOs and gates for enclosed modules, FC/RF connectors, routed 19-channel wiring, compact console, motorized aperture and staged fly foreleg gestures. Use its named-port asset manifest and wiring plan. It supplements existing optics/training acceptance checks; the assets' existence does not mark those checks or scene integration complete. See [DevLog 005](../DevLog/005-hardware-assets-and-coding-handoff.md) for collection/generation status.
