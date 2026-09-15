# 002 — Implementation report (coding agent)

Date: 2026-09-15
Scope: T01–T08 core implemented and tested; T09/T10 partial. Simulation-only; no physical laser interface.

This entry records what was built, how it was validated, the physics-contract findings that need reviewer attention, and what remains open. Checkboxes in [`docs/CODING_TASKS.md`](../docs/CODING_TASKS.md) were updated to match.

## Summary

- **Stack as proposed:** TypeScript + Vite + Three.js browser client, a headless simulation core that never imports renderer/DOM code, and a `tsx` CLI. Vitest for unit/numerical tests, Playwright for browser interaction tests.
- **T01** scaffold, configs, 19-channel hex geometry with stable CH01–CH19 IDs, deterministic ordering, SI units.
- **T02** independent optical reference (sampled launch aperture, angular-spectrum propagation, direct paraxial Fresnel integral, FFT far field) plus a fast analytic Gaussian evaluator, with the nine required numerical tests.
- **T03** analytic reference controller, random baseline, and SPGD; SPGD recovers phase alignment across 10 fixed seeds.
- **T04** 3D seed/splitter/branch/modulator/emitter bench with phase rings and translucent beams, plus a shared-state 2D schematic and selected-channel controls.
- **T05** forward-hemisphere far-field dome (vertex-colored, weighted centroid + independent peak marker) and a movable measured transverse section sampled from the field evaluator.
- **T06** display scaling isolated from physics, log/linear intensity colormaps, frame loop instrumentation (benchmark), browser screenshots.
- **T07** sparse reservoir with rate dynamics and a LIF-ready interface, real-connectome JSON loader (no bundled MaleCNS data), trained readout via evolution strategies, matched ablations.
- **T08** static → moving-target curriculum (`static`, `lissajous`, `circle`, `fly`) with tip/tilt steering + learned piston correction.
- **T09/T10** partial: assets are code-native (no FlyBody mesh yet); CLI, manifests, learning-curve/tracking SVG, and Playwright screenshots exist; no rendered video yet.

## Validation (exact commands and results)

Machine: macOS 15.5, Apple M4 Pro, Node v23.7.0. Serial tests.

| Command | Result |
|---|---|
| `npm test` | 26 passed (geometry 10, optics 10, controllers 4, learning 2) |
| `npx tsc --noEmit` | clean |
| `npm run build` | vite build ok (bundle ~529 kB) |
| `npx playwright test` | 3 passed (render, mode/channel selection, target options) |
| `npx tsx src/cli.ts train --task phase_lock --generations 40 --population 16 --steps 100 --reservoir 300` | reward 0.148 → 0.981; 640 objective evals |
| `npx tsx src/cli.ts train --task tracking --generations 20 --population 12 --steps 90 --reservoir 300` | reward 0.524 → 0.891; 240 evals |
| `npx tsx src/cli.ts benchmark` | 1.23 M fast-field samples/s; 6333 closed-loop steps/s; darwin arm64 |

Evidence files: `DevLog/evidence/t06-optics-benchmark.json`, `t07-phase-lock-{history,evaluation}.json`, `t08-tracking-{history,evaluation}.json`. Browser screenshots are under `test-results/` (gitignored): `app-analytic.png`, `app-connectome-selected.png`, `app-fly-target.png`.

SPGD Strehl over 10 fixed seeds: 0.936, 0.954, 0.949, 0.914, 0.931, 0.930, 0.890, 0.881, 0.944, 0.938 (8/10 > 0.9). This exceeds the original > 0.8 goal for the static phase-lock fixture.

## Physics-contract findings (reviewer action requested)

1. **Piston-only steering aliases badly at the declared fixture.** With `pitch = 0.5 mm`, `lambda = 1550 nm`, adjacent-emitter phase for a piston ramp wraps every `lambda/pitch ~ 0.177 deg`, and the Gaussian element envelope half-angle is `lambda/(pi w0) ~ 0.157 deg`. Grating lobes therefore sit just outside the element envelope and piston-only steering is effectively limited to a sub-0.2 deg field. **Resolution used:** steering is by common tip/tilt (all channels pointed at the target) with a piston ramp only to phase them at that direction; the connectome learns piston corrections. This is physically appropriate for a tiled fibre array and matches the spec's "tip/tilt and focus" actuator set. Reviewer should confirm this is the intended V1 regime.
2. **Coherent overlap inflates the launch-plane power integral.** With `w0 = 0.18 mm` and `pitch = 0.5 mm`, in-phase channels overlap slightly, so `integral |sum U|^2` exceeds `sum P_n` by ~9% (constructive interference, not a gain). Tests report this explicitly. PIB uses the declared radiated power `sum a_n^2` as denominator via a bucket integral; Strehl uses the same-power, same-geometry ideal ramp.
3. **Angular-spectrum sampling limit documented:** alias-free propagation requires `z <= n dx^2 / lambda` (`angularSpectrumCriticalZ`). Long-range finite-target checks use the direct Fresnel integral instead. The default dome/objectives use the analytic array factor (far field), which is the intended domain.
4. **PIB bucket** is a cone of half-angle `1e-3 rad` about the target, integrated with solid-angle weights and divided by emitted power. It is now a live, nonzero metric (early connectome reads ~0.68).

## Learning results and honest limitations

- The readout trains against a **single hidden-error realization** by default. Evaluation now reports `sameArray` (training realization) and `transferToNewArrays` (unseen realizations) separately when both are run.
- Phase lock: `sameArray` normalized Strehl 0.81, `transfer` 0.73, random baseline 0.055.
- Tracking: `sameArray` 0.73, `transfer` 0.60.
- **Ablation does not yet show a connectome advantage.** Edge-shuffled versus real topology differences are within run-to-run noise (e.g. real 0.84 vs shuffled 0.92 in one test batch). The reservoir currently provides a basis for a mostly static correction; it is not yet solving per-channel identification from beam-level observations. This is the central scientific gap and is stated rather than oversold.
- No MaleCNS graph is bundled. `connectomeFromJson` accepts an audited file; `manifest.json` records `connectomeSource`. T07's availability/version/license audit is still outstanding.

## Still open / TBD for the planning agent

1. **Steering regime:** confirm tip/tilt steering + piston phase-lock (finding 1) versus a smaller pitch that makes piston-only steering valid.
2. **Generalization target:** should the readout be trained across hidden-error realizations (needs a richer sensory channel or an online optimizer) or is per-array calibration the intended deliverable?
3. **Connectome provenance:** which MaleCNS release/format is authorized, where the file lives, and its license.
4. **Reward weights:** current defaults `target 1.0, pib 0.25, pointing 0.1, energy 0.001` are provisional.
5. **Video scope:** T10 timelapse/final video not implemented; needs a decision on capture pipeline (Playwright `recordVideo` vs offline frame rendering).

## Verification needed (delegated, not claimed)

- Visual: selecting CH01/CH10/CH19 highlights the full path in both the 3D bench and schematic; section samples equal direct field queries; dome centroid can fall between lobes and is never peak-snapped.
- Device: keyboard/trackpad camera interaction on a real machine (browser tests use synthetic events).
- Visual/complex: beam envelopes follow actual axes; interference overlays derive from the total complex field, not per-beam animation.

---

## Update 2 (2026-09-15): real MaleCNS v1.0 integration and fly asset

The reviewing agent cached actual MaleCNS v1.0 resources (`data/README.md`) and a
converted articulated FlyBody GLB (`assets/README.md`) while this report was in
progress. Both were integrated and validated:

### Real connectome (T07)

- `src/connectome/malecns.ts` loads `derived/body-ids.npy` (int64) and
  `derived/edges.u32` (headerless uint32 triples) with typed arrays. It samples a
  deterministic subset, preserves ascending original body IDs, and rejects
  out-of-range endpoints instead of mapping them to index zero.
- Explicit assumptions recorded in code and manifest: raw synapse counts are used
  as magnitudes; transmitter-based excitatory/inhibitory signs are **not** yet
  applied (parsing `neurotransmitters.feather` needs an Arrow reader), so the
  reservoir uses an all-positive placeholder sign and renormalizes. This is
  labelled, not presented as a biological sign assignment. Degree-based scaling
  does not guarantee a spectral radius; it is a practical normalization.
- An ablation harness, matched edge-shuffle, and bypass remain available. No
  topological advantage is claimed.

Validation:

- `npx vitest run tests/malecns.test.ts` — pass (deterministic subset, IDs
  ascending, indices in range; 164,606 nodes and 25,558,671 edges available).
- `npx tsx src/cli.ts train --task phase_lock --malecns 3000 --maxEdges 200000 --generations 20 --population 12 --steps 80 --out outputs/malecns-phase-lock`:
  reward 0.134 → 0.875; `sameArray` Strehl 0.709, `transfer` 0.699, random 0.055.
  Manifest `connectomeSource = malecns-v1.0-cache-subset`.
- `npx tsx src/cli.ts benchmark`: MaleCNS subset 5,000 nodes / 22,427 edges,
  load 163 ms, reservoir 6,961 steps/s, process RSS 693 MB (macOS arm64, Node
  v23.7.0). Evidence `DevLog/evidence/t06-optics-benchmark.json`.

### Fly asset (T09, partial)

- `src/renderer/flyActors.ts` loads
  `assets/generated/flybody/flybody-articulated.glb` with `GLTFLoader`,
  clones learner and target via `SkeletonUtils`, scales to display units, and
  runs the illustrative wing clip. A procedural placeholder is shown until load,
  so the simulation never depends on the asset.
- Validated in headless Chromium: `npx playwright test` (4 passed), including a
  test that waits for `flyLoaded` and asserts no load error. Screenshot
  `test-results/app-flybody-loaded.png`.
- Outstanding: two-instance performance with the full-resolution asset, resting
  pose, and matched-camera source comparisons (owned by the asset workstream).

### Updated still-open items

- Transmitter sign assignment and superclass-based sensory/output population
  mapping for the real graph.
- Whether a real-graph topology advantage survives matched ablations at larger
  subsets and longer training.
- T10 rendered video: not yet produced.

### T10 demo capture (partial)

- `npm run make-video` builds the client, starts a preview server, and records a
  ~16 s webm across analytic, SPGD, fly-target, and connectome modes. Produced
  `outputs/videos/bddb8587e8e305e1777fdb5d30fbbbdc.webm` (2.8 MB). `outputs/` is
  Git-ignored; the command is reproducible.
- Still open: checkpoint-matched **training timelapse**, and an muxed
  mp4/polished final cut. The current clip is a single continuous capture, not a
  curriculum timelapse.

---

## Update 3 (2026-09-15): inspector discoverability, fly roles, section span

User-reported issues fixed:

1. **"Inspect 3D neurons" did nothing unless a run was already loaded.** The
   button only moved the camera; with no `NeuralActivityView` it focused empty
   space. It now self-loads the saved MaleCNS run (and its mapped somata) before
   focusing, so it works as a standalone entry point. Regression test:
   `Inspect 3D neurons loads the run standalone and focuses the mapped somata`.
2. **Both flies flapped and the target looked stationary.** `FlyActors` gave an
   animation mixer to both instances. Now only the **target** carries the
   illustrative wing clip; the learner (neural controller) keeps the GLB rest
   pose and stays fixed beside the console. The target is repositioned on the
   displayed target plane using the same transverse scale as the beam envelopes
   (`DISPLAY.scale`) instead of the previous tiny angular offset, and yaws to
   face its direction of travel. Regression test: `target fly moves across the
   scene while the learner stays fixed`.
3. **Intensity section span increased 10× in both in-plane directions** (3 mm →
   30 mm, `samples` 72 → 160). Mesh opacity dropped to 0.55 so the enlarged
   measured plane does not occlude the array/flies; the `Section` layer toggle
   still hides it.

Validation: `npx tsc --noEmit` clean; `npm test` 33 passed; browser tests 8 passed
(`CBC_INSPECTOR_URL=http://127.0.0.1:5173 npx playwright test`) against the
existing dev server. Screenshots: `test-results/fix2-fly-b.png`,
`test-results/fix-neurons.png`, `test-results/inspect-initial.png`.

---

## Update 4 (2026-09-15): always-on neurons, label, and beam/plane alignment

1. **3D neuron cloud is always on.** The saved MaleCNS run auto-loads at startup
   (`autoLoadRealRun`), builds `NeuralActivityView`, and drives the scene with
   the trained ("after") policy in connectome mode. The reservoir is also stepped
   every frame outside connectome mode, so the cloud stays active in
   analytic/manual/SPGD too. Default target is now the moving **fly**. The
   Before/After buttons and the manual "Load real training run" button still work;
   reloading removes any previous neural group first.
2. **Simplified the 3D label.** Removed the "Activity from the controller · lines
   are sampled graph edges" line; the sprite now reads only
   `MaleCNS · N mapped somata`.
3. **Beam / measurement-plane alignment fixed.** The section plane's display
   center used a single uniform scale (`n * targetDistance * range`) while the
   beam envelopes in `bench.ts` use anisotropic scaling (transverse
   `DISPLAY.scale`, axial `DISPLAY.targetDistance`); that ~60× mismatch put the
   plane off the beam intersection. The section center now uses the same mapping.
   The residual visible offset came from a coarse (~7 mrad) centroid scan, so
   `Environment.scanBeam` now does a second fine stage around the coarse centroid;
   pointing error dropped from ~2,900 µrad to ~7–60 µrad and the plane centering
   is sub-mrad. Locked by `tests/sectionAlignment.test.ts`.

Validation: `npx tsc --noEmit` clean; `npm test` 34 passed; browser tests 8 passed
(`CBC_INSPECTOR_URL=http://127.0.0.1:5173 npx playwright test`). Screenshots:
`test-results/align2-target-analytic.png`, `test-results/align-overview.png`.

---

## Update 5 (2026-09-15): measurement plane fixed; hardware scene H0 + H1/H2 first pass

- **Measurement plane fixed at boresight.** `main.ts` now passes the nominal
  `+Z` direction to `MeasurementSection` instead of the moving beam centroid, so
  the intensity plot no longer sweeps around. Sparsely, the fine centroid
  refinement from Update 4 remains for telemetry and the dome/peak markers.
- **H0 done.** Typed hardware registry (`src/renderer/hardware/registry.ts`) from
  the generated manifest + wiring plan, with validation
  (`tests/hardwareRegistry.test.ts`): nine components, 19 unique channel
  mappings, CH01 optical path, connector-kind compatibility, no vendor-STEP
  import, display/solver separation.
- **H1/H2 first pass.** `src/renderer/hardware/hardwareScene.ts` loads the nine
  hardware-v2 GLBs via `import.meta.glob` (so builds emit them), places one
  instance per registry node (100 modules incl. placeholders), and routes 195
  sampled cables from `wiring-plan.json`; opt-in **Hardware bench (v2)** toggle.
  Browser test `hardware bench loads on demand and routes cables` passes.
- **Still open:** H1 mated connectors/labels/clearances, H2 tube cables + bend
  checks + per-channel tracing gate, H3 console binding, H4 foreleg gestures,
  H5 mechanical tip/tilt + focus stage, H6 evidence. The current hardware layer
  is an illustrative first pass, not the H1/H2 acceptance result.

---

## Update 6 (2026-09-15): all hardware-v2 assets incorporated

The previous hardware pass loaded the assets but placed them small and off to
the side, and one component name was wrong (`splitter` vs `splitter-19`), so only
placeholders were obvious. Fixed and expanded:

- All nine GLBs are instantiated and visible by default: splitter-19,
  fly-console, phase-cassette x19, optical-amplifier x19, phase-driver x19,
  tiptilt-collimator x19, plus fc-apc-plug, sma-plug, fc-bulkhead connectors.
  Metrics: **78 asset modules, 172 mated connectors, 9/9 asset kinds, 0 failures,
  215 routed cables**.
- Connectors are mated as children of the GLB port nodes (plugs extend outward
  along the declared port normal; bulkhead feedthroughs on splitter outputs), so
  they follow tip/tilt motion.
- Bench lowered into the default camera view; a `hardware` camera preset frames
  the exploded rack. Channel-ID sprite labels mark the aperture collimators.
- Nested tip/tilt pivots are driven from actual pointing (display gain 25,
  documented as presentation scaling).
- Browser test now asserts 9 component kinds, >70 modules, >100 connectors, and
  zero load failures.

Validation: `npx tsc --noEmit` clean; `npm test` 39 passed; browser 9 passed.
Screenshots: `test-results/hardware-default.png`, `test-results/hardware-bench2.png`.
Remaining: H1 connector closeups/clearances, H2 tube cables + bend checks +
CH01/CH10/CH19 tracing gate, H3 console binding, H4 foreleg gestures, H5 focus
stage, H6 evidence.

---

## Update 7 (2026-09-15): CBC assembly spec Pass 1 + first half of Pass 2

Implemented the concrete assembly from `docs/CBC_ASSEMBLY_SPEC.md` (replacing
the old approximate rack):

- `src/renderer/hardware/assemblySpec.ts` is the typed implementation artifact:
  mechanical-mm placements for TABLE, RACK-L/R, SEED, SPLIT (orientation B), PSU,
  PDU-M/S/R1-4, IO, MC, CON, OP-PLATFORM, AP-FRAME; 19 cell assignments;
  canonical 3+4+5+4+3 aperture sites from `channel-layout.json` x 65 mm; port
  schedules with socket / pigtail / free-space typing; and the full cable
  schedule: **241 internal runs** (58 optical + 19 RF + 40 command + 48 DC +
  76 motor) plus one external supply cord.
- `hardwareScene.ts` rebuilt to the spec: one uniform mm->scene conversion that
  **registers the aperture emission center to the optical array origin**, so the
  solver beams leave the real hex aperture; horizontal table/racks/trays,
  vertical aperture frame with mount ledges; phase cassettes/drivers/amplifiers/
  splitter/collimators/console reuse the GLBs with A/B/C/world transforms; other
  instruments and supports are procedural boxes with named port anchors.
  Cables route with straight terminal leads plus family lanes (F-L, E-R,
  AP-MOTOR) instead of a universal midpoint arch. GLB socket ports receive one
  mating plug; pigtail and free-space ports are not given sockets.
- Runtime: 112 placed equipment, 78 asset instances, 115 connectors, 8 GLB kinds,
  242 cables, 0 failures.
- Tests: `tests/assemblySpec.test.ts` audits the 241-run inventory, canonical
  3+4+5+4+3 aperture with CH10 at (0,300,100), 78 FC sockets/plugs, endpoint
  resolution, and pigtail/free-space typing. `npm test` 44 passed; browser 9
  passed.

Still open per `docs/CODING_AGENT_NEXT_PASS.md`: Pass 1 orientation-arrow
evidence and fly-in-front composition; Pass 2 CH10 closeups, tube geometry with
sampled-curvature/clearance checks, service loops, and the CH01/CH10/CH19
tracing gate; Pass 3 the 43-knob console and articulated foreleg gestures
(H4/H5). Counts are implemented and audited; visual/geometric acceptance is not
claimed.

---

## Update 8 (2026-09-15): Pass 2 cables + Pass 3 console/fly (partial gates)

Pass 2 — terminal-aware cabling:
- Plugs now belong to cable assemblies (`terminal()` in `hardwareScene.ts`): a
  male FC/APC or SMA plug is added only at socket ends and the cable starts at
  its downstream `port_cable_exit`; captive PM pigtails start at the pigtail exit
  with no plug. 116 plugs total (78 FC + 38 SMA), matching §2.3.
- Family lanes (F-L, E-R, AP-MOTOR) plus an Ω service loop behind each moving
  collimator; selected channel re-renders as tube geometry (6 radial segments),
  other channels stay as lines.
- `auditRouting()` reports sampled curvature radius and inflated clearance.
  **Result: min sampled radius 2.67 mm and ~13.9k sphere-clearance hits — the R3
  gate (30 mm optical, clean clearance) is NOT met.** The routing is
  terminal-tangent and lane-based but still turns too sharply; this is reported,
  not claimed as passing. Exposed as `hardwareMinRadiusMm`/`hardwareClearanceViolations`.

Pass 3 — console and operator fly:
- `consolePanel.ts` builds the §8.2 panel: 19 phase/amplitude pairs + five larger
  selected-channel knobs = **43 knobs**, a 19-button canonical hex selector, and
  a status display, bound every frame to the actual per-channel command state
  (`updateConsole`). Verified 43/19 in the browser.
- The controlling fly is placed as the console operator
  (`FlyActors.setOperatorMode`, registered to the console front, native +X
  forward → world -Z) and drives a staged foreleg reach/contact/retract on the
  six `femur/tibia/tarsus_T1_left/right` joints; wings remain at rest. A
  `console` camera preset was added.

Validation: `npx tsc --noEmit` clean; `npm test` 44 passed; browser 9 passed
(`hardwareKnobs=43`, `hardwareSelectors=19`, `flyForelegJoints=6`). Screenshots:
`test-results/assembly-pass1.png`, `test-results/pass3-console2.png`.

Not met / still open: R3 routing radius+clearance gate; Pass 1 orientation-arrow
evidence and a clean three-quarter bench with the fly in front; Pass 2 CH10
IN/OUT/channel labels and the CH01/CH10/CH19 attachment-under-motion gate; Pass 3
foreleg contact accuracy against actual `touch_*` targets and a recorded gesture
clip; H6 evidence bundle.

### Update 8a — routing-audit correction

After removing a hairpin detour in `laneFor` (lane points now use the cable's
start/end depths instead of a shared midpoint), the sampled minimum radius
measured **1.02 mm** (previously 2.67 mm) and sphere-clearance hits ~12.1k. The
curvature estimate uses a nonuniform finite-difference formula and the routing is
a display approximation, so neither number is a calibrated mechanical result —
but both confirm the R3 radius/clearance gate is still unmet. Kept as a reported
diagnostic (`hardwareMinRadiusMm`, `hardwareClearanceViolations`); not claimed as
passing.
