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
