# 004 — Training review, replay integration, and 3D neural inspection

Date: 2026-09-15
Status: completed bounded training experiment and before/after browser integration. Scientific/model limitations remain below.

## Why the earlier scene did not show training progress

The CLI trained a real MaleCNS subset, but the frontend initially ran an independent synthetic 240-node reservoir and defaulted to analytic control. Its only neural visualization was a 2D activity bar chart. Publishing a learning-curve SVG did not connect that trained controller to the scene. The completed offline run therefore did not visibly change the browser simulation.

This gap is now addressed with explicit **Before training / After training** playback. It is not a reconstruction of intermediate generations: only the final policy was saved, and initial readout weights are reproduced from the saved seed and dimensions.

## Inspected implementation and fixes

- Initial verification: 27 unit tests and TypeScript checking passed; a Three.js bench, schematic, dome, section, and FlyBody instances already existed.
- Fixed checkpoint evaluation rebuilding default configuration. New `run-spec.json` stores the effective training configuration; replay restores graph choice, dimensions, environment settings, and saved options. Legacy JSON-connectome checkpoints without recoverable graphs are rejected.
- CLI now saves per-generation progress and reports actual graph metadata on new runs. The original run log printed the inactive synthetic `n=600` default; the saved MaleCNS options and deterministic reconstruction confirm 5,000 active nodes.
- Scene inspection now includes camera presets, pause/resume, layer visibility, selected-channel controls, and section range.
- Manual offsets persist across channel selection instead of affecting only the currently selected emitter.
- Replaced centroid-directed decorative beam tubes with individual Gaussian envelopes derived from actual tip/tilt and curvature. Envelope transparency is illustrative; summed-field interference remains in measured sections.
- Reduced opaque dome shading and masked unsupported directions. The dome is still too coarse for quantitative narrow-lobe inspection.

## Completed training run

```sh
npm run train -- --task phase_lock --malecns 5000 --maxEdges 300000 \
  --generations 40 --population 16 --steps 120 \
  --out outputs/review-malecns-phase-lock-20260915
```

Actual induced subset: **5,000 neurons, 23,447 connections**. Evolution strategies train **1,235 readout parameters** (19 outputs × 64 features, plus 19 biases). Connectivity is fixed. The graph has provisional all-positive signs and generic sensory/output selection; it is not a fully annotated biological dynamical model.

| Result | Value |
|---|---:|
| Generations / candidate evaluations | 40 / 640 |
| Initial mean reward | 0.1363 |
| Final mean reward | 0.9725 |
| Training-array mean normalized target intensity | 0.7995 |
| Four held-out arrays, mean normalized target intensity | 0.7250 |
| Random-phase baseline reported by CLI | 0.0554 |

The CLI calls the intensity metric `meanStrehl`; it is the normalized intensity at the target direction, not a separately validated peak-search Strehl measurement. Training uses hidden-error realization 1; held-out evaluation uses 2–5. Steering remains analytic. This run does not establish a connectome advantage over matched baselines.

Artifacts: `outputs/review-malecns-phase-lock-20260915/` contains the log, effective specification, manifest, per-generation history, final readout, learning/tracking SVGs, evaluation, and review metadata. Source hashes in review metadata were recorded during review after some frontend/CLI changes, not at the original process launch.

## Exact implemented objective

Per simulation step:

```text
reward = normalized_target_intensity
       + 0.25 * PIB
       - 0.1 * angular_pointing_error_rad
       - 0.001 * mean_over_19_channels((piston_command_t - piston_command_previous)^2)
```

- Normalized target intensity: `|E(target)|² / (sum of enabled actual field amplitudes)²`.
- PIB: approximate angular power inside a 1 mrad target bucket, divided by the implementation's emitted-power reference.
- Fitness: mean reward across 120 steps; the training configuration averages over hidden-error seed `[1]`.
- Each candidate changes the linear readout and biases. Its output is passed through tanh and mapped to absolute piston corrections; common tip/tilt and base steering phases are supplied analytically.

Review caveats: the piston-change penalty uses unwrapped differences; the pointing scan is sparse; the PIB quadrature/power convention needs refinement. Reward can exceed one because it adds the PIB term. The displayed raw phase RMS does not remove global piston, so it can increase while optical combining improves. These diagnostics must not be overinterpreted.

## Browser before/after replay

- `scripts/export-training-scene.ts` reconstructs the actual training graph and exports the already-scaled weights, effective reservoir settings, source body IDs, initial/final readouts, and recorded learning history.
- `scripts/export-neuron-locations.py` joins original IDs to cached soma annotations in graph order.
- `public/training/scene-run.json` and `neuron-locations.json` supply the browser. They contain only the 5,000-node selected graph, not the full raw data cache.
- The Before/After buttons create fresh reservoirs with the same input/output mapping and noise initialization, install the selected readout, reset the same hidden-error seed, and replay the same 120-step phase-lock episode.
- Automatic pause at the end keeps results inspectable. Quarter-speed playback and single-step controls slow presentation without changing the per-step dynamics.
- Other target/controller modes remain available as exploration; they are not the fixed before/after evaluation.

Browser validation measured **0.06544 before vs 0.79948 after** mean normalized target intensity. The initial reward and intensity differ because reward also includes PIB and penalties. Saved evidence: `outputs/review-malecns-phase-lock-20260915/browser-replay.json` and `neural-replay.png`.

## 3D neural activation rendering

`src/renderer/neuralActivity.ts` renders:

- **4,268 real soma positions** from the 5,000 simulated neurons; missing-location neurons remain in dynamics but are omitted from point geometry.
- **1,781 sampled actual graph edges** between mapped points.
- Point color/brightness driven by the same live reservoir state that feeds the optical readout: cyan for positive, amber for negative, brightness for magnitude.
- Edge brightness driven by presynaptic activity. Lines represent connectivity between somata, not traced axonal paths or measured transmission delays.
- Clickable points exposing original body ID, cell type, and superclass.

A shared centering/scaling transform and display-axis rotation preserve relative soma positions. This is an adjacent inspection view, not anatomical registration into the fly mesh. It is distinct from the two previously cached DNge104 skeleton assets.

The present reservoir often approaches saturation. Visible activity and readout improvement do not demonstrate biological fidelity, useful transient computation, or superiority of the actual topology. Transmitter signs, sensory/output population mapping, normalization/stability, and matched real-graph ablations remain high-priority work.

## Ports and local inspection

User instruction: **always inspect occupied ports before starting servers**.

Port 5173 was occupied by this project's existing Vite process (PID 26803 at inspection). It was reused; no replacement server was started for replay work. `scripts/dev-server.mjs` now checks the requested port before launching Vite and exits with an explanatory message if occupied. It never kills another process or silently switches ports.

```sh
lsof -nP -iTCP:5173 -sTCP:LISTEN
# Reuse the existing project server at http://127.0.0.1:5173/
# If an explicitly chosen port is free:
npm run dev -- --port 5173
```

Open the scene, choose **Load real training run + 3D neurons**, then compare **Before training / After training**. **Inspect 3D neurons** changes the camera. The UI identifies the active graph and distinguishes endpoint replay from ongoing training.

## Reproduction and checks

```sh
npx tsx scripts/export-training-scene.ts outputs/review-malecns-phase-lock-20260915
npm test
npm run build
# Browser tests connect to the inspected existing server; no new web server:
CBC_INSPECTOR_URL=http://127.0.0.1:5173 npm run test:browser
node scripts/check-training-replay.mjs http://127.0.0.1:5173
```

Earlier scene fixes passed 31 unit tests and five browser tests. The added replay checks verify retained source IDs/weights, endpoint rejection, real graph selection, mapped soma count, nonzero controller activity, and before/after optical improvement. Final checks: **33 unit tests and six browser tests passed**. Build includes a known large-bundle notice and the 23 MB fly GLB.

## Next TODOs

- [ ] Save per-generation policy checkpoints and same-episode evaluations for a true training timelapse.
- [ ] Move browser training off the render thread and expose live run progress; the current browser training button is a separate synchronous demo and disabled after loading the real saved run.
- [ ] Annotated transmitter signs and sensory/descending population interfaces; measure stability/saturation before longer biological comparisons.
- [ ] Refine pointing and PIB integration; remove global piston in phase-quality diagnostics and wrap phase increments in smoothness penalties.
- [ ] Higher-resolution/adaptive angular rendering, longitudinal/manual sections, and final scene polish.
