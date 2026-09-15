# Fruit-Fly CBC — neural controller for a coherent beam-combining array

An entertainment/simulation project: a sparse, biologically motivated recurrent
substrate (a reservoir) receives a sensory bottleneck from a simulated optical
scene and drives a tiled coherent beam-combining (CBC), phased-array laser. The
fly "learns" to phase-lock and steer the beam; a second virtual fly then becomes
a moving target to track.

**Simulation and art only. There is no interface to, or control code for, any
physical laser system. The laser, targets, and connectome dynamics are virtual.**

## What is biological vs engineered

Read this before citing anything:

1. **MaleCNS connectome** — actual MaleCNS v1.0 data are cached locally
   (`data/cache/`, CC BY 4.0, Git-ignored) and loaded by
   `src/connectome/malecns.ts` as a typed-array derived graph (164,606 nodes,
   25,558,671 connection rows). Training uses a deterministic **subset**;
   transmitter-based excitatory/inhibitory signs are **not yet applied**
   (all-positive placeholder, renormalized). Synthetic graphs remain the default
   when no cache is requested. See `data/README.md`.
2. **Fly body model** — the converted articulated FlyBody GLB
   (`assets/generated/flybody/flybody-articulated.glb`, Apache-2.0 upstream) is
   loaded for the learner and target instances with an illustrative wing clip.
   Not a learned locomotion policy.
3. **Neural dynamics** — our approximation: a leaky rate reservoir
   `x[t+1] = (1-a)x[t] + a tanh(Wx[t] + W_in I[t] + b) + noise`, with a LIF-ready
   interface. Not a biophysical simulation.
4. **Sensory encoding** — a declared 8-channel bottleneck derived from the optical
   scene (target/beam directions, error, peak, PIB), not a compound-eye model.
5. **Trainable readout** — our own small linear/tanh `W_out` trained by evolution
   strategies. The connectome/reservoir graph is fixed.
6. **CBC simulator** — our scalar/paraxial model with hidden actuator errors.
7. **Reward learning** — our ES objective on the optical far field.

Connector data provide **connectivity, not a pretrained task policy**. Google/Janelia
did not release a fruit fly capable of arbitrary RL tasks.

## Architecture

```
src/
  optics/       geometry, launch apertures, angular-spectrum + Fresnel reference,
                fast Gaussian evaluator, far field, metrics
  controllers/  analytic reference, random baseline, SPGD, objectives
  connectome/   sparse graph, connectome loader, rate reservoir
  learning/     readout, evolution strategies, training loop
  sim/          environment (observations/reward), target paths, episode runner,
                reservoir controller, CLI runner
  renderer/     Three.js bench, far-field dome, measured section, schematic
  analysis/     dependency-free SVG plots
  cli.ts        train / evaluate / benchmark
  main.ts       browser app wiring
tests/          vitest numerical tests + Playwright browser tests
```

The simulation core never imports renderer/DOM code, so training stays headless.

## Commands

```bash
npm install
npm test                       # numerical + learning tests (vitest)
npm run typecheck              # tsc --noEmit
npm run build                  # typecheck + vite build
npm run dev                    # browser app
npm run test:browser           # Playwright interaction tests + screenshots
npm run train -- --task phase_lock --generations 40 --out outputs/phase_lock
npm run train -- --task tracking --generations 20 --out outputs/tracking
npm run train -- --task phase_lock --malecns 3000 --maxEdges 200000 --out outputs/malecns-phase-lock
npm run evaluate -- --run outputs/phase_lock
npm run benchmark              # writes outputs/benchmark.json
npm run make-video             # records outputs/videos/*.webm via Playwright
```

Training writes a run manifest (seed, config, synthetic-vs-provided provenance),
learning history, SVG learning/tracking plots, learned readout weights, and an
evaluation report comparing `sameArray` vs `transferToNewArrays` and a random
baseline.

## Current results (see `DevLog/002-implementation-report.md`)

- SPGD baseline: Strehl 0.88–0.95 across 10 fixed seeds (> 0.8 goal).
- Reservoir + ES readout phase lock: reward 0.15 → 0.98; normalized Strehl 0.81
  on the training realization, 0.73 on unseen realizations, random 0.055.
- Same controller on a **real MaleCNS v1.0** 3,000-node subset: reward 0.134 →
  0.875; normalized Strehl 0.709 training / 0.699 transfer, random 0.055.
- Tracking (Lissajous): reward 0.52 → 0.89; Strehl 0.73 / 0.60.
- Ablations (real vs edge-shuffled topology) are currently within noise; **no
  connectome advantage is claimed.**
- MaleCNS subset (5,000 nodes / 22,427 edges): load 163 ms, 6,961 reservoir
  steps/s, 693 MB RSS on an M4 Pro.

## Physics notes

- 19 hex-packed channels, rows 3/4/5/4/3, SI units, forward +Z.
- Steering is common tip/tilt plus a piston ramp; the learned controller adds
  piston corrections on top. Piston-only steering aliases at the declared
  `pitch = 0.5 mm` / `lambda = 1550 nm` fixture (see the report), which is why
  tip/tilt carries the steering.
- Aperture overlap and reference-model validity domains are documented in
  `src/optics/launch.ts` and `src/optics/reference.ts`.

## Attribution / references

- Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila
  male central nervous system," Cell, 2026 (connectivity dataset; not used yet).
- Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion,"
  Nature 643, 1312–1320 (2025); `TuragaLab/flybody` (planned asset).
- SPGD and LOCSET CBC references, and OPA calibration, are catalogued in
  `DevLog/000-initial-discussion.md` section 8–9.

## License / provenance

Project code is original to this repository. Third-party assets are not yet
bundled; any future FlyBody import must retain its license, revision, and
attribution per `docs/RENDERING_REFERENCES.md`.
