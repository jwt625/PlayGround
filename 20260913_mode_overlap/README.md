# Mode Overlap — development foundation

Plans and a reproducible example/cache foundation for a photonic mode-overlap intuition game.

Start with [DevLog execution index](DevLog/README.md). The [preset/RNG knowledge base](DevLog/knowledge-base/005-presets-and-randomization.md) explains nominal configurations, source evidence, educational ranges and validity limits. [Implementation results](DevLog/coding/009-foundation-results.md) records the C1 pass; [C3 solver plan](DevLog/coding/007-vector-solver-and-validation.md) records the 1D slab backend. The C2 UI follows the local SLM-Guessr reference; the 1D slab full-vector solver exists and is benchmarked against closed-form dispersion, while the 2D full-vector backend and its external benchmark are still pending.

```sh
# Node >=20; install dev dependencies once for the Svelte app
npm install
npm test          # node --test over physics, cache, scoring, session, mode store
npm run cache:build
npm run cache:audit
npm run cache:modes  # nominal waveguide-mode gallery (experimental scalar)
npm run check     # svelte-check
npm run dev       # SvelteKit dev server
npm run build     # static site into build/
```

- `data/presets.json`: 18 nominal fiber/guide/SSC/hybrid presets with bounded candidate RNG and sources.
- `data/examples.json`: all 20 mandatory examples from the initial brief.
- `src/physics/gaussian.mjs`: normalized scalar Gaussian overlap and complex field reconstruction.
- `src/physics/materials.mjs`, `mesh.mjs`: sourced isotropic indices with hard validity windows, and boundary-aligned 1D/2D meshes with priority rasterization.
- `src/physics/slab.mjs`, `overlap.mjs`, `solve-modes.mjs`: 1D slab full-vector FDE (TE/TM) validated against closed-form dispersion, plus common-grid flux overlap and the `solveModes` dispatcher.
- `src/physics/fem2d.mjs`, `projection.mjs`: sparse 2D scalar FEM (validated in the slab limit, not release-eligible) and selected-mode projection.
- `src/solver/worker-queue.mjs`, `provenance.mjs`: dedup/cancel/telemetry solve queue and warm-start provenance gate.
- `src/lib/adaptive.ts`: deterministic adaptive profile with a difficulty clamp that never selects unvalidated levels.
- `src/scenarios/generator.mjs`: deterministic sampling and balanced alignment bank.
- `src/storage/cache.mjs`: canonical content keys for questions and future modes; `src/storage/mode-store.mjs` adds the C1.4 IndexedDB store with memory fallback.
- `src/lib`: scoring/session/explanation core plus Svelte gallery and 20-question speedrun; `src/routes`: static SvelteKit routes.
- `static/cache/manifest.json`: 146 ready analytic examples and references to 14 pending recipes.

Level 2 of the speedrun mixes the shipped analytic anchors with freshly generated Gaussian variants (random aspect ratio and bounded misalignment) and waveguide mode questions built at runtime from the solved nominal modes (same-guide offset and cross-guide pairs). These waveguide questions are experimental scalar quasi-TE, not full-vector, and are labelled as such. Full-vector and anisotropic scenarios remain disabled until C3 supplies validated fields; the vector `Sz` control is explicitly disabled.

Rebuilds reuse verified identical entries and repair corrupt entries. To build into a separate directory: `node scripts/build-cache.mjs /tmp/mode-overlap-cache`. Default paths resolve from the script location. The audit command checks the project's shipped bank.

All lengths are µm unless explicitly named otherwise. Fiber MFD is the intensity 1/e² diameter in the Gaussian model. Ready answers are scalar paraxial Gaussian overlaps at a common waist plane; numerical recipes have no computed answer until validated. `lossDB: null` with `lossIsInfinite: true` encodes exact zero overlap. Mode-mismatch loss is not a claim of exact discontinuity insertion loss.
