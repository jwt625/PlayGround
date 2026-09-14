# 009 — Delivered foundation and verification record

Date: 2026-09-13. Status: completed for this implementation pass.

## Delivered

- Preserved original `000-initial-discussion.md`; added numbered execution logs in coding, knowledge-base and game-design tracks with dependency/acceptance tables.
- Inspected SLM-Guessr source and mapped its card/gallery/quiz patterns into UI004. No frontend has been built in this pass.
- Added 18 versioned presets with source URLs, explicit units, nominal configurations, educational sampling envelopes and approximation/eligibility labels.
- Added all 20 required canonical example definitions. Six analytic anchors have computed answers; 13 numerical-interface recipes await a vector solver; one parity fixture awaits complex-field construction.
- Implemented axis-aligned Gaussian size/offset/tilt/linear-polarization overlap, complex scalar field reconstruction and explicit infinite-loss serialization.
- Implemented uint32 seeded RNG, triangular preset sampling, cross-geometry checks and a balanced alignment diagnostic bank.
- Implemented canonical SHA-256 keys plus static cache generation, warm reuse, corruption repair, manifest and audit command. No third-party runtime dependencies.

## Verification actually run

Environment: local Node v23.7.0. Commands from project root:

```sh
npm test
npm run cache:build
npm run cache:audit
npm run cache:build
```

Nine tests passed. Physics tests include a separate 2D complex midpoint quadrature calculation for combined elliptical size mismatch, translation, tilt and polarization, with |Δη|<1e−8. RNG checks cover 1,000 seeds for each of 18 presets; these check parameter bounds and geometric constraints, not mode guidance.

Cold build: 146 ready analytic records + 14 pending recipes, 148 written files including manifest/recipe bank. Warm build: 148 verified hits, zero writes. Audit recomputes every analytic η and verifies hashes/buckets. Corruption test changes one cached record; rebuild repairs only that record. Cache total at this version: 193,596 bytes (~189 KiB).

| η bucket | Cached ready count |
|---|---:|
| [0,.1) | 21 |
| [.1,.3) | 20 |
| [.3,.6) | 20 |
| [.6,.8) | 23 |
| [.8,.95) | 20 |
| [.95,.99) | 21 |
| [.99,1] | 21 |

The 140 generated questions are alignment exercises (20 per bucket); six canonical anchors add size, tilt and polarization. This is a coverage/physics bank, not yet a fully mixed 20-question game scheduler.

| Canonical example | η | Mismatch dB |
|---|---:|---:|
| Identical | 1 | 0 |
| Circular 2× waist ratio | .640000 | 1.938200 |
| 1 µm offset, MFD 4 µm | .778801 | 1.085736 |
| 1° air tilt, MFD 10.4 µm, λ=1.55 µm | .966732 | .146939 |
| Orthogonal linear polarization | 0 | ∞ (JSON null + explicit flag) |
| Elliptical size mismatch + x/y shift | .701376 | 1.540490 |

## Remaining work / recommended next execution

C2: build the usable Svelte speedrun/gallery against the shipped analytic bank and G003 score/reveal contract. C1.4: add IndexedDB with real-browser persistence/quota tests alongside UI work. C3: implement and externally validate full-vector FDE before converting numerical recipes into playable cached answers. K5.4: measure mode size and confinement over proposed guide envelopes and tighten RNG using acceptance rates.

No guide neff, guide MFD, vector η, taper transmission or browser eigensolve performance has been fabricated or claimed. Guide ranges are plausible candidate design envelopes requiring simulation, not production-qualified process windows. Numerical default material dispersion functions, finite BOX/handle stacks and tensor benchmarks remain research/implementation tasks in K002/K005.
