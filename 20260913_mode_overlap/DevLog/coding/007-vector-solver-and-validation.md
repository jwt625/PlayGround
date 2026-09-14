# 007 — Vector solver implementation plan and release gates

Status: 1D slab full-vector backend implemented and benchmarked against closed-form TE/TM dispersion; 2D backend and worker planned. Dependencies K002, K005, C1. Date: 2026-09-13.

1. Implement versioned material functions with wavelength validity and region precedence; polygon/rib/trapezoid dimensions explicit. Start isotropic Si/SiO2/SiN rectangle with oxide cladding. LN, slots and hybrid stacks stay disabled until their own tests.
2. Rasterize onto boundary-aligned nonuniform mesh; preserve field staggering and quadrature weights. Initial heuristic ≤ smallest important feature/12, then refinement and domain expansion. Heuristic is a starting mesh, never a correctness certificate.
3. Implement established full-vector transverse-H FDE, reconstruct E/H, and sparse iterative solve behind `solveModes`. Benchmark matrix-free vs sparse/WASM on fixed sizes before selecting optimization; do not assume Hermitian Lanczos applies to every discretized operator.
4. Return residuals, forward-power normalization, mode index, beta/neff, polarization metric definition, boundary fraction. Use previous-mode overlap plus parity to track identity; reject ambiguous crossings.
5. Implement complex common-grid integration and documented projection convention. Check phase gauge invariance and positive power, and do not turn coefficients >1 into clamped efficiency questions.
6. Worker wrapper with cancellation, memory limits and timeout. Prefetch only eligible modes; fallback to analytic on unsupported devices/backlog.
7. External comparison: first slab TE/TM analytic dispersion, then Si 450×220 nm and SiN 1.5×0.8 µm with identical domains/materials in one respected external solver. Record solver version, mesh, fields and license; screenshots are not fixtures.
8. Solve nominal 20-case recipe bank, then ± geometry/tolerance variants; publish only converged cases. Broaden to near-cutoff, higher-order and anisotropic structures after separate gates.

Required tests: mesh refinement and domain enlargement; guide translated relative to grid; global complex phase change; scaling E/H together; same-mode unity; same-guide orthogonality; transverse polarization; interchange A/B for supported reciprocal convention; selected orthonormal-basis sum; rejection of leaky/ambiguous modes. Tolerances in K002.

Performance acceptance: measure p50/p95 solve latency and peak memory for desktop browser cold/warm cases; aim sub-second to a few seconds, not an unsupported promise. Never block the UI on an eigensolve. Precision mode requires tighter uncertainty than its displayed three-decimal answer; otherwise show insufficient-precision status.

## Delivered in the C3 foundation pass (2026-09-13)

- K2.2-lite materials (`src/physics/materials.mjs`): air/SiO2/Si sourced dispersion with hard validity windows; SiN and LN marked placeholder/disabled and ineligible for released answers.
- C3.1 mesh (`src/physics/mesh.mjs`): boundary-aligned nonuniform 1D slab mesh with one material per element, plus a 2D axis-aligned grid and priority rasterization. Feature/12 is only the starting heuristic.
- C3.2 slab FDE (`src/physics/slab.mjs`): Hermitian weak-form TE/TM operators, Dirichlet reduction, Sturm bisection and inverse iteration with no third-party linear algebra. Validated against closed-form 3-layer dispersion: |Δneff| ≈ 2e-5–3e-5 at feature/96 (release gate 1e-4), residual < 1e-8, boundary energy fraction < 1e-6, positive normalized forward power. TM uses the generalized `1/ε` mass; that and the `D^{-1/2}` similarity were the two operator bugs caught by the slab gate.
- C3.3 overlap (`src/physics/overlap.mjs`): common-grid symmetrized flux overlap and forward power; self-overlap 1, same-guide orthogonality, TE/TM decoupling, phase-gauge invariance and A/B reciprocity verified; raw η is never clamped.
- `solveModes` dispatcher (`src/physics/solve-modes.mjs`) supports `geometry.kind='slab'`; other kinds throw rather than fabricate.

Not done: 2D full-vector FDE and worker queue (step 6), external-solver benchmark (step 7 needs an external tool and license), mode-identity tracking across crossings, and solving the 20-case recipe bank. No numerical recipe answer has been released, and the pending recipes remain `pending-vector-solver`.

## Follow-on pass (2026-09-13): 2D infra, projection, scheduling, queue, adaptation

- C3.2 intermediate: `src/physics/fem2d.mjs` implements the 2D rectilinear Q1 assembly and a sparse Lanczos (shifted, full reorthogonalization, no third-party linalg) for the scalar operator. In the y-invariant limit it reproduces the slab TE benchmark (|Δneff| < 1.5e-3 at feature/36 and converging); residual < 1e-4 at 250 Lanczos steps. Its unit mismatch (k0 in 1/m vs µm coordinates) and the `D^{-1/2}` eigenvector mapping were both caught by the slab-equivalence test. `solveModes` supports `kind: 'scalar-2d'` and marks it `releaseEligible: false`.
- Full-vector 2D remains the blocker: the published transverse-H weak form is non-symmetric under naive nodal Galerkin, and the standard 2-component FDE needs either a mixed formulation or a non-Hermitian eigensolver. A residual < 1e-8 also needs a shift-invert (sparse solve) rather than plain Lanczos. Do not release recipes against the scalar operator.
- C5.1 `src/physics/projection.mjs`: selected-mode projection with `captured = Ση`, qualified remainder label, and complex mode combination. Slab TE0/TE1 orthonormal-basis tests pass.
- C4.2 `scheduleQuestions` (in `session-core`): G008 bucket quotas, category caps, bounded attempts, explicit shortage/coverage report and a final anti-starvation relaxation.
- C4.3 `src/solver/worker-queue.mjs`: worker pool with in-flight dedup, cancellation, telemetry, backlog exposure for analytic fallback; tested with a fake worker factory.
- C6.1 `src/lib/adaptive.ts`: deterministic profile, difficulty clamp (levels 3–5 never selected), and bounded sum-preserving bucket adaptation.
- C6.6 `src/solver/provenance.mjs`: warm-start gate on solver/material/mesh/wavelength and finite neff / positive power.

Still required before releasing the 20 recipes: the 2D full-vector operator, a shift-invert eigensolver for release-grade residuals, the external-solver benchmark, and mode-identity tracking. K5.4 (guide-envelope solves) and the browser E2E pass also remain open.

## 2D full-vector attempt (2026-09-13): experimental, gate not passed

`src/physics/fullvector2d.mjs` implements the self-adjoint magnetic curl-curl weak form on Q1 rectangles with the z-component encoded as `H_z = i·g`, so the transverse components and `g` are real and `K0, K1, K2` are real symmetric. Propagation gives `(K0 + βK1 + β²K2)x = 0`, solved by nonlinear Rayleigh iteration with a dense LU shift solve.

Two derivation errors were found and fixed: the sign of the β²/β terms, and (in the 1D standalone check) the sign of the shift in `A1 − σB1`.

What is verified: the assembled matrices are real symmetric and finite; a standalone 1D run of the same TM weak form matches closed-form dispersion (2.0485 vs 2.0528 at feature/20); the variational quotient of the sampled analytic TM profile converges toward the analytic value as the mesh refines.

Correction: the earlier "operator residual O(0.4–0.9)" measurement was a test bug (the trial vector kept nonzero values at Dirichlet nodes). After zeroing them the operator annihilates the analytic profile to discretisation accuracy, consistent with the 1D check. So the operator is not the problem.

Per review guidance (`spurious modes are fine; filter by energy confinement in a 2× core box, then take the highest-neff modes`), `solveFullVector2D` now infers the core box from the permittivity, computes each candidate's E-energy fraction inside a 2× window (`confinementFactor`, threshold `confinementThreshold`), discards low-confinement spurious modes, and ranks by neff so the fundamental is first. The companion-pencil shift-invert inverse iteration was also implemented (the plain `S(σ)⁻¹` step was not the right spectral transform for the quadratic).

What still fails: the Rayleigh/companion iteration does not reliably converge to the physical slab TE/TM pair on a resolved 2D grid. At feature/24 it lands near `neff 2.838` (TE-like, residual 0.076) but at feature/30 no candidate passes a 0.5 confinement threshold; TM (`2.053`) is not found. This module is marked experimental, `releaseEligible: false`, is not wired into `solveModes`, and no recipe answer uses it.

Needed to finish: a robust shift-invert eigensolver for the quadratic (sparse, with a properly scaled/preconditioned solve) and an external benchmark. The confinement filter is in place and ready to use once the iteration reaches the guided modes.
