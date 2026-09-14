# 001 — Coding contracts and milestone tasks

Status: C1 implemented; C2 UI and C1.4 store implemented; C3.1–C3.3 1D slab + overlap foundation implemented (2D full-vector pending). Date: 2026-09-13.

Use browser-compatible ESM physics/data modules now; add SvelteKit + TypeScript UI following SLM-Guessr at C2. Keep numerical arrays separate from presentation. No server dependency for gameplay. Units: geometry, MFD and wavelength in µm; angles in radians; index dimensionless; E/H in consistent SI normalization when vector solving exists. MFD is diameter, Gaussian w=MFD/2.

| Task | Concrete output | Depends on | Acceptance |
|---|---|---|---|
| C1.1 | Versioned presets, seeded RNG, explicit units | K005 | Replay matches; bounds and cross-constraints checked |
| C1.2 | Analytic Gaussian overlap and fixtures | K002 | Independent quadrature agrees; identity, shift, tilt, polarization regressions |
| C1.3 | Canonical SHA-256 keys, persistent analytic bank, numerical recipes | C1.1–2 | Rebuild byte-identical; settings invalidate keys; no fake numerical answers |
| C1.4 | IndexedDB mode store and memory fallback | C1.3 | Real-browser reload, quota, corruption and schema-upgrade tests |
| C2.1 | Svelte app, gallery, paired field viewer | C1; UI004 | Responsive usable routes; same physical scale; keyboard controls |
| C2.2 | Session/scoring/reveal/results; 20 questions | G003, C2.1 | Submit once; restart; zero/one endpoints; no visible answer before submit |
| C2.3 | Interval and ranking answer adapters | C2.2 | Deterministic options, boundary bins and tied ranks tested |
| C3.1 | Material functions, geometry rasterization, nonuniform mesh | K002 | Version/source metadata, supported wavelengths, interface placement stability |
| C3.2 | Full-vector FDE worker backend | C3.1 | Slab and external-solver benchmark gates in 007 |
| C3.3 | Common-grid E/H overlap and quality checks | C3.2 | Symmetry, convergence, power normalization validated |
| C4.1 | Solve pending recipes and publish numerical field bundles | C3.3, K005 | All released examples have validation provenance |
| C4.2 | Constraint-aware pair generation and outcome balancing | C4.1, G008 | No starvation; measured bin/category coverage; bounded rejection |
| C4.3 | Worker queue of 5–20, telemetry and cache reuse | C1.4, C4.2 | No UI-thread solve; backlog falls back to eligible analytic/cached pool |
| C5.1 | TM, TE1, phase/parity and selected-mode projection | C3–4 | Correct identity tracking; selected basis qualified |
| C5.2 | LN tensor rotation and hybrid stack scenarios | C5.1 | Anisotropic external benchmark before release |
| C6.1 | Adaptive profile, campaign and extra modes | C2, C4–5, G008 | Reproducible sessions; no unvalidated content selected |

Contracts for later implementation:

- `solveModes(problem, signal) -> ModeBundle`: problem includes geometry, resolved material tensors, wavelength, boundary conditions, mesh and solver versions/settings. Bundle contains neff/beta, complex Ex/Ey/Ez/Hx/Hy/Hz with stagger locations, quadrature mesh, forward power, polarization convention, residual and refinement diagnostics.
- `overlap(A, B, transform, settings) -> result`: include definition/version and eligibility limits. Never derive physics from image pixels.
- `Question`: seed, generator/preset/material/solver/mesh/overlap versions, complete sampled parameters, field references, transforms, concepts, difficulty and answer. Persist full parameters; floating-point/eigensolver cross-browser bitwise replay is not guaranteed merely by a seed.
- `QuestionView`: explicitly allowlisted pre-answer fields; private answer stays outside rendered view. This prevents accidental UX leakage, not inspection by a determined local user.
- `SolverCache`: mode key excludes question seed, pairing, colormap and pure post-solve transforms. Question key includes both modes, transforms, integration and overlap versions.

Original brief coverage: §§1,12–19,27 → K002/C3; §§2–4,21–25,34–35 → G003/G008/C2/C6; §§5–11,20,26,33 → K005/G008/C4/C5; §§28–30 → C1/C4.3/006; §§31–32 → contracts above; §§36–38 → K002 and milestone gates. FEM, WebGPU, arbitrary polygons, rounded corners and curvature/rotated-Gaussian support remain explicit extensions after their validation, not prerequisites for C2.
