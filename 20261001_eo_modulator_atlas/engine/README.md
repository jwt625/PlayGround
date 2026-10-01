# Cross-section engine

The browser worker and Node CLI share `src/run.mjs`; YAML parsing and input
validation share `src/config.mjs`. The engine is GPL-3.0-or-later as declared in
its original module headers and package manifest; the license text is in
`LICENSE`. The design follows the `eo_fem` lineage recorded in the project
DevLogs. `robust-predicates` and `yaml` retain their upstream licenses.

Implemented stages:

- Conforming triangle meshing in JavaScript (not a WASM backend).
- Anisotropic P1 electrostatic FEM, terminal-charge and energy capacitance,
  vacuum capacitance and quasi-TEM section estimates of L′, nRF and Z₀.
- Scalar E/H optical FEM and finite-difference group index for dielectric windows.

RF loss, EO overlap, periodic-cell ABCD and traveling-wave EO response are not
implemented. `line`/`sweep` inputs are preserved for these future stages. A
requested optical solve that intersects a conductor without a supported optical
model errors explicitly. The current Chen input has this condition.

Capacitance uses **one volt of full terminal voltage difference**. The solved
half-domain energy is doubled for mirror symmetry, then C′ = 2W′/V²; no separate
drive factor is applied from `line.differential`. Charge and energy agreement is
a consistency diagnostic, not a substitute for convergence testing.

Run `npm test`. The independent analytic references are:

- [MIT 8.02, Capacitance, §§5.2–5.4](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/99407b81a217613e8ddec21dc3da07de_chap5capacitance.pdf):
  parallel plates and electrostatic energy.
- [MIT 6.974, Optical Waveguides and Integrated Optics](https://www.ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/7ce86803e5d51ad749a86c37f5d72ce6_wavegude_int_opt.pdf):
  symmetric dielectric slab dispersion.

The Chen electrostatic input runs at its configured mesh, but a single static
section cannot validate its periodic transmission line. The analytic tests do
not establish accuracy for that device. Neither a passing test run nor opening
the UI changes `validation_status` in any input file.

The callable contract is documented in `../sims/SPEC.md` (E1 subsection).
`targetSummary` reports evaluated coverage separately from passing comparisons.
Static RF comparisons require explicitly uniform topology. Frequency-specific,
alternate-section and configured-input targets are excluded. A stage error aborts
the run; there is no partially successful serialized result. Each mesh inherits
the configured vertex budget (default/maximum 80,000) and fails rather than
silently reducing resolution. Browser cancellation terminates the worker.

`tests/fixtures/parallel-plates.yaml` is a synthetic analytic test input, not a
literature config. Tests create transient CLI files under the OS temporary
directory and remove them. `../app/scripts/smoke.mjs` runs the same input in Chrome
and Node; its optional screenshots belong in ignored scratch paths.
