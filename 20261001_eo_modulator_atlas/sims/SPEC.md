# Simulation config spec: `eo-atlas.sim/v1`

Contract between three parties: the distillation skill (writes configs), the browser engine in `engine/` (runs them), the app `/sim` route (shows them). Status: draft 1, 2026-10-01. The engine author may refine it; every change must be recorded in the Changelog at the bottom and mirrored in `engine/schema/sim.schema.json`.

**Implemented subset:** E1 runs electrostatics and optional scalar optical modes.
The full-chain example and physics roadmap below include future stages; they are
not claims of implemented EO overlap, RF loss, loaded-line or EO-response support.
The example uses placeholder coordinates and is not itself a runnable config.

Rules
- A config holds inputs only: geometry, materials, line and optical settings, and the paper-reported targets with tolerances. Solver outputs are never stored.
- Lengths use explicit unit suffixes in keys (`_um`, `_nm`, `_mm`), frequencies `_ghz`, impedance `_ohm`, conductivity `_Sm`. The engine converts to SI.
- Every parameter that came from a paper carries a provenance entry under `provenance`, keyed by the dotted parameter path, with `class` in `paper_exact | figure_digitized | project_inference | standard_reference | unknown` and a `locator` (page / figure / table of `references/<paper_id>/text.md` pages) or a citation key for `standard_reference`. `unknown` values may not be silently defaulted: leave them out and list them under `missing`.
- File is YAML 1.2 (parsed in the browser with a real YAML parser) and also valid as JSON after parsing.

```yaml
schema: eo-atlas.sim/v1
id: chen2022-a                    # = device_id in data/devices.csv
paper_id: chen2022
device_id: chen2022-a
title: short human title            # not a claim of reproduction
repro_grade: A                      # A | B (C has no config)
validation_status: unvalidated      # unvalidated | analytic_gates_pass | literature_regression | cross_solver
chain: [electrostatics, optical_mode, eo_overlap, rf_line, loaded_line, eo_response]   # subset allowed

optics:
  wavelength_nm: 1550
  polarization: TE                  # quasi-TE | quasi-TM
  mode_index: 0
  group_index_from: finite_difference_wavelength   # or: fixed
  ng_fixed: null
  eo_axis: auto                     # engine derives the Pockels contraction from crystal + cut

materials:                          # name -> properties (tensors in the crystal frame)
  lithium_niobate:
    crystal: {cut: x, propagation: y}     # lab frame: x lateral, y film normal (up), z propagation
    n_o: 2.211
    n_e: 2.138
    eps_r: {perp: 43.0, par: 28.0}        # RF relative permittivity
    r_pm_per_v: {r13: 8.6, r22: 3.4, r33: 30.8, r51: 28.0}
    tan_delta_rf: 0.0
  silicon_dioxide: {n: 1.444, eps_r: 3.9, tan_delta_rf: 0.0}
  gold: {conductor: true, sigma_Sm: 4.1e7, n_complex: null}
  air: {n: 1.0, eps_r: 1.0}

geometry:                           # cross-section; origin on the optical axis; symmetry optional
  symmetry: none                    # none | x_mirror_odd (differential/push-pull half-domain)
  domain: {x_um: [-100, 100], y_um: [-50, 40]}
  regions:                          # painter order: later wins
    - {name: substrate, material: silicon, polygon_um: [[-100,-50],[100,-50],[100,-3],[-100,-3]]}
    - {name: box, material: silicon_dioxide, rect: {x_um: [-100,100], y_um: [-3,0]}}
    - {name: ln_slab, material: lithium_niobate, rect: {x_um: [-100,100], y_um: [0,0.25]}}
    - {name: ln_rib, material: lithium_niobate, polygon_um: [[-0.7,0.25],[0.7,0.25],[0.5,0.6],[-0.5,0.6]]}
  electrodes:
    - {name: signal, role: signal, weight: 1.0, rect: {x_um: [..], y_um: [..]}, material: gold}
    - {name: ground_l, role: ground, weight: 0.0, rect: {x_um: [..], y_um: [..]}, material: gold}
  optical_window: {x_um: [-3, 3], y_um: [-1, 1.5]}   # crop for mode solve
  mesh: {max_edge_um: 0.1, electrode_edge_um: 0.05}  # hints; engine refines near thin features

line:
  length_mm: 10
  conductor_loss_model: skin_effect_wheeler    # | from_paper_alpha | none
  rf_loss_table: null                          # [{f_ghz: .., alpha_db_per_cm: ..}] only when taken from the paper
  source_ohm: 50
  load_ohm: 50
  differential: true                           # voltage convention: Vpi quoted for differential (V+ - V-) drive
  loading:                                     # optional capacitively loaded electrode (T-rail) unit cell
    type: none                                 # none | periodic_t_rail
    period_um: null
    loaded_length_um: null
    unloaded_cross_section: null               # name of an alternate geometry block below, or null
    loaded_cross_section: null
sweep: {f_start_ghz: 0.1, f_stop_ghz: 110, n_points: 220}

alt_geometries: {}                  # name -> geometry block (same keys as `geometry`) for loaded / unloaded cuts

targets:                            # paper-reported values the sim is compared against (nothing else is stored)
  - {metric: vpi_l_dc_vcm, value: 2.2, tol_rel: 0.15, source: {device_id: chen2022-a, field: vpil_dc_vcm}}
  - {metric: n_rf, value: 2.2, tol_rel: 0.1, source: {device_id: chen2022-a, field: n_rf}}
  - {metric: z0_ohm, value: 50, tol_rel: 0.1, source: {device_id: chen2022-a, field: z0_ohm}}
  - {metric: eo_rolloff_db, at_ghz: 67, value: -1.4, tol_abs: 0.5, source: {...}}
  - {metric: bw3db_ghz, value: 110, tol_rel: 0.15, source: {...}}
reported_curves: []                 # optional [{name, x: f_ghz[], y: S21_dB[], provenance: figure_digitized, locator: ..}]
provenance: {}                      # dotted.path -> {class, locator|citation, note}
missing: []                         # parameters the paper does not disclose (never defaulted)
limitations: []                     # e.g. sidewall angle, scalar optical approximation, 2D quasi-static
```

Recognized target `metric` names: `vpi_l_dc_vcm`, `vpi_dc_v`, `n_eff`, `ng_opt`, `n_rf`, `z0_ohm`, `c_pul_pf_per_m`, `c0_pul_pf_per_m`, `l_pul_nh_per_m`, `rf_loss_db_per_cm` (with `at_ghz`), `eo_rolloff_db` (with `at_ghz`), `bw3db_ghz`, `bw6db_ghz`, `optical_confinement_in_region`. Recognition does not imply the metric is implemented.

Planned physics chain (each implemented stage reports its assumptions)
1. Electrostatics 2D (anisotropic eps tensor) -> C' (with dielectrics) and C0' (all eps = 1) -> `n_rf = sqrt(C'/C0')`, `Z0 = 1/(c sqrt(C' C0'))`.
2. Optical mode (quasi-TE/TM on the window) -> n_eff, group index via wavelength finite difference, mode profile.
3. EO overlap: refractive-index change from the Pockels tensor with the static field, perturbation integral -> `Vpi*L`. Convention flag: differential vs per-arm; push-pull factor explicit.
4. RF line: conductor loss from skin-effect / incremental inductance, dielectric loss from tan_delta -> alpha(f), complex gamma(f), Z(f).
5. Loaded line (optional): ABCD cascade of loaded/unloaded unit cells -> effective gamma, Z, Bragg limit.
6. EO response: traveling-wave integral with velocity mismatch, loss, source/load mismatch -> `S21_EO(f)`, 3 dB / 6 dB bandwidth. Optical-group velocity from stage 2.

Known approximation labels (must appear in `limitations` or in engine output): `scalar_optical_not_full_vector`, `quasi_static_2d_rf`, `rectangular_or_polygon_sidewalls_only`, `no_3d_launch_pad_effects`.

## E1 cross-section execution contract

`runCrossSection(yaml, {section, optical, meshScale, onProgress})` is shared by the
Node CLI and browser worker. It always solves electrostatics. `optical: true`
additionally requests the scalar optical stage; `chain` records the intended
device pipeline and does not cause the unimplemented stages to execute.

`inspectConfig(yaml)` is a separate, display-only path returning `{preview,
solveError}`. It validates YAML, metadata, material declarations and geometry,
then reports the strict input validator's first error without hiding the geometry
or disclosures. The preview contains only `raw` and `geometries`; no resolved
solver materials or default physical constants. Malformed metadata/geometry still
throws. A nonempty `solveError` disables Run. An empty one means the input boundary
passed, not that meshing, an optional optical stage or literature validation will
succeed. The worker and CLI always reparse through the strict solver boundary.
This allows geometry records with omitted `eps_r` to be inspected without making
them valid solver inputs or changing the JSON input schema. No `unknown` sentinel
or additional validation-status enum is introduced.

Inputs and numerical controls:

- Required: `schema`, non-empty `id`, material definitions, and a covered domain
  with regions and electrode shapes. Geometry uses painter order. Regions may
  extend beyond the domain and are clipped by the mesher. Shapes must be valid
  simple polygons/rectangles; arbitrary CAD validation is outside this contract.
- Dielectrics require positive `eps_r`, either scalar, `{perp, par}` or diagonal
  `{xx, yy, zz}` in the crystal frame. Anisotropy requires crystal orientation.
  Missing RF permittivity is an error, not vacuum. Conductors must be electrodes.
  Optical inputs use either `n` or both `n_o` and `n_e`; do not mix them.
- Electrode `weight` is the potential pattern, normalized to one volt of full
  terminal difference. For odd mirror symmetry this is twice the largest absolute
  weight; half-domain energy is doubled. `line.differential` does not add a second
  capacitance/voltage factor. This does not yet define optical arm/push-pull factors.
- `symmetry` defaults to `none`; supported mirror values are `x_mirror_odd` and
  `x_mirror_even`, with `mirror_x_um` defaulting to 0. Symmetry is a user-supplied
  geometry assumption, not inferred or independently verified by the runner.
- Outer electrostatic `boundary` sides (`left`, `right`, `top`, `bottom`) default
  to zero-normal-flux `neumann`; `dirichlet` means zero potential. Electrodes
  impose their normalized values. These finite-domain conditions require convergence
  studies before interpreting a device result.
- `meshScale` is 0.5–4, default 1; it scales local refinement hints, not physical
  geometry. Coarse/configured/fine UI choices are 2/1/0.5. Refinement is graded, so
  a single scale is not a uniform multiplier for every triangle.
- `mesh.max_vertices` is an integer from 4 to 80,000, default 80,000, for each
  electrostatic/optical mesh. Exceeding the budget errors instead of silently
  coarsening. The limit is a mesher stopping budget, not a memory guarantee.
- Mesh hints: positive `max_edge_um`, `electrode_edge_um`, `electrode_face_um`,
  `window_edge_um`, `far_edge_um`, `min_edge_um`, `grade`, `grade_far`, `grade_corner`,
  plus `min_angle_bound >= 1` and per-region positive edge sizes in `regions`.
  Unknown mesh/boundary/geometry keys are rejected. Defaults remain in
  `section.mjs`: max local edge = domain diagonal/40, electrode corner edge =
  min(max edge, thickness/6), face edge = min(max edge, 8×corner), optical-window
  refinement = min(max edge, 0.05 µm), far edge = max(local edge, diagonal/12).
- Optical mode solving requires `optical_window` within the domain and an `optics`
  block with positive `wavelength_nm`, TE/TM (or quasi-TE/quasi-TM), mode index 0–4
  (default 0), and `group_index_from`. Fixed group index requires positive
  `ng_fixed`. Finite differences use ±min(0.01 µm, wavelength/100).
- Optical boundaries are zero-field Dirichlet on the window; scalar slab tests
  use lower-level APIs with Neumann side boundaries. Optical mesh defaults use
  core edge λ/(22 nmax), far edge λ/8 and grading 0.25, scaled by `meshScale`.
  A conductor actually inside the optical mesh is unsupported even if a real
  index is supplied; an off-window conductor does not require an optical index.
- Named library materials supply dispersion slopes anchored to configured indices,
  or library indices if omitted. Out-of-range library evaluation errors in the
  optical solve. Custom constant indices have no material dispersion. This
  existing scalar model needs E2 review; it is not a vector/metal solver.
- `line.loading.type` may be `none` or `periodic_t_rail`. Periodic inputs require
  positive period and loaded length (no greater than period), an existing
  unloaded section and an existing loaded section; null/absent loaded section
  means top-level `geometry`. `--section` explicitly chooses which static slice
  to solve. Periodic settings never imply a cascade has been computed.
- The JSON schema describes input structure; runtime additionally checks numerical
  finiteness, intervals, references and solver constraints. Provenance completeness,
  physical geometry validity and literature accuracy still require review.

Result and target semantics:

- Outputs include `scope: cross_section`, completed `stages`, declared
  `pendingStages`, unit-labelled `metrics`, mesh diagnostics, warnings, elapsed
  time and `targets`. Physical arrays/fields are not serialized by this runner.
- Electrostatic metrics: C′ and C₀′ in pF/m, L′ in nH/m, static section nRF and Z₀.
  The optional optical stage adds scalar effective index and group index.
- `targetSummary` reports total/evaluated/passed/failed separately. Missing
  predictions are `actual: null`, `status: not_evaluated`, with an explicit reason.
- A target needs finite `value` and nonnegative `tol_abs` and/or `tol_rel`.
  When both are present, the tolerance is max(tol_abs, |value|×tol_rel). Comparison
  is inclusive. Future metric names are accepted only from the recognized list.
- Alternate-section targets, targets with `at_ghz`, and unimplemented metrics
  are not evaluated. Static RF targets require explicitly uniform topology
  (`line.loading.type: none`); periodic or unspecified topology is not comparable.
  A configured fixed group index is never tested against a target as a prediction.
- Requested-stage failure aborts the run with an error; no partial success result
  is returned. Worker cancellation terminates the worker and discards that run.
  No inputs or validation statuses are modified and no result files are created.
- CLI exit codes: 0 = requested stages ran (even if every target is unevaluated),
  1 = input/solver error, 2 = one or more evaluated targets failed. A zero exit
  code is not evidence of literature reproduction.

## Changelog
- 2026-10-01 draft 0
- 2026-10-01 draft 1 (E1/U1): documented the implemented cross-section subset,
  validation/resource controls, loading semantics and honest target eligibility;
  added C₀′ diagnostic metric. Full-chain roadmap and paper inputs remain unchanged.
- 2026-10-01 draft 1 follow-up: documented the separate geometry/disclosure preview
  for incomplete drafts from p1_01/p1_02; strict input schema and physical-input
  requirements unchanged. YAML 1.2 remains required for schema checks.
