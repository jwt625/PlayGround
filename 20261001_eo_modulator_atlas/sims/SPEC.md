# Simulation config spec: `eo-atlas.sim/v1`

Contract between three parties: the distillation skill (writes configs), the browser engine in `engine/` (runs them), the app `/sim` route (shows them). Status: draft 0, 2026-10-01. The engine author may refine it; every change must be recorded in the Changelog at the bottom and mirrored in `engine/schema/sim.schema.json`.

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

Supported `metric` names (engine): `vpi_l_dc_vcm`, `vpi_dc_v`, `n_eff`, `ng_opt`, `n_rf`, `z0_ohm`, `c_pul_pf_per_m`, `l_pul_nh_per_m`, `rf_loss_db_per_cm` (with `at_ghz`), `eo_rolloff_db` (with `at_ghz`), `bw3db_ghz`, `bw6db_ghz`, `optical_confinement_in_region`.

Physics chain (what the engine computes; each stage reported with its own assumptions)
1. Electrostatics 2D (anisotropic eps tensor) -> C' (with dielectrics) and C0' (all eps = 1) -> `n_rf = sqrt(C'/C0')`, `Z0 = 1/(c sqrt(C' C0'))`.
2. Optical mode (quasi-TE/TM on the window) -> n_eff, group index via wavelength finite difference, mode profile.
3. EO overlap: refractive-index change from the Pockels tensor with the static field, perturbation integral -> `Vpi*L`. Convention flag: differential vs per-arm; push-pull factor explicit.
4. RF line: conductor loss from skin-effect / incremental inductance, dielectric loss from tan_delta -> alpha(f), complex gamma(f), Z(f).
5. Loaded line (optional): ABCD cascade of loaded/unloaded unit cells -> effective gamma, Z, Bragg limit.
6. EO response: traveling-wave integral with velocity mismatch, loss, source/load mismatch -> `S21_EO(f)`, 3 dB / 6 dB bandwidth. Optical-group velocity from stage 2.

Known approximation labels (must appear in `limitations` or in engine output): `scalar_optical_not_full_vector`, `quasi_static_2d_rf`, `rectangular_or_polygon_sidewalls_only`, `no_3d_launch_pad_effects`.

## Changelog
- 2026-10-01 draft 0
