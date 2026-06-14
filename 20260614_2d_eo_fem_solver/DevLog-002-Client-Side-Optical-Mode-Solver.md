# DevLog 002: Client-Side Optical Mode Solver

Date: 2026-06-14

## Decision

Add an optical/photonic mode-solver path to the same browser-only client, but
start with a scalar transverse Helmholtz eigenmode solver rather than a full
vector FDFD solver.

This is a pragmatic first slice:

- It reuses the existing YAML, material rectangles, domain parsing, structured
  grid, plotting, and validation infrastructure.
- It gives immediate browser-side fields and `n_eff` for waveguide
  cross-section sanity checks.
- It creates a concrete path to benchmark against Tidy3D `ModeSolver` and
  `ModeSimulation` examples before implementing full-vector modes.

The scalar solver is not a replacement for Tidy3D, Lumerical MODE, MPB, or
COMSOL. It is a local estimator and UI/contract scaffold.

## Tidy3D Benchmark Notes

Sources checked on 2026-06-14:

- Tidy3D example notebook, "Using the mode solver for optical mode analysis":
  https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolver/
- Tidy3D API docs for `tidy3d.plugins.mode.ModeSolver`:
  https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.plugins.mode.ModeSolver.html
- Tidy3D API docs for `tidy3d.ModeSimulation`:
  https://docs.simulation.cloud/projects/tidy3d/en/latest/api/_autosummary/tidy3d.ModeSimulation.html

Important details from the benchmark surface:

- `ModeSolver` is initialized from a Tidy3D `Simulation`, a modal plane, a
  `ModeSpec`, and one or more frequencies.
- `ModeSpec` exposes `num_modes` and `target_neff`; if `target_neff` is absent,
  modes with largest real effective index are computed.
- Tidy3D mode data exposes `n_eff`, complex effective index, loss-like
  imaginary quantities, field profiles, polarization fractions, and mode area.
- Tidy3D fields are xarray arrays such as `Ex`, `Ez`, and can be plotted by
  `mode_solver.plot_field(...)`.
- Tidy3D warns that the remote/server mode solve with subpixel averaging is more
  accurate than the local mode solver. For this repo, Tidy3D should be treated
  as the reference/eval path, not as a dependency of the browser app.
- Tidy3D now also documents `ModeSimulation` as a 2D electromagnetic eigenmode
  simulation for translationally invariant geometries.

Useful initial eval cases:

1. Tidy3D tutorial waveguide:
   - rectangular waveguide,
   - `wg_width = 1.5 um`,
   - `wg_height = 1.0 um`,
   - `permittivity = 4.0`,
   - wavelength around `2.0 um`,
   - compare first few `n_eff` values and scalar field shape qualitatively.
2. Si strip waveguide:
   - `500 nm x 220 nm` silicon core,
   - `n_Si ~= 3.476`,
   - `n_SiO2 ~= 1.444`,
   - wavelength `1.55 um`,
   - compare browser scalar `n_eff` against Tidy3D vector mode data.

Expected scalar error:

- For high-index-contrast Si strip, scalar `n_eff` can be materially different
  from vector TE/TM results because the vector boundary conditions and
  polarization mixing are ignored.
- For weak-guidance or larger low-contrast waveguides, scalar field shape and
  `n_eff` should be a better trend estimator.

## Implemented In This Slice

Browser core:

- Added `web/core/mode_solver.js`.
- Added `Simulation.physics: optical_mode` route.
- Added scalar finite-difference eigenproblem:

```text
(L_t + k0^2 n(x,y)^2) psi = beta^2 psi
n_eff = beta / k0
```

- Uses uniform structured grid with Dirichlet outer boundary.
- Finds largest-real `beta^2` modes by shifted power iteration plus
  Gram-Schmidt deflation.
- Reports:
  - `n_eff`,
  - `beta`,
  - wavelength/frequency,
  - scalar modal field,
  - `|mode|`,
  - modal intensity,
  - approximate core confinement,
  - approximate mode area,
  - eigen iterations and residual.

Materials and validation:

- Added optical material keys:
  - `n`,
  - `n_xx`,
  - `n_yy`,
  - `n_xy`.
- `material.n` is preferred for optical mode solving.
- If `n` is absent, optical index falls back to `sqrt(eps_r)` for compatibility.
- Electrodes are no longer required when `Simulation.physics` is
  `optical_mode` or `mode_solver`.
- Optical mode configs require `Simulation.wavelength`.
- The optical MVP rejects `Simulation.refinement.enabled: true`; nonuniform-grid
  FDFD stencils are a next step.

UI:

- Added a `Si mode` example button.
- App dispatches to the electrostatic solver or optical mode solver based on
  `Simulation.physics`.
- Result panel switches from capacitance fields to mode fields.
- Quantity selector switches between electrostatic quantities and optical
  quantities:
  - `mode`,
  - `mode_abs`,
  - `mode_intensity`,
  - `n`,
  - `eps_r`.

Example:

- Added `examples/si_strip_mode.yaml`:
  - `500 nm x 220 nm` Si core,
  - SiO2-like background,
  - `lambda = 1.55 um`,
  - browser scalar mode solve.

Current browser scalar result for that example:

```text
n_eff       2.669276
beta        1.082036e7 1/m
confinement 79.38%
mode area   1.432958e-13 m^2
mesh nodes  11011
iteration   583
residual    1.993e-4
```

This number should not be presented as the real TE0/TM0 vector mode of the
strip. It is the scalar MVP output to benchmark and improve.

## Verification

Browser tests:

```text
npm test
17 passed
```

Python reference tests:

```text
.venv/bin/python -m pytest -q
10 passed
```

New browser tests:

- `examples/si_strip_mode.yaml` parses, validates, and solves.
- Scalar `n_eff` lies between cladding and core index.
- Core confinement is nonzero and meaningful.
- Increasing core index increases scalar `n_eff`.

## Proposal For Next Slices

### Slice 1: Tidy3D Eval Harness

Create a separate offline comparison harness, not used by the browser app:

- `benchmarks/tidy3d/si_strip_mode.py`
- `benchmarks/tidy3d/tidy3d_rect_waveguide.py`
- Export a small JSON reference:

```json
{
  "case": "si_strip_500x220_sio2_1550",
  "tool": "tidy3d",
  "mode": 0,
  "wavelength_m": 1.55e-6,
  "n_eff": 2.4,
  "mode_area_m2": null,
  "notes": "vector Tidy3D reference; exact value depends on grid, plane size, and polarization filter"
}
```

Keep generated benchmark data checked in only if it is small and clearly labeled
by tool/version/date.

### Slice 2: Better Eigen Solver

Replace shifted power iteration with a more robust browser eigen solve:

- Lanczos with reorthogonalization for largest algebraic eigenpairs.
- Optional shift-invert later if a WASM sparse solver is introduced.
- Report convergence status explicitly:
  - converged,
  - max iterations reached,
  - residual.

### Slice 3: Full-Vector 2D FDFD

Implement a vectorial mode solver before using browser results for serious
high-index-contrast photonic design:

- Start with transverse H-field or E-field FDFD formulation.
- Support isotropic optical index first.
- Add anisotropic optical tensors after scalar/vector isotropic convergence is
  validated.
- Compare TE fraction, TM fraction, `Ex/Ey/Ez` shape, and `n_eff` against
  Tidy3D.

### Slice 4: Optical-RF Overlap

Once vector optical fields are stable:

- Interpolate RF field from electrostatic solver to optical grid.
- Compute EO overlap for selected material regions.
- Report `VpiL` with explicit assumptions:
  - optical mode normalization,
  - RF voltage normalization,
  - EO tensor component/orientation,
  - scalar vs vector optical-field model.

## Known Limitations

- Scalar mode equation only.
- No vector boundary conditions.
- No PML or radiation-loss mode filtering.
- No anisotropic optical tensor assembly.
- No dispersive material model.
- Uniform-grid optical solve only.
- Dirichlet outer boundary can bias weakly confined modes.
- Current eigen solver is acceptable for MVP examples but not robust enough for
  crowded higher-order modes.

