# DevLog 003: Full-Vector EM Mode Solver Plan

Date: 2026-06-14

## Decision

The scalar optical mode solver from DevLog 002 is not sufficient. It is useful
as a UI/data-contract scaffold, but it cannot produce real `Ex`, `Ey`, `Ez`,
polarization fractions, or anisotropic-media modes.

The next optical implementation should be a full-vector 2D mode solver for
waveguide cross-sections, still browser/client-side only.

Target behavior:

- Solve vector Maxwell eigenmodes for translationally invariant waveguides.
- Return actual `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`, `|E|`, `|H|`, and
  intensity-like quantities.
- Support isotropic optical index first.
- Add diagonal anisotropic optical tensors after isotropic vector validation.
- Use Tidy3D as the primary benchmark/eval reference.

## Why Scalar Is Not Enough

Current scalar equation:

```text
(L_t + k0^2 n(x,y)^2) psi = beta^2 psi
```

This has only one unknown, `psi`. Mapping `psi` to `Ex`, `Ey`, or `Ez` is only a
visual convention. It cannot represent:

- vector boundary conditions at dielectric interfaces,
- longitudinal fields,
- hybrid TE/TM modes,
- polarization fraction,
- anisotropic tensor coupling,
- accurate high-index-contrast Si/BTO/TFLN modes.

For the BTO-on-Si example, scalar mode 0 can now look fundamental-like after
Gaussian initialization, but it is still not a full-wave result.

## Recommended Formulation

Use a finite-difference frequency-domain vector mode solver on a Yee-like
staggered grid.

Preferred first implementation:

```text
Unknowns: Hx, Hy on transverse grid
Eigenvalue: beta^2
Recover: Hz, Ex, Ey, Ez after solve
```

Reasoning:

- H-field transverse formulations are common for dielectric waveguide mode
  solvers.
- They handle discontinuities in dielectric media better than naive nodal
  E-field scalar formulations.
- They can produce electric-field components needed for EO overlap.
- They map reasonably onto structured grids already used by the browser app.

Alternative:

```text
Unknowns: Ex, Ey
Recover: Ez and H fields
```

This may be easier to connect to EO overlap, but naive E-field finite
differences are more prone to interface artifacts unless the discretization is
careful.

## Mathematical Target

For isotropic material:

```text
epsilon(x,y) = epsilon0 * n(x,y)^2
mu = mu0
fields(x,y,z) = fields(x,y) * exp(i beta z)
```

Frequency-domain Maxwell equations:

```text
curl E = -i omega mu0 H
curl H =  i omega epsilon E
```

A practical vector mode solver should assemble a sparse generalized or standard
eigenproblem:

```text
A u = beta^2 B u
```

where `u` is a vector of transverse field components.

The implementation should avoid dense matrices for normal UI mesh sizes.

## Browser Numerical Plan

### Phase 1: Isotropic Vector FDFD

Scope:

- Uniform structured grid only.
- Isotropic optical index `n`.
- PEC/Dirichlet-like outer boundary first.
- No PML yet.
- Solve a small number of modes near `target_neff`.

Implementation:

1. Build derivative operators on the optical `mode_window` grid.
2. Assemble sparse block operator for transverse field components.
3. Use CSR typed arrays for block matrices.
4. Implement matrix-free operator application where possible.
5. Use iterative eigensolver:
   - start with block subspace iteration or Lanczos,
   - then improve to shift-invert if a sparse linear solver is available.
6. Return real-valued fields for lossless media initially.

Initial output contract:

```js
{
  physics: "vector_mode",
  modes: [
    {
      nEff,
      beta,
      Ex,
      Ey,
      Ez,
      Hx,
      Hy,
      Hz,
      normE,
      normH,
      intensity,
      teFraction,
      tmFraction,
      modeArea,
      targetOverlap,
      residual
    }
  ]
}
```

### Phase 2: Targeted Eigen Solve

The scalar solver currently reports `target_neff` but still behaves like a
largest-beta solver. A vector solver needs real targeting.

Target:

```text
sigma = (k0 * target_neff)^2
find eigenpairs nearest sigma
```

Options:

- Matrix-free Lanczos with polynomial filtering around `sigma`.
- Shift-invert with an iterative linear solve:

```text
(A - sigma B)^-1 B u
```

- WASM sparse direct solver later if JS iterative solves are insufficient.

Implementation sequence:

1. Build robust orthogonalization and Ritz extraction.
2. Add residual checks based on `||A u - beta^2 B u||`.
3. Sort by:
   - closeness to `target_neff`,
   - mode-region overlap,
   - low node count / smoothness only as a fallback diagnostic.

### Phase 3: Boundary Conditions And PML

Start:

- Dirichlet/PEC-like outer boundary for strongly guided modes.

Then add:

- absorbing boundary/PML around optical `mode_window`,
- radiation/leaky-mode handling,
- complex `n_eff` or loss estimates if needed.

### Phase 4: Anisotropic Optical Tensor

After isotropic vector validation:

- Support diagonal optical tensors:

```yaml
Materials:
  ln:
    n_xx: 2.138
    n_yy: 2.211
    n_zz: 2.211
```

- Internally use:

```text
epsilon_xx = n_xx^2
epsilon_yy = n_yy^2
epsilon_zz = n_zz^2
```

- Later support rotated tensors:

```yaml
orientation_deg: ...
epsilon_rotation: ...
```

Do not add rotated off-diagonal tensor support until diagonal anisotropy passes
Tidy3D comparisons.

## UI Plan

When vector solver is active:

- Quantity dropdown should expose:
  - `Ex`, `Ey`, `Ez`,
  - `Hx`, `Hy`, `Hz`,
  - `|E|`, `|H|`,
  - intensity,
  - optical index maps.
- Mode dropdown should list:

```text
0: n_eff 2.41, TE 91%, overlap 84%, residual 1e-6
1: n_eff 1.89, TE 12%, overlap 51%, residual 2e-6
```

- Result panel should include:
  - `n_eff`,
  - `beta`,
  - `target_neff`,
  - TE/TM fraction,
  - mode area,
  - confinement/overlap,
  - residual,
  - solver iteration count,
  - boundary condition/PML status.

## YAML Contract

Keep one YAML per physical device.

Shared physical structure:

```yaml
Domain: ...
Materials: ...
Electrodes: ...
```

Physics-specific controls stay in `Simulation`:

```yaml
Simulation:
  physics: vector_mode
  wavelength: 1.55e-6
  target_neff: auto
  num_modes: 4
  mode_window:
    shape: rectangle
    x_min: ...
    x_max: ...
    y_min: ...
    y_max: ...
  mode_region:
    shape: rectangle
    x_min: ...
    x_max: ...
    y_min: ...
    y_max: ...
  optical_boundary: pec
```

Keep `physics: optical_mode` as a legacy alias for scalar mode until the vector
solver is stable. Once vector mode works, prefer:

```yaml
physics: vector_mode
```

## Validation Plan

### Analytic / Semi-Analytic

1. Uniform dielectric box:
   - known separable modes with PEC boundaries.
2. Slab waveguide:
   - compare TE/TM effective indices to slab dispersion equations.
3. Weak-guidance fiber-like channel:
   - scalar and vector should agree approximately.

### Tidy3D Benchmarks

Use Tidy3D as eval, not as browser dependency.

Cases:

1. Tidy3D rectangular dielectric waveguide tutorial.
2. Si strip:
   - 500 nm x 220 nm,
   - SiO2 cladding,
   - 1550 nm.
3. TFLN rib:
   - same YAML geometry,
   - compare `n_eff`, field shape, dominant polarization.
4. BTO-on-Si:
   - Si core with BTO blanket,
   - compare `n_eff` and BTO field overlap.

Metrics:

- `n_eff` absolute error.
- Field visual agreement.
- TE/TM fraction.
- Mode area.
- Material-region intensity overlap.

Target acceptance for first vector solver:

```text
Si strip n_eff within 1-3% of Tidy3D on comparable grid/window.
Field peak in same material region.
Dominant polarization agrees.
```

Tighter agreement requires PML, subpixel smoothing, and better material
interfaces.

## Implementation Tasks

- [ ] Create `web/core/vector_mode_solver.js`.
- [ ] Add vector-mode result and plot-value contract.
- [ ] Add derivative/stencil helpers for Yee-like grid.
- [ ] Add matrix-free block operator for isotropic vector mode.
- [ ] Add eigensolver utilities:
      orthogonalization, Ritz projection, residual calculation.
- [ ] Add field reconstruction for `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`.
- [ ] Add mode sorting by target neff and region overlap.
- [ ] Add UI quantity groups for true vector fields.
- [ ] Add tests for uniform box/slab sanity.
- [ ] Add Tidy3D benchmark scripts and checked-in small reference JSON.
- [ ] Add diagonal anisotropic tensor support.
- [ ] Add PML or absorbing boundary option.
- [ ] Add optical-RF overlap only after vector fields are credible.

## Risks

- A correct vector solver is much more sensitive to discretization than the
  scalar MVP.
- Naive finite differences can create spurious modes.
- Browser-only eigensolving may become slow for useful grid sizes.
- Shift-invert may require a stronger sparse linear solver than current JS
  utilities.
- Anisotropic rotated tensors can easily be wrong if coordinate conventions are
  not explicit.

## Near-Term Recommendation

Do not keep extending scalar `Ex/Ey/Ez` UI proxies. Keep them clearly labeled as
temporary.

Next coding slice should be:

1. Add Tidy3D benchmark harness for Si strip and BTO-on-Si.
2. Implement isotropic vector mode solver on a small uniform grid.
3. Validate against slab/Si strip before touching anisotropy.

