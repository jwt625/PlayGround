# DevLog 004: Gmsh/MFEM Backend Plan

Date: 2026-07-10

## Decision Context

The current project is a browser-first 2D EO cross-section prototype. It already
has a compact YAML input format, an interactive frontend, a structured-grid
electrostatic solver, and a placeholder vector optical-mode result contract.

The main numerical limitations are now clear:

- Material and electrode boundaries are not geometry-conforming.
- Electrodes are pinned by nodes inside conductor shapes, not tagged boundary
  entities.
- Charge extraction is not boundary-flux integration.
- The vector optical mode solver is not a validated full-vector Maxwell
  waveguide eigensolver.

Gmsh and MFEM are a strong candidate pair for a serious backend path:

- Gmsh should own geometry-conforming 2D meshing and physical tags.
- MFEM should own FEM assembly, sparse solves, H1 electrostatics, and later
  H(curl) / eigenmode work.

This should be added as a backend, not a wholesale replacement of the browser
solver.

## Open-Question Decisions

Decisions from 2026-07-10:

- Package the MFEM/Gmsh path as a CLI or API-only backend. The frontend should
  call a local service when needed, but the numerical backend should remain
  usable from scripts and tests without the browser.
- Keep the backend as performant as practical. Prefer the fastest reasonable
  path that preserves correctness: PyMFEM is acceptable for first experiments,
  but C++ MFEM or compiled extension paths remain open if Python overhead
  becomes material.
- Pursue both electrostatics and optical modes, but implement electrostatics
  first because it is scalar and much easier to validate.
- Replace implicit material overlap semantics with explicit material/domain
  assignment.
- Adopt a COMSOL-like conceptual model: geometry creates domains, boundaries,
  edges, and vertices; materials, physics, and boundary conditions are assigned
  to explicit domains/boundaries or named selections.
- Add selection functionality later, including explicit selections, box
  selections, and derived selections.
- Start with simple output artifacts, then move toward general/compatible
  formats.
- Use analytic validation first. Use Tidy3D only if needed; its 2D mode solver
  is a good practical reference because it is accessible/free for this scale.
- A local server is acceptable. It must choose ports carefully and avoid
  interrupting existing services.
- GPL licensing is acceptable. The project should move toward GPL-compatible
  licensing if Gmsh becomes a required dependency.

## Local Installation Status

Initial machine inspection:

- macOS Darwin arm64 on Apple Silicon.
- Homebrew available at `/opt/homebrew/bin/brew`.
- Default Python: 3.14.3.
- Existing repo venv: `.venv`, also Python 3.14.3.
- Before installation, `gmsh` CLI and Python `gmsh`/`mfem` modules were absent.

Homebrew packages selected first because they provide bottled Apple Silicon
builds. Installation completed:

- `gmsh` 4.15.2
- `mfem` 4.9

Python package installation into `.venv` also completed:

- `gmsh` 4.15.2 from macOS arm64 wheel.
- `mfem` / PyMFEM 4.8.0.1 built successfully from source for Python 3.14 arm64.
- Installed Python numerical dependencies: `numpy` 2.4.6, `scipy` 1.18.0,
  `numba` 0.66.0, `llvmlite` 0.48.0.

Smoke tests passed:

- `gmsh --version` reports `4.15.2-git`.
- Python `gmsh` imported, initialized, and generated a unit-square 2D mesh with
  physical tags.
- Python `mfem.ser` imported and exposes `Mesh`, `H1_FECollection`, and
  `ND_FECollection`.

Note: Homebrew MFEM and PyMFEM are not the same version (`4.9` vs `4.8.0.1`).
That is acceptable for first experiments, but backend code should avoid mixing
the Homebrew C++ library and the PyPI-built Python wrapper in one process.

## Proposed Architecture

Keep the current browser implementation as the interactive/default solver.
Add a Python backend service or CLI route that accepts the same YAML config and
returns the same result contract.

Proposed config extension:

```yaml
Simulation:
  physics: electrostatic        # electrostatic | optical_mode | vector_mode
  backend: browser              # browser | mfem
  mesh_backend: gmsh            # structured | gmsh
```

Longer-term geometry/material model:

```yaml
Geometry:
  objects:
    domain_box:
      shape: rectangle
      x_min: -12e-6
      x_max: 12e-6
      y_min: -8e-6
      y_max: 8e-6
    core:
      shape: rectangle
      x_min: -0.25e-6
      x_max: 0.25e-6
      y_min: -0.11e-6
      y_max: 0.11e-6

Selections:
  silicon_core:
    method: explicit
    domains: [core]
  signal_electrode_boundary:
    method: explicit
    boundaries: [signal_top, signal_bottom, signal_left, signal_right]

Materials:
  silicon:
    selection: silicon_core
    n: 3.476
    eps_r: 11.7

Physics:
  electrostatic:
    dirichlet:
      signal:
        selection: signal_electrode_boundary
        potential: 1.0
```

The current compact YAML can be kept as a compatibility layer, but the Gmsh/MFEM
backend should move toward explicit domains, boundaries, and selections instead
of relying on shape overlap order.

Backend flow:

1. Browser sends YAML to a local/server endpoint, or tests/scripts call the CLI
   directly.
2. Backend parses the existing project YAML.
3. Gmsh builds a 2D conforming mesh from the same domain/material/electrode
   shapes.
4. Gmsh assigns physical surfaces for material regions and physical curves for
   electrode boundaries.
5. MFEM reads the Gmsh mesh and maps physical tags to material and boundary
   attributes.
6. MFEM solves the requested physics.
7. A result adapter converts MFEM grid functions into the frontend result
   schema.
8. The frontend plots the returned unstructured mesh and fields.

Local server behavior:

- Bind to a configurable host/port.
- Prefer an unused default port and fail gracefully if occupied.
- Do not kill or replace existing services.
- Print the chosen URL clearly when started.
- Keep CLI execution independent of server startup.

Likely file layout:

```text
eo_fem/backends/
  gmsh_mesh.py
  mfem_electrostatic.py
  mfem_scalar_mode.py
  result_adapter.py
  api.py
```

The frontend should not know much about MFEM. It should only know that a solver
backend can return a `SolveResult`-like JSON object.

## Result Contract

For electrostatics, keep the current result shape as much as possible:

```js
{
  physics: "electrostatic",
  backend: "mfem",
  mesh: {
    domain,
    nodes,
    triangles,
    boundaryEdges,
    stats
  },
  phi,
  fields: {
    Ex,
    Ey,
    normE
  },
  materials,
  electrodeLabels,
  capacitanceEnergy,
  capacitanceCharge,
  energyPerLength,
  residual,
  iterations,
  units,
  reference
}
```

The existing renderer already has `mesh.triangles`, so unstructured plotting is
mostly an adapter problem. Current code that assumes `nx`, `ny`, `xCoords`, and
`yCoords` should receive structured fallbacks only when present.

For optical/vector modes, preserve the DevLog 003 contract:

```js
{
  physics: "vector_mode",
  backend: "mfem",
  mesh,
  wavelength,
  frequency,
  targetNeff,
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
      confinement,
      targetOverlap,
      modeArea,
      residual
    }
  ]
}
```

## Visualization Decision

Do not replace the current browser visualization with full ParaView. The app is
a domain-specific EO cross-section workbench, not a generic scientific
postprocessor. The primary workflow should remain:

- Edit or load YAML.
- Solve electrostatic or optical modes.
- Inspect field quantities, material maps, mesh overlays, capacitance/mode
  metrics, logs, and validation messages in a modern browser UI.

Full ParaView is valuable as an external inspection/debugging tool, but it
should not become the main product interface.

Recommended visualization stack:

```text
MFEM/Gmsh backend
  -> result.json       -> current browser UI / custom canvas
  -> result.vtu/.vtk   -> ParaView / ParaView Glance / external debugging
  -> optional vtk.js   -> richer in-app unstructured visualization
```

### Short-Term Path: Existing Browser Canvas

Keep and improve the current custom browser visualization first.

Required changes for Gmsh/MFEM:

- Add explicit viewer modes similar to COMSOL:
  - `Geometry`: inspect domains, boundaries, vertices, and selections before
    meshing.
  - `Mesh`: inspect generated elements, mesh quality, physical groups, and
    boundary tags.
  - `Results`: inspect solved fields, derived quantities, capacitance, modal
    fields, and plots.
- Support arbitrary unstructured triangle meshes.
- Plot scalar nodal fields over `mesh.triangles` without assuming structured
  `nx`, `ny`, `xCoords`, or `yCoords`.
- Add boundary-edge overlays for electrode and material-interface tags.
- Add hover/picking for nearest node or containing triangle.
- Keep existing quantity dropdowns, mode selector, scale selector, mesh overlay,
  logs, and result summaries.

This preserves usability and minimizes UI disruption while the backend changes.

### Medium-Term Path: vtk.js

Consider `vtk.js` for richer in-browser FEM visualization after the custom
unstructured canvas path works.

Good uses:

- Unstructured-grid rendering.
- Contours/isolines.
- Vector glyphs.
- Streamline-like field inspection if needed.
- 3D extruded cross-section views.
- Loading backend-generated VTK/VTU artifacts directly in browser.

This is the lower-level web-native piece behind ParaView Glance and is a better
fit for embedding in this app than a full ParaView UI.

### Companion Tool: ParaView / ParaView Glance

Export VTK/VTU artifacts by default or behind a debug flag so results can be
opened in ParaView or ParaView Glance.

Use cases:

- Verify unstructured mesh tags.
- Inspect field interpolation and boundary conditions.
- Debug MFEM grid functions.
- Compare against other FEM/EM tools.
- Share portable simulation artifacts.

ParaView should be a compatibility/debug path, not the main application shell.

### Later Optional Path: trame + ParaView

Only consider trame/ParaView-backed browser visualization if the project needs:

- remote/server-side rendering,
- very large datasets,
- advanced ParaView filters,
- multi-user visualization sessions,
- or a Python-native visual analytics app.

This is probably too heavy for the first Gmsh/MFEM backend milestone.

## Implementation Phases

### Phase 1: Gmsh Mesh Backend

Scope:

- Domain rectangle.
- Rectangle and circle materials.
- Rectangle and circle electrodes.
- Explicit geometric domains and boundaries.
- Physical surfaces for named domains/selections.
- Physical curves for named boundary selections.
- Mesh-size controls using existing `mesh_nx` / `mesh_ny` as approximate global
  targets plus optional local refinement fields later.

Important behavior:

- Move away from existing overlap semantics. Material assignment should be
  explicit through domains/selections.
- Keep a compatibility adapter for old YAML examples during the migration.
- Emit a mesh debug artifact (`.msh`) for inspection.
- Record physical tag maps in JSON for reproducibility.
- Add minimal explicit selection support first; add box and derived selections
  later.

### Phase 2: MFEM Electrostatic Backend

Scope:

- H1 scalar potential solve.
- Scalar and diagonal/tensor permittivity coefficients.
- Dirichlet boundary conditions on tagged electrode boundaries.
- Natural outer boundary unless specified otherwise.
- Boundary-flux charge integration on signal electrode.

Validation target:

- Match or improve current analytic-reference tests with lower mesh sensitivity.
- Add tests against explicit domain/material assignment rather than relying on
  overlap ordering.

### Phase 3: Frontend Backend Adapter

Scope:

- Add a backend selector or config-driven backend route.
- Add `POST /api/solve` or a local CLI bridge.
- Provide a CLI entry point for backend solves.
- Start the local API server only when explicitly requested.
- Select an unused port instead of interrupting existing services.
- Make plot and hover paths robust to unstructured meshes.
- Preserve the current custom browser canvas as the primary visualization path.
- Emit optional VTK/VTU artifacts for ParaView/Glance debugging.
- Keep current browser backend as fallback/default.

### Phase 3B: Rich Browser Visualization

Scope:

- Evaluate `vtk.js` once unstructured canvas plotting and backend JSON are
  stable.
- Add `vtk.js` only for visualization capabilities that are painful to maintain
  manually, not as a replacement for the domain-specific controls.
- Keep frontend state and solver controls owned by the current app.

### Phase 4: Scalar FEM Optical Mode

Scope:

- Implement scalar Helmholtz-like mode solve on conforming mesh.
- Use this as a bridge from current scalar optical mode to MFEM eigensolvers.
- Validate against analytic slab waveguide results.
- Add Tidy3D 2D mode-solver comparison only after the analytic ladder is clean
  or if a practical high-index-contrast reference is needed.

### Phase 5: Full-Vector Waveguide Mode Solver

Scope:

- Implement fixed-frequency, beta-eigenvalue waveguide formulation.
- Use MFEM H(curl) or mixed spaces as appropriate.
- Add spurious-mode filtering and residual checks.
- Add boundary/PML strategy.
- Validate against Tidy3D, MPB, or Lumerical before using for design numbers.

This is the hardest phase. MFEM examples cover Maxwell eigenproblems, including
anisotropic Maxwell eigenproblems, but the integrated-photonics waveguide
problem is not just a drop-in cavity eigenproblem.

## Validation Ladder

### Electrostatic Validation

| Case | Reference | Purpose |
|---|---|---|
| Parallel plates | `C' = eps0 epsr W / g` | Basic H1 Poisson and capacitance |
| Two cylinders | closed-form capacitance | Curved boundary and mesh conformity |
| Layered dielectric plates | series capacitance per area | Material tags and interfaces |
| Anisotropic slab | directional capacitance response | Tensor coefficient sanity |
| Manufactured Poisson solution | exact `phi(x,y)` with source | FEM convergence independent of device geometry |
| Mesh convergence sweep | h-refinement slope | Boundary tags and charge extraction |

### Optical Validation

| Case | Reference | Purpose |
|---|---|---|
| Uniform PEC cavity | analytic Maxwell eigenvalues | H(curl) eigenproblem sanity |
| Symmetric slab waveguide | analytic TE/TM transcendental modes | beta/effective-index validation |
| Weak-guidance step-index guide | semi-analytic LP modes | polarization/degen sanity |
| Si strip waveguide | MPB/Tidy3D/Lumerical | practical high-index-contrast benchmark |
| TFLN/BTO stack | Tidy3D/Lumerical | anisotropic and modulator-relevant benchmark |

## Immediate Step-by-Step Plan

The first implementation milestone should be a CLI-callable Gmsh/MFEM
electrostatic solve that reproduces the current parallel-plate example with a
geometry-conforming mesh.

### Milestone 0: Dependency And Licensing Cleanup

1. Add optional backend dependencies to `pyproject.toml`, for example
   `.[backend] = ["gmsh", "mfem", "numpy", "scipy"]`.
2. Add a short setup note to `README.md` for `.venv` backend installation.
3. Add or update project license metadata toward GPL-compatible licensing if
   Gmsh becomes a required project dependency.
4. Add a backend smoke test that imports `gmsh` and `mfem.ser`.

Exit criteria:

- `python -m pytest` can skip backend tests if optional dependencies are absent.
- Backend-capable environments can import both packages cleanly.

### Milestone 1: Explicit Geometry/Selection Schema

1. Define a minimal internal geometry model:
   - domains,
   - boundaries,
   - vertices,
   - named selections.
2. Add a compatibility translator from the current compact YAML examples into
   that internal model.
3. Support explicit domain material assignment.
4. Support explicit boundary-condition assignment to boundary selections.
5. Keep old YAML examples working during migration.

Exit criteria:

- `examples/parallel_plate.yaml` can be translated into explicit domain and
  electrode-boundary selections.
- Material assignment no longer depends on "later material wins" in the backend
  path.

### Milestone 2: Gmsh Mesh Generator

1. Implement `eo_fem/backends/gmsh_mesh.py`.
2. Generate a rectangular 2D domain with tagged physical surfaces.
3. Generate electrode boundary curves and tag them as physical groups.
4. Export a simple `.msh` debug artifact.
5. Export a tag map JSON artifact that records domain/boundary/selection names.
6. Add a mesh smoke test for the parallel-plate example.

Exit criteria:

- Gmsh generates a valid `.msh` for parallel plates.
- Physical surface and boundary tags are recoverable from the artifact.
- Mesh generation is deterministic enough for tests.

### Milestone 3: MFEM Electrostatic Solve

1. Implement `eo_fem/backends/mfem_electrostatic.py`.
2. Read the Gmsh mesh into MFEM.
3. Build an H1 finite-element space for potential.
4. Assign scalar permittivity by physical domain.
5. Apply Dirichlet conditions on tagged electrode boundaries.
6. Solve the linear system.
7. Compute energy per length.
8. Compute signal charge by boundary-flux integration.
9. Return a result object compatible with the existing Python/browser result
   schema.

Exit criteria:

- Parallel-plate capacitance is within the existing analytic tolerance first,
  then tightened after mesh/boundary extraction is stable.
- Result JSON includes mesh nodes, triangles, `phi`, `Ex`, `Ey`, `normE`,
  capacitance, and debug metadata.

### Milestone 4: CLI Backend Entry Point

1. Add a CLI route such as:

   ```bash
   python -m eo_fem backend-solve examples/parallel_plate.yaml \
     --backend mfem \
     --out artifacts/parallel_plate_mfem/result.json
   ```

2. Create an artifact directory per run.
3. Write `result.json`.
4. Write simple mesh artifacts first: `.msh` and tag-map JSON.
5. Add VTK/VTU export after the simple artifacts are stable.

Exit criteria:

- Backend solve can run without starting a browser or server.
- Artifacts are reproducible and easy to inspect.

### Milestone 5: Local API Server

1. Add an explicitly started local API server.
2. Bind to a configurable host/port.
3. Select an unused port if the preferred port is occupied.
4. Never kill or replace existing services.
5. Add `POST /api/solve` for YAML-in, JSON-out.
6. Keep the CLI route independent of the server.

Exit criteria:

- Browser can request a backend solve.
- Existing static-browser workflow still works without the backend server.

### Milestone 6: Frontend Unstructured Visualization

1. Make plotting robust to unstructured meshes.
2. Render nodal scalar fields over arbitrary triangles.
3. Add boundary-edge overlays for electrodes and material boundaries.
4. Add hover behavior for nearest node or containing triangle.
5. Add viewer modes:
   - Geometry mode for domains/boundaries/vertices/selections.
   - Mesh mode for elements, physical groups, and mesh-quality diagnostics.
   - Results mode for scalar/vector fields and derived plots.
6. Keep existing quantity and mode controls.

Exit criteria:

- MFEM electrostatic result plots in the current browser UI.
- Browser solver and MFEM backend result contracts can coexist.
- Users can inspect geometry and mesh before running or trusting a solve.

### Milestone 7: Expanded Electrostatic Validation

1. Parallel plates.
2. Two cylinders.
3. Layered dielectric plates.
4. Anisotropic slab.
5. Manufactured Poisson solution.
6. Mesh convergence sweep.

Exit criteria:

- Analytic tests pass with documented tolerances.
- Convergence behavior is recorded before optical work begins.

### Milestone 8: Scalar Optical FEM

1. Implement scalar optical mode on conforming mesh.
2. Validate against analytic slab waveguide modes.
3. Add Tidy3D comparison only if analytic tests are clean and a practical
   high-index-contrast benchmark is needed.

Exit criteria:

- Scalar optical FEM reproduces slab waveguide effective indices.
- The result contract matches the existing optical mode UI.

### Milestone 9: Full-Vector Waveguide Modes

1. Choose and document the MFEM formulation.
2. Implement the isotropic fixed-frequency beta eigenproblem.
3. Add spurious-mode filtering and residual checks.
4. Validate against analytic/cavity/slab tests before strip-waveguide examples.
5. Add anisotropy only after isotropic vector validation.

Exit criteria:

- The solver returns quantitatively defensible `Ex`, `Ey`, `Ez`, `Hx`, `Hy`,
  `Hz`, `n_eff`, and residuals for benchmark cases.

## Current Recommendation

Start with Gmsh + MFEM electrostatics only. That gives immediate value:

- Geometry-conforming material/electrode interfaces.
- Tagged conductor boundaries.
- Better capacitance extraction.
- A clean backend adapter pattern.

Do not start with full-vector waveguide modes. Build confidence through the
electrostatic validation ladder, then scalar optical FEM, then full-vector
waveguide modes.
