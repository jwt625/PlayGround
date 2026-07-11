# 2D EO Cross-Section Solver Prototype

This is a browser-first prototype for EO modulator cross-section studies. The
primary implementation is in `web/core` and runs entirely on the client side:

- 2D electrostatic extraction for capacitance and RF fields.
- Tensor-permittivity RF assembly for anisotropic EO materials.
- A first scalar optical/EM mode solver for waveguide field sanity checks.

The Python package is kept as a reference harness for electrostatics while the
browser implementation is the main product path.

The input format is a deliberately small, MOOSE-like YAML subset:

```yaml
Simulation:
  name: parallel_plate
  mesh_nx: 121
  mesh_ny: 81
Domain:
  x_min: -12e-6
  x_max: 12e-6
  y_min: -8e-6
  y_max: 8e-6
Materials:
  background:
    eps_r: 3.9
Electrodes:
  signal:
    shape: rectangle
    potential: 1.0
    x_min: -5e-6
    x_max: 5e-6
    y_min: 1e-6
    y_max: 1.5e-6
  ground:
    shape: rectangle
    potential: 0.0
    x_min: -5e-6
    x_max: 5e-6
    y_min: -1.5e-6
    y_max: -1e-6
Outputs:
  reference: parallel_plate
  plate_width: 10e-6
  plate_gap: 2e-6
```

Run:

```bash
make frontend
```

The launcher prints the frontend URL. It prefers port `5173`; if that port is
occupied, it scans upward and uses the first available port without stopping
the existing service. Override the starting port with `make frontend
FE_PORT=5174`.

The equivalent command without Make is:

```bash
.venv/bin/python -m eo_fem.dev_server --host 127.0.0.1 --port 5173 --directory .
```

Useful development targets:

```bash
make setup          # install frontend/reference/backend development dependencies
make solve          # run CONFIG=examples/parallel_plate.yaml with the Python reference solver
make backend-check  # verify the optional Gmsh and PyMFEM imports
make gmsh-mesh      # persist a Gmsh mesh under artifacts/gmsh_preview
make gmsh-examples  # generate advanced 2D mesh-control examples
make gmsh-examples-3d # generate initial 3D tet and swept-prism examples
make gmsh-test      # generate and validate the tagged Gmsh mesh fixture
make check          # Python/browser tests, Ruff, and mypy
```

There is not yet a backend API server to start. Gmsh meshing is implemented,
but the MFEM electrostatic solver, backend CLI route, and API service remain
planned work in DevLog 004 milestones 3-5.

Advanced Gmsh sizing/refinement research, COMSOL mesh-control mapping, and the
2D/3D example roadmap are documented in
`DevLog-006-Advanced-Gmsh-Mesh-Control-Research.md`. The primary-source evidence
index is cached at `references/meshing/source-index.json`.

### Geometry, mesh, and solution artifacts

The browser workspace has independent `Geometry`, `Mesh`, and `Results` layers
in its Model Builder. Use the `Open` controls in the top toolbar, or select a
loaded layer in the model tree.

When started with `make frontend`, the same local process also exposes a small
development API and populates the `Library` selector:

```bash
curl http://127.0.0.1:5173/api/health
curl http://127.0.0.1:5173/api/examples
curl -X POST http://127.0.0.1:5173/api/mesh/generate \
  -H 'Content-Type: application/json' \
  --data '{"config_path":"examples/mesh_controls/parallel_plate_proximity.yaml","name":"api_parallel_plate"}'
```

Generation accepts only configs under `examples/`, writes under
`artifacts/api/`, and runs each Gmsh build in an isolated subprocess because
Gmsh uses process-global native state. A frontend mesh can also be deep-linked
for repeatable inspection:

```text
http://127.0.0.1:5173/web/?artifact=/artifacts/mesh_controls/parallel_plate_proximity/parallel_plate_proximity.msh
```

- Geometry: current project YAML (`.yaml` / `.yml`) or JSON containing the same
  config structure. Loading new geometry invalidates the previous mesh and
  results.
- Mesh: ASCII Gmsh MSH 4.1 (`.msh`) with first- or second-order line/triangle
  elements, or normalized mesh JSON. Physical names and tagged boundary edges
  are preserved.
- Solution: JSON containing a mesh plus nodal field arrays. A solution may also
  contain its source config so Geometry, Mesh, and Results can be restored
  together.

To generate and inspect a persistent Gmsh mesh:

```bash
make gmsh-mesh CONFIG=examples/parallel_plate.yaml
make frontend
```

Then choose `Open -> Mesh` and select
`artifacts/gmsh_preview/mesh_preview.msh`.

Advanced validation artifacts:

```bash
make gmsh-examples
make gmsh-examples-3d
```

The 2D artifacts under `artifacts/mesh_controls/` can be loaded in the browser.
The 3D artifacts under `artifacts/mesh_controls_3d/` should currently be opened
in Gmsh or ParaView; the in-app viewer remains 2D-only.

Automated inspection outputs, including frontend screenshots, clipped ParaView
screenshots, VTK conversions, and quality reports, are written under
`artifacts/inspection/`.

Normalized mesh JSON uses zero-based node indices:

```json
{
  "mesh": {
    "nodes": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    "triangles": [[0, 1, 2]],
    "boundaryEdges": [{"nodes": [0, 1], "name": "ground"}]
  }
}
```

A saved solution adds nodal fields whose lengths equal `mesh.nodes.length`:

```json
{
  "schemaVersion": "eo-fem.workspace/v1",
  "mesh": {
    "nodes": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    "triangles": [[0, 1, 2]]
  },
  "fields": {
    "phi": [0.0, 1.0, 0.5]
  },
  "metadata": {
    "backend": "mfem"
  }
}
```

The example buttons reload YAML with cache busting. If you edit
`examples/parallel_plate.yaml` or another example on disk, click the
corresponding example button again to load the current file contents into the
browser editor. The app does not currently watch files automatically.

The browser UI has two example groups:

- `ES`: electrostatic examples.
- `EM`: optical mode examples.

The `Physics` selector can use the YAML route (`config`) or temporarily force
`ES` / `EM mode`. The TFLN and BTO buttons in the ES and EM groups load the
same physical device YAMLs; the EM buttons force optical mode solving without
duplicating geometry/material definitions.

Reference Python CLI:

```bash
python3 -m eo_fem examples/parallel_plate.yaml
python3 -m eo_fem examples/two_cylinders.yaml
```

Validate:

```bash
npm test
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy
```

Optional Gmsh/MFEM backend setup:

```bash
uv sync --extra dev --extra backend
uv run pytest -q
```

Current mesh:

- The browser backend creates an `nx` by `ny` Cartesian grid over `Domain`.
- Each rectangular cell is split into two linear triangles: `(n00,n10,n11)` and
  `(n00,n11,n01)`.
- Optional browser-side structured refinement can snap tensor-product grid
  coordinates to material/electrode boundaries and add local guard spacing.
- Electrodes are not yet geometry-conforming mesh boundaries; nodes inside each
  electrode shape are pinned to its Dirichlet potential.

Optical/EM mode solver:

- Enabled with:

```yaml
Simulation:
  physics: vector_mode
  wavelength: 1.55e-6
  target_neff: auto
  num_modes: 4
```

- `physics: optical_mode` remains as a legacy scalar finite-difference
  Helmholtz path:

```text
(L_t + k0^2 n(x,y)^2) psi = beta^2 psi
n_eff = beta / k0
```

- `mode_polarization` selects the optical tensor component used by the scalar
  solve:
  - `Ex -> n_xx`
  - `Ey -> n_yy`
  - `Ez -> n_zz`
  - omitted/unknown -> scalar `n`
- `target_neff: auto` resolves to the maximum selected optical index over the
  optical solve mesh and is reported in results. The current eigensolver still
  computes largest-beta scalar modes; true target-centered search needs
  shift-invert or a robust Lanczos filter.
- `physics: vector_mode` adds the first client-side isotropic vector-mode
  contract. It generates transverse Ex/Ey candidates on the same structured
  grid, reconstructs Ez and Hx/Hy/Hz with finite-difference Maxwell relations,
  reports TE/TM fractions, and exposes `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`,
  `|E|`, `|H|`, and intensity in the plot controls.
- Optical examples default to `physics: vector_mode`. The browser's ES toolbar
  buttons explicitly override shared TFLN/BTO device YAMLs back to
  electrostatic solves.
- `mode_window` can crop the EM eigenproblem around the waveguide while reusing
  the same global `Domain` and `Materials`.
- `mode_region` reports/selects modes by intensity overlap with a rectangular
  waveguide/core region.
- EM plot controls include a mode dropdown when `num_modes > 1`; changing the
  selected mode updates the plot and result text without rerunning the solver.
- Vector-mode results should still be benchmarked against Tidy3D/Lumerical/MPB
  before using for quantitative design; PML, diagonal anisotropy, and true
  target-centered shift-invert are not implemented yet.

Visualizer:

- Quantity selector uses solver-specific nested groups. ES runs show ES/RF/EO
  quantities; EM runs show EM mode fields and optical material indices.
- Quantity options and hover tooltips include short descriptions and
  expressions, for example `expr: -d(phi)/dx (x electric-field component)`.
- Scale selector: linear, linear symmetric, log magnitude.
- Mesh overlay toggle.
- Mesh size controls for `Simulation.mesh_nx` and `Simulation.mesh_ny`.
- Draggable splitter between the YAML editor and plot/results panel.
- Left pane is viewport-bounded; the YAML editor and log panel scroll
  internally.
- Right pane reserves stable rows for controls, progress, canvas, and a
  scrollable plain-text numerical result panel so the plot canvas does not jump
  during solve/error states.
- Plot interaction: drag to pan, mouse wheel to zoom, double-click or Reset view
  to restore full-domain view, resize-aware canvas redraw.
- Hover tooltip reports `x`, `y`, selected value, and the expression used for
  the selected quantity.
- During solve, the plot clears and shows an in-canvas WIP state instead of
  stale field data.
- Solve progress indicator is shown during parse/validation/solve. True
  iteration-level progress requires moving the solver into a Web Worker or
  making the CG loop asynchronous.
- Left-bottom log panel records timestamped validation, solve start, solve
  finish, and error messages. Validation failures remain visible there with
  actionable error text.
- Dark theme with sharp-corner controls by default.
- The `material_stack` example exercises diagnostic material-property maps.

Validation:

- The browser validates config values before solving and before realtime update.
- It rejects non-finite numeric fields, non-positive domain dimensions, invalid
  mesh sizes, non-positive permittivity values, malformed electrode geometry,
  and invalid analytic-reference dimensions.

The current browser backend is intentionally simple: P1 triangles on a
structured rectangle, scalar or tensor `eps_r`, Dirichlet conductor regions,
and natural Neumann outer boundaries. The next performance step is to move the
same config/results contract into a Web Worker, then consider Rust/WASM or
WebGPU only if the interactive mesh sizes need it.

Important material-model status:

- The electrostatic solve uses per-triangle permittivity evaluated at the
  triangle centroid.
- If `eps_r_xx`, `eps_r_yy`, or `eps_r_xy` are present, assembly uses the
  symmetric anisotropic tensor. Otherwise it reduces to scalar `eps_r`.
- Non-background material regions use the same ordering as the material maps:
  if regions overlap, the later non-background material in `Materials` wins.
- EO coefficients and material-boundary overlays remain diagnostic until EO
  overlap is implemented.
- Optical mode solves use `n`, `n_xx`, `n_yy`, and `n_zz` if available; if
  optical index is absent, the browser falls back to `sqrt(eps_r)`.
- TFLN and BTO examples now carry both RF and optical material properties in
  one YAML file per physical device. Geometry/material definitions are shared
  between ES and EM runs; physics-specific solver controls live under
  `Simulation`.

Performance notes:

- Sparse stiffness storage uses CSR typed arrays in the browser core.
- The browser and Python solvers use Jacobi-preconditioned CG with a default
  relative residual tolerance of `1e-6`, which is well below the current
  geometry/discretization error of the structured-mesh prototype.
- A 281 x 221 high-contrast BTO example is roughly 1 s in Node on a local
  laptop; browser timing depends on rendering and main-thread load.
- WebGPU is possible, but should come after a Web Worker split and a stable CSR
  or stencil kernel. The best first GPU target would be repeated sparse matvec
  in CG; assembly, validation, UI, and small meshes are not worth moving to GPU.

True FEM mesh path:

- Add a browser triangulation library or WASM mesher that supports constrained
  segments and region labels.
- Generate vertices/segments from material and electrode boundaries.
- Tag triangle regions for material properties and boundary edges for electrode
  charge extraction.
- Reuse the existing P1 triangle stiffness assembly; it already accepts
  arbitrary triangle coordinates.
- Replace node-inside-electrode Dirichlet pinning with boundary/entity tags.

Current limitations:

- Browser and Python assembly support spatially varying scalar and anisotropic
  tensor permittivity.
- Browser structured refinement is available behind
  `Simulation.refinement.enabled`; the Python reference path still uses the
  uniform structured grid.
- Electrode/material interfaces are not geometry-conforming.
- Charge extraction is not yet boundary-edge integration on tagged conductor
  boundaries.
- The solver runs on the main browser thread. The UI yields before solving so
  status/log/WIP can paint, but the numerical solve itself still blocks until
  complete.
- Web Worker execution, real iteration progress, stronger preconditioning,
  Python-side nonuniform refinement, true FEM meshing, tagged-edge charge
  extraction, full-vector optical modes, target-centered optical eigen solving,
  and EO overlap remain next steps.

Next implementation step:

- For electrostatics: mirror structured nonuniform refinement in the Python
  reference path, add convergence tests for TFLN/BTO examples, then move charge
  extraction from conductor-node residuals to tagged boundary-edge integration
  once geometry-conforming meshing exists.
- For optics: add a Tidy3D benchmark harness and replace the scalar
  largest-beta mode solve with a robust target-centered/vector mode solver.
