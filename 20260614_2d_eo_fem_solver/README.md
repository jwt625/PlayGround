# 2D EO Electrostatic Prototype

This is a browser-first prototype for the EO cross-section solver. The primary
implementation is in `web/core` and runs scalar-permittivity 2D electrostatics
on a structured triangular mesh entirely on the client side. The Python package
is kept as a reference harness while the browser implementation becomes the
main product path.

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
python3 -m http.server 5173
open http://localhost:5173/web/
```

The example buttons reload YAML with cache busting. If you edit
`examples/parallel_plate.yaml` or another example on disk, click the
corresponding example button again to load the current file contents into the
browser editor. The app does not currently watch files automatically.

Reference Python CLI:

```bash
python3 -m eo_fem examples/parallel_plate.yaml
python3 -m eo_fem examples/two_cylinders.yaml
```

Validate:

```bash
npm test
python3 -m pip install -e ".[test]"
python3 -m pytest -q
```

Current mesh:

- The browser backend creates an `nx` by `ny` Cartesian grid over `Domain`.
- Each rectangular cell is split into two linear triangles: `(n00,n10,n11)` and
  `(n00,n11,n01)`.
- Electrodes are not yet geometry-conforming mesh boundaries; nodes inside each
  electrode shape are pinned to its Dirichlet potential.

Visualizer:

- Quantity selector: `phi`, `Ex`, `Ey`, `|E|`, `epsilon_r`,
  `epsilon_r_xx`, `epsilon_r_yy`, `epsilon_r_xy`, `r13`, `r33`, `r22`,
  `r_eff`.
- Quantity options and hover tooltips include short descriptions and
  expressions, for example `expr: -d(phi)/dx (x electric-field component)`.
- Scale selector: linear, linear symmetric, log magnitude.
- Mesh overlay toggle.
- Mesh size controls for `Simulation.mesh_nx` and `Simulation.mesh_ny`.
- Draggable splitter between the YAML editor and plot/results panel.
- Left pane is viewport-bounded; the YAML editor and log panel scroll
  internally.
- Right pane reserves stable rows for controls, progress, canvas, and numerical
  results so the plot canvas does not jump during solve/error states.
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
structured rectangle, scalar `eps_r`, Dirichlet conductor regions, and natural
Neumann outer boundaries. The next performance step is to move the same
config/results contract into a Web Worker, then consider Rust/WASM or WebGPU
only if the interactive mesh sizes need it.

Important material-model status:

- The electrostatic solve uses per-triangle scalar permittivity: each triangle
  is assembled with `epsilon_0 * eps_r(material)` evaluated at its centroid.
- Non-background material regions use the same ordering as the material maps:
  if regions overlap, the later non-background material in `Materials` wins.
- `epsilon_r_xx`, `epsilon_r_yy`, `epsilon_r_xy`, EO coefficients, and
  material-boundary overlays remain diagnostic until anisotropic tensor
  assembly and EO overlap are implemented.

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

- Browser and Python assembly support spatially varying scalar `eps_r`, but not
  anisotropic tensor permittivity.
- Electrode/material interfaces are not geometry-conforming.
- Charge extraction is not yet boundary-edge integration on tagged conductor
  boundaries.
- The solver runs on the main browser thread. The UI yields before solving so
  status/log/WIP can paint, but the numerical solve itself still blocks until
  complete.
- Web Worker execution, real iteration progress, preconditioning, nonuniform
  refinement, true FEM meshing, anisotropic tensors, and EO overlap remain next
  steps.

Next implementation step:

- Add anisotropic tensor assembly for `epsilon_r_xx`, `epsilon_r_yy`, and
  `epsilon_r_xy`, then move charge extraction from conductor-node residuals to
  tagged boundary-edge integration once geometry-conforming meshing exists.
