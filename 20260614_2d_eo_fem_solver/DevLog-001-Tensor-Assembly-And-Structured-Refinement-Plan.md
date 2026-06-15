# DevLog 001: Tensor Assembly And Structured Refinement Plan

Date: 2026-06-14

## Decision

Implement anisotropic permittivity assembly before mesh refinement. This is now
complete for the browser and Python reference paths.

The current examples already contain tensor RF permittivity values, but the
old solver treated them as diagnostic maps. That was a larger physics error for
TFLN and BTO examples than the uniform structured mesh error. After tensor
assembly, the next numerical accuracy step is structured nonuniform refinement;
the browser path now has a first implementation, while the Python reference
path still needs to mirror it.

## Current State

- Browser and Python solvers use P1 triangles on structured tensor-product
  grids.
- Material regions are sampled at triangle centroids.
- Browser and Python assembly support scalar `eps_r` and symmetric anisotropic
  tensors from `eps_r_xx`, `eps_r_yy`, and `eps_r_xy`.
- Browser validation rejects non-positive scalar/diagonal permittivity and
  non-positive-definite in-plane tensor combinations.
- Browser-side structured refinement is available through
  `Simulation.refinement.enabled`.
- Browser refined meshes snap tensor-product x/y coordinates to material and
  electrode rectangle boundaries plus circle center/bounding coordinates, then
  add guard coordinates around those interfaces.
- Browser result/log text reports permittivity model, mesh type, mesh size,
  triangle count, and min/max `dx`/`dy`.
- Electrodes are imposed by pinning nodes inside conductor regions.
- Charge extraction uses conductor-node residuals, not tagged boundary edges.
- The Python reference path still uses the old uniform structured grid.
- The stable local browser URL for this project is `http://localhost:5173/web/`.
- Later update: DevLog 002 added a browser-only scalar optical/EM mode solver,
  optical material-index tensor keys, ES/EM UI routing, and unified TFLN/BTO
  YAMLs that share geometry/material definitions between electrostatic and EM
  runs.

## Completed In This Slice

- Added browser tensor material accessor and tensor P1 element stiffness.
- Added Python tensor material accessor and matching tensor P1 element
  stiffness.
- Added `permittivityModel` descriptions for homogeneous/spatial scalar and
  homogeneous/spatial tensor runs.
- Added focused JS and Python tests:
  - isotropic tensor assembly matches scalar assembly,
  - vertical-field capacitance responds more strongly to `eps_r_yy` than
    `eps_r_xx`.
- Added browser structured nonuniform coordinate generation.
- Added browser mesh statistics and UI/log reporting.
- Updated tooltip lookup and field-component finite differences for nonuniform
  structured coordinates.
- Updated `README.md` so material-model and next-step status matches the code.

Verification:

```text
npm test
15 passed at completion of this slice

Current browser suite after DevLog 002:
npm test
20 passed

.venv/bin/python -m pytest -q
10 passed
```

## Proposed Sequence

1. Add anisotropic tensor assembly. Done.
2. Add focused tensor regression tests. Done.
3. Update result/log text so users know whether scalar or tensor permittivity
   was used. Done.
4. Add browser structured nonuniform mesh generation with boundary snapping.
   Done.
5. Mirror structured nonuniform refinement in the Python reference path.
6. Add mesh convergence utilities and tests.
7. Defer true unstructured meshing and tagged boundary-edge charge integration
   until the structured tensor/refinement path is stable.

## Tensor Assembly Plan

For each triangle, evaluate material properties at the centroid and assemble:

```text
K_ij = epsilon0 * area * grad(N_i)^T * eps_r_tensor * grad(N_j)

grad(N_i) = [b_i, c_i] / (2A)

K_ij = epsilon0 / (4A) *
       (eps_xx * b_i * b_j
      + eps_xy * b_i * c_j
      + eps_yx * c_i * b_j
      + eps_yy * c_i * c_j)
```

Assume a symmetric tensor for now:

```text
eps_yx = eps_xy
```

Material property fallback:

- If `eps_r_xx` and/or `eps_r_yy` exist, use tensor assembly.
- Missing diagonal tensor components fall back to scalar `eps_r`.
- Missing `eps_r_xy` falls back to `0`.
- If only `eps_r` exists, the tensor path must reduce exactly to scalar
  isotropic assembly with `eps_xx = eps_yy = eps_r`, `eps_xy = 0`.

## Structured Nonuniform Mesh Plan

Use a tensor-product structured grid first, not a full unstructured mesher.
This keeps the browser implementation simple while resolving thin films and
electrode gaps much better than a uniform grid.

Candidate algorithm:

1. Collect mandatory x/y coordinates:
   - domain boundaries,
   - rectangle material boundaries,
   - rectangle electrode boundaries,
   - circle centers and bounding-box extrema.
2. Add local guard coordinates around mandatory interfaces:
   - `edge +/- h_min`,
   - `edge +/- 2 h_min`,
   - clipped to the domain.
3. Fill large intervals with spacing capped by `h_max`.
4. Preserve existing triangle splitting and P1 assembly.
5. Report mesh statistics:
   - node count,
   - triangle count,
   - min/max `dx`,
   - min/max `dy`.

Implemented browser YAML extension:

```yaml
Simulation:
  mesh_type: structured
  mesh_nx: 281
  mesh_ny: 221
  refinement:
    enabled: true
    h_min: 25e-9
    h_max: 200e-9
    guard_layers: 2
```

Keep `mesh_nx` and `mesh_ny` supported for the current uniform path.

## TODO

- [x] Browser: add tensor material accessor for `eps_r_xx`, `eps_r_yy`,
      `eps_r_xy`.
- [x] Browser: replace scalar element stiffness with tensor stiffness.
- [x] Browser: update `permittivityModel` text to distinguish scalar vs tensor.
- [x] Browser tests: isotropic tensor equals scalar capacitance within numerical
      tolerance.
- [x] Browser tests: high `eps_r_yy` between horizontal plates increases
      capacitance more than high `eps_r_xx`.
- [x] Python: mirror tensor material accessor and stiffness assembly.
- [x] Python tests: mirror the browser tensor regressions.
- [x] Run `npm test`.
- [x] Run `.venv/bin/python -m pytest -q`.
- [x] Browser structured mesh: add nonuniform coordinate generation.
- [x] Browser structured mesh: add boundary coordinate collection from
      rectangles and circle bounds.
- [x] Browser structured mesh: add mesh stats to result object and UI/log
      output.
- [x] Browser mesh tests: verify mandatory boundaries appear in generated
      coordinates.
- [x] Keep local browser server convention on port `5173`.
- [ ] Python structured mesh: mirror nonuniform coordinate generation.
- [ ] Mesh tests: verify refined BTO/TFLN examples converge with fewer nodes
      than brute-force uniform refinement.
- [ ] Add one refined example YAML or documented snippet for BTO/TFLN.
- [ ] Decide whether refinement defaults should be example-specific or UI
      controlled.

## Not Doing Yet

- Full constrained Delaunay triangulation.
- Geometry-conforming electrode surfaces.
- Boundary-edge charge integration.
- Web Worker migration.
- WebGPU acceleration.

Those remain valuable, but they should follow after tensor assembly and
structured refinement convergence tests expose the dominant accuracy and
performance behavior.
