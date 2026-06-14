# DevLog 000: 2D EO FEM Solver Planning

Date: 2026-06-14

## Goal

Build a lightweight 2D electrostatic finite-element solver for estimating:

- Capacitance per unit length, `C'` or `C_per_length`, of EO modulator electrode cross-sections.
- Modulation efficiency, `VpiL`, for thin-film Pockels modulators.
- Geometry/material trends for TFLN, TFLT, BTO, and related integrated EO platforms.

The project is intended as an engineering estimator, not a replacement for COMSOL, Ansys, or foundry-qualified extraction.

## Intended Use Cases

- Compare electrode gap, signal/ground width, metal thickness, oxide thickness, and substrate stack.
- Estimate whether a lumped or segmented modulator can be charged directly by a low-voltage electrical source.
- Produce quick `VpiL` and `fF/mm` tradeoff plots for different EO materials and cuts.
- Generate reproducible cross-section studies before moving a small number of cases into commercial multiphysics tools.

## Physical Model

### Electrostatics

Solve the 2D anisotropic electrostatic problem:

```text
div(epsilon_r tensor * epsilon_0 * grad(phi)) = 0
E = -grad(phi)
```

Boundary conditions:

```text
signal electrode: phi = 1 V
ground electrodes: phi = 0 V
outer boundary: far-away Neumann or grounded box, tested by convergence
```

Capacitance per unit length:

```text
Q' = integral_boundary D . n dl
C' = Q' / V
```

Unit conversion:

```text
1 pF/cm = 100 fF/mm
```

### EO Overlap And VpiL

For a phase modulator:

```text
Delta n ~= -0.5 * n^3 * r_eff * E_RF
Delta phi = (2 pi / lambda) * integral(Delta n weighted by optical mode) dz
VpiL = pi * V * L / Delta phi
```

For the first implementation, use one of two levels:

1. Fast estimator: approximate optical mode as a Gaussian or imported scalar field.
2. Better estimator: import optical mode data from an external mode solver and compute RF-optical overlap.

## Materials To Support

Start with scalar permittivity and scalar `r_eff`, then add tensor orientation.

| Material | First-pass RF permittivity | Typical EO coefficient to expose |
|---|---:|---|
| LiNbO3 / TFLN | `eps_r ~ 28-43`, orientation-dependent | `r33`, `r13`, `r22`, effective `r_eff` |
| LiTaO3 / TFLT | similar workflow; use literature values per cut | `r33`, effective `r_eff` |
| BaTiO3 / BTO | high and anisotropic; film-quality dependent | large effective Pockels terms, orientation-dependent |
| SiO2 | `eps_r ~ 3.9` | none |
| Air | `eps_r ~ 1.0` | none |
| Si substrate | dielectric, lossy, or grounded conductor mode | none |

Keep material parameters in a versioned YAML or JSON file with source notes.

## Candidate Software Stack

Browser-first product path:

- TypeScript or modern JavaScript for the interactive solver core.
- HTML canvas for field, mesh, and material-property visualization.
- Web Worker execution for non-blocking solves once mesh sizes grow.
- Keep the config/results contract browser-native so a later Rust/WASM or WebGPU backend can replace the numerical kernel without changing user workflows.
- Python remains useful as a reference harness for offline validation, sweeps, and comparisons against commercial solvers, but should not be required for the main interactive experience.

Reference/offline path:

- Python.
- `gmsh` for geometry and meshing.
- `scikit-fem` for FEM assembly and solves.
- `numpy`, `scipy`, `matplotlib`.
- `pydantic` or dataclasses for structured simulation input.

Alternatives:

- `FEniCSx`: stronger PDE framework, heavier install.
- `pygmsh` + `meshio`: convenient but extra dependency layer.
- `shapely`: useful for geometry composition, not required for v0.

## Proposed Repository Shape

```text
.
├── DevLog-000-2D-EO-FEM-Solver-Planning.md
├── README.md
├── pyproject.toml
├── examples/
│   ├── tfln_gsg_baseline.yaml
│   ├── tflt_gsg_baseline.yaml
│   └── bto_slot_baseline.yaml
├── eo_fem/
│   ├── geometry.py
│   ├── materials.py
│   ├── electrostatic.py
│   ├── capacitance.py
│   ├── overlap.py
│   ├── sweep.py
│   └── plotting.py
├── notebooks/
│   └── 001_tfln_capacitance_sweep.ipynb
└── tests/
    ├── test_parallel_plate.py
    ├── test_coaxial_reference.py
    └── test_mesh_convergence.py
```

## Minimum Viable Solver

1. Create rectangular multilayer cross-sections: air, oxide, EO film, buried oxide, substrate.
2. Add rectangular signal and ground electrodes.
3. Mesh with local refinement near electrode gaps and material interfaces.
4. Solve `div(eps grad(phi)) = 0`.
5. Integrate charge around the signal electrode.
6. Report `C'` in `pF/cm` and `fF/mm`.
7. Sweep electrode gap and EO-film thickness.
8. Validate against analytic parallel-plate and coaxial/fringing reference cases.

## Current Prototype Status

As of 2026-06-14, the repository has a browser-first runnable prototype plus a Python reference path.

Implemented browser path:

- Static browser app under `web/`, served by `python3 -m http.server 5173`.
- Client-side MOOSE-like YAML loading/editing.
- Cache-busted example YAML fetches so clicking an example button reloads current disk contents rather than stale browser cache.
- Structured triangular mesh:
  - `nx` by `ny` Cartesian node grid over `Domain`.
  - Each rectangular cell is split into two P1 triangles.
  - Electrodes are currently imposed by pinning mesh nodes inside conductor shapes to Dirichlet potentials.
- Scalar-permittivity electrostatic solve:
  - P1 triangle stiffness assembly.
  - CSR typed-array sparse storage.
  - Conjugate-gradient solve on free nodes.
  - Energy-method and electrode-node charge capacitance estimates.
  - Current assembly is homogeneous: it uses `Materials.background.eps_r` only.
    Non-background material regions are not yet included in the solve matrix.
- Analytic validation cases:
  - Parallel-plate reference.
  - Two-cylinder reference.
- Field/material visualization:
  - `phi`, `Ex`, `Ey`, `|E|`.
  - Diagnostic material/EO-property maps: `epsilon_r`, `epsilon_r_xx`, `epsilon_r_yy`, `epsilon_r_xy`, `r13`, `r33`, `r22`, `r_eff`.
  - Material boundary outlines over field/material plots.
  - Linear, symmetric linear, and log-magnitude color scaling.
  - Colorbar, mesh overlay, electrode overlay, quantity descriptions and expressions.
- Canvas interaction:
  - Drag pan.
  - Mouse-wheel zoom.
  - Double-click or Reset view to restore domain view.
  - Resize-aware redraw.
  - Hover tooltip with `x`, `y`, selected value, and expression/description.
- UI/runtime behavior:
  - Dark theme and sharp-corner controls.
  - Resizable editor/output panes.
  - Left pane bounded to viewport; YAML and log scroll internally.
  - Stable right-pane layout so canvas does not jump during solve/progress/error states.
  - In-canvas WIP state while the solver is running.
  - Timestamped log panel for validation, solve start, solve finish, and errors.
  - Config validation before solving: finite numeric values, positive domain dimensions, valid mesh size, positive permittivity, valid conductor geometry, valid analytic-reference dimensions.

Implemented reference path:

- Python package `eo_fem` mirrors the simple structured scalar solver.
- Pytest reference tests remain available for cross-checking.

Current limitations:

- The browser and Python solvers still use scalar background `eps_r` in assembly; material-property maps and material-boundary outlines are diagnostic only.
- Electrode and material boundaries are not geometry-conforming.
- Charge extraction is still based on conductor-node residuals, not tagged boundary-edge integration.
- The solve still runs on the main browser thread; progress is a status/WIP indicator, not true iteration-level progress.
- No Web Worker, preconditioner, WebGPU, true unstructured mesh, anisotropic tensor assembly, or EO overlap yet.
- Python tests and browser tests are separate; browser tests are the primary product-path tests.

Immediate next implementation target:

1. Add spatially varying scalar permittivity to browser assembly.
   - Parse material regions once with existing `parseMaterials`.
   - For each P1 triangle, evaluate the material at the triangle centroid.
   - Assemble local stiffness with `epsilon_0 * eps_r_centroid`.
   - Keep `eps_r_xx`, `eps_r_yy`, and `eps_r_xy` as diagnostic placeholders until tensor assembly is implemented.
2. Mirror the same centroid-based scalar assembly in the Python reference path.
3. Add regression tests:
   - Homogeneous material case remains unchanged.
   - A high-`eps_r` slab between electrodes increases capacitance relative to low-`eps_r` background.
   - Material ordering/overlap behavior follows the existing "later non-background material wins" map rule.
4. Update result/log text to make clear whether a run used homogeneous or spatially varying scalar permittivity.

## Browser Visualizer Requirements

The browser UI should expose enough numerical internals that the user can debug whether a result is physically plausible before trusting scalar metrics.

- Mesh view:
  - Show the generated mesh as an overlay on top of field/material plots.
  - For the first structured-mesh backend, document that the mesh is an `nx` by `ny` Cartesian node grid with each rectangular cell split into two P1 triangles.
  - Make conductor regions visible separately from field color so it is obvious which mesh nodes are pinned by Dirichlet boundary conditions.
- Field views:
  - Plot electrostatic potential `phi`.
  - Plot RF field components `Ex = -d phi / dx`, `Ey = -d phi / dy`.
  - Plot field norm `|E| = sqrt(Ex^2 + Ey^2)`.
  - Provide color scaling options:
    - linear automatic min/max,
    - symmetric linear scale around zero for signed quantities such as `Ex` and `Ey`,
    - logarithmic magnitude scale for high-dynamic-range `|E|` views.
  - Show a colorbar with numeric min/max or mapped range.
- Material-property views:
  - Plot scalar `epsilon_r` for isotropic material definitions.
  - Reserve UI/config names for tensor components such as `epsilon_r_xx`, `epsilon_r_yy`, `epsilon_r_xy` once anisotropic assembly is implemented.
  - Reserve property-map slots for EO coefficients such as `r13`, `r33`, `r22`, and `r_eff`.
  - Material maps are diagnostic overlays first; solver support for anisotropic tensors and EO overlap can come later.
- Implementation constraint:
  - Keep visualization and result objects browser-native.
  - Avoid making Python a runtime dependency for the interactive browser UI.

## Next Implementation Milestones

| Milestone | Output |
|---|---|
| M0 | Static repo scaffold, installable Python package, one example YAML |
| M1 | Scalar-permittivity electrostatic solve with capacitance extraction |
| M2 | Mesh convergence harness and analytic tests |
| M3 | Browser visualizer for mesh, `phi`, `Ex`, `Ey`, `|E|`, colorbar, and material-property maps |
| M4 | Stable browser UX: validation, logging, pan/zoom/hover, cache-busted example reloads, stable layout |
| M5 | Web Worker solver execution and real iteration-level progress without blocking the UI |
| M6 | Spatially varying scalar permittivity in assembly; next task |
| M7 | Nonuniform structured mesh refinement near electrode/material edges |
| M8 | True geometry-conforming FEM mesh with region/boundary tags |
| M9 | Anisotropic permittivity tensors with material orientation |
| M10 | Approximate optical-mode overlap and `VpiL` estimator |
| M11 | Parameter sweeps and plots: `C_per_length`, `VpiL`, RC-limited length |
| M12 | Imported optical mode support from external solvers |

## Key Checks

- Outer boundary convergence: grow simulation box until `C'` changes by less than target tolerance.
- Mesh convergence: refine electrode edges, gaps, and EO/oxide interfaces.
- Charge consistency: compare boundary charge integral against energy method:

```text
U' = 0.5 * integral epsilon |E|^2 dA
C' = 2 U' / V^2
```

- Units: enforce SI internally; print photonics-friendly units at output.
- Sign and orientation: explicitly document crystal axes, optical polarization, RF field direction, and selected EO tensor element.

## Representative Public References

- Wang et al., "Integrated lithium niobate electro-optic modulators operating at CMOS-compatible voltages", Nature, 2018. Demonstrates low-voltage integrated LN modulators and motivates compact Pockels platforms.
- Zhang et al., "Monolithic ultra-high-Q lithium niobate microring resonator", Optica, 2017, and related TFLN platform papers. Useful background on thin-film LN integrated photonics.
- He et al., "High-performance hybrid silicon and lithium niobate Mach-Zehnder modulators for 100 Gbit/s and beyond", Nature Photonics, 2019. Representative high-speed hybrid LN modulator context.
- Zhu et al., "Integrated electro-optics on thin-film lithium niobate", arXiv:2404.06398 / review-style tutorial. Useful for modulator metrics, traveling-wave electrodes, impedance, and measurement caveats.
- Ansys Optics example, "Thin-Film Lithium Niobate Electro-Optic Phase Modulator". Useful public workflow reference for coupled electrical and optical EO simulation.
- Abel et al., "A strong electro-optically active lead-free ferroelectric integrated on silicon", Nature Materials, 2019, and follow-on BTO-on-silicon modulator literature. Representative BTO platform context.
- Eltes et al., "A BaTiO3-based electro-optic Pockels modulator monolithically integrated on an advanced silicon photonics platform", Journal of Lightwave Technology, 2020. Reports strong BTO modulation efficiency and high-speed operation.
- Posadas et al. / related works on "Electro-optic barium titanate modulators on silicon photonics platform". Useful for BTO orientation, effective Pockels coefficients, and `VpiL` benchmarks.
- Recent thin-film lithium tantalate EO modulator papers and preprints on DC-stable or UV/visible TFLT modulators. Use as early context, but verify material constants and wavelength dependence before encoding defaults.

## Notes For The Implementation Agent

- Keep the first version deliberately simple: scalar permittivity, rectangular geometry, capacitance extraction.
- Do not overfit the code to one material. The core solver should accept arbitrary material tensors and electrode polygons.
- Treat `VpiL` as a separate layer above electrostatics; it requires optical-mode assumptions.
- Store every default material constant with a citation or source note.
- Prefer reproducible sweeps and tests over GUI features.
