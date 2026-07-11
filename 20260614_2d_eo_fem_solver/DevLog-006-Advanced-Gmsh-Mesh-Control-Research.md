# DevLog 006: Advanced Gmsh Mesh Controls And 3D Example Research

Date: 2026-07-10

## Decision Context

The project now loads and displays a tagged Gmsh 2D mesh, but the generator is
still a single global-size operation. The next decision is how to add local
refinement, complex multi-domain examples, 3D meshes, and a scalable mesh UI
without hard-coding Gmsh details directly into frontend controls.

COMSOL is the UX reference. The goal is not to clone every COMSOL feature. The
useful pattern is an ordered mesh sequence containing generator operations and
selection-scoped attributes, with build state, warnings, statistics, and
quality inspection.

Primary-source claims and retrieval metadata are cached in
`references/meshing/source-index.json`. This DevLog contains paraphrased
evidence and implementation conclusions, not copied documentation.

## Executive Answer

Yes, Gmsh supports local refinement by surfaces and domains, primarily through
mesh-size fields rather than a COMSOL-style `Size` node:

- `Restrict` applies an input field to selected points, curves, surfaces, or
  volumes.
- `Distance` plus `Threshold` refines near selected curves or OpenCASCADE/
  discrete surfaces with a controlled transition distance.
- `Box`, `Ball`, `Cylinder`, `MathEval`, `Constant`, and other fields define
  spatial target sizes.
- `Min` combines several local controls into one background size field.
- Curvature and point sizes can contribute additional constraints.
- Boundary sizes can be extended into surfaces/volumes, although explicitly
  field-driven workflows often disable automatic extension to avoid accidental
  over-refinement.

Important distinction: `gmsh.model.mesh.refine()` is documented as uniform
refinement of the current model mesh. It is not the equivalent of COMSOL's
selection-scoped Refine operation. For targeted Gmsh refinement, rebuild the
mesh using selection-scoped/background fields. Post-solution adaptation can use
a `PostView`/background metric workflow, but that is a separate later feature.

## Primary Evidence

### Gmsh 4.15.2

Official manual: <https://gmsh.info/doc/texinfo/>

Verified behavior:

1. Gmsh computes local target size from point sizes, curvature sizing,
   background fields, and per-entity constraints, then clamps it using global
   min/max and a scale factor.
2. Tutorial `t10` demonstrates general mesh-size fields and combining fields.
3. `Restrict` supports `PointsList`, `CurvesList`, `SurfacesList`, and
   `VolumesList`, plus boundary/embedded-entity inclusion.
4. `Distance` supports curves and, for OpenCASCADE/discrete geometry, surfaces.
5. `Extend` propagates sizes from curves into surfaces or surfaces into volumes.
6. `setSize` currently handles point entities only; arbitrary surface/volume
   sizing should not be modeled as a direct `setSize(surface)` API call.
7. `model.mesh.refine()` uniformly splits the current mesh.
8. Tutorial `t3` shows layered extrusion into 3D; recombination can create
   prisms, hexahedra, or pyramids.
9. Tutorial `t17` shows anisotropic background metrics in 2D using BAMG.
10. Gmsh has a 2D `BoundaryLayer` field. Extruded boundary-layer geometry is
    another option, but Gmsh's corner treatment is not a drop-in equivalent of
    COMSOL's mature boundary-layer workflow.

Local installed-version smoke test:

- Built two conformal fragmented box volumes with Gmsh Python 4.15.2.
- Applied a coarse `Constant` field everywhere and a finer `Constant` field
  restricted to one `VolumesList`, combined using `Min`.
- Target-volume mean sampled element-edge length: `0.08975`.
- Other-volume mean sampled element-edge length: `0.14372`.
- Fine/coarse edge ratio: `0.625`; target and other volume element counts were
  7,286 and 1,339, respectively.
- This verifies that selection-scoped volume sizing works in the installed
  version; it is not only a manual-level capability claim.

### COMSOL 6.3

Official references:

- Mesh node: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.11.html>
- Operations and attributes: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.41.html>
- Size: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.86.html>
- Refine: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.81.html>
- Refine versus Adapt: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.22.html>
- Boundary Layers: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.43.html>
- Free Tetrahedral: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.65.html>
- Element quality: <https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_mesh.24.24.html>

Verified UX concepts worth adopting:

- Physics-controlled versus user-controlled mesh sequences.
- An ordered tree of generator operations and local attributes.
- Named-selection-based targeting at domain/boundary/edge/point level.
- A global Size node plus more specific local Size attributes.
- Separate generators: free triangle/quad, mapped, free tet, swept, and boundary
  layer.
- Local size parameters: maximum/minimum size, growth, curvature, and narrow
  region resolution.
- Distribution controls along edges and sweep directions.
- Post-mesh Refine versus solution/error-driven Adapt.
- Information/warning/error nodes attached to the operation that caused them.
- Mesh statistics, element-quality plots, and low-quality-element localization.

## COMSOL-To-Gmsh Capability Mapping

| COMSOL concept | Gmsh mechanism | Fidelity / caveat |
|---|---|---|
| Global Size | global Constant/background field plus `MeshSizeMin/Max` | Strong |
| Size on domain | `Constant` or other field restricted with `SurfacesList` in 2D or `VolumesList` in 3D | Strong |
| Size on boundary | `Restrict`, or `Distance` + `Threshold` for adjacent transition | Strong, but propagation semantics must be explicit |
| Size on edge/point | curve/point restriction, point size, Distance field | Strong |
| Growth rate | Threshold transition, field composition, smoothing options | Approximate; no single identical parameter |
| Curvature factor | `Mesh.MeshSizeFromCurvature` | Similar global mechanism; local composition needs care |
| Narrow-region resolution | distance/medial-feature fields or explicit geometry controls | Must be built; no direct one-knob equivalent |
| Free Triangular | default 2D algorithms | Strong |
| Free Tetrahedral | 3D algorithms | Strong |
| Mapped | transfinite curves/surfaces plus recombination | Topology requirements differ |
| Swept | extruded mesh with `Layers`, optionally `Recombine` | Strong for extrusion-compatible geometry |
| Boundary Layers | 2D BoundaryLayer field or geometric extrusion | Partial relative to COMSOL corner/control UX |
| Local Refine after mesh | no direct selection-scoped equivalent; remesh using fields | Gap |
| Uniform Refine | `model.mesh.refine()` | Strong |
| Adapt | `PostView` or metric background mesh in an iterative solve-estimate-remesh loop | Feasible, not yet integrated |
| Quality plot | Gmsh quality APIs/plugins plus frontend metrics | Feasible; define metric contract first |

## Proposed Internal Mesh Model

Do not expose raw Gmsh field IDs in the project config. Use backend-neutral
mesh operations and compile them to Gmsh:

```yaml
MeshSequences:
  mesh1:
    dimension: auto
    sequence_type: user_controlled
    operations:
      global_size:
        type: size
        selection: entire_geometry
        h_max: 8e-7
        h_min: 5e-8
        growth_rate: 1.35
        curvature_resolution: 24

      electrode_proximity:
        type: distance_threshold
        selection: electrode_boundaries
        h_min: 3e-8
        h_max: 8e-7
        distance_min: 1e-7
        distance_max: 1.5e-6

      core_domain_size:
        type: size
        selection: optical_core_domains
        h_max: 4e-8
        include_boundary: true

      volume_mesh:
        type: free_tetrahedral
        selection: all_domains

      final_refine:
        type: refine
        selection: hotspot_domains
        levels: 1
        enabled: false
```

Compilation rules:

1. Resolve named selections to stable geometry entity tags after OCC Boolean
   fragmentation.
2. Compile each `size`/proximity operation into a Gmsh field.
3. Combine active size fields with `Min` and set one background field.
4. Set point/curvature/boundary-extension contributions explicitly rather than
   relying on Gmsh defaults.
5. Execute generator operations in tree order.
6. Treat selection-scoped `refine` as remeshing with a smaller local target
   size, not as `model.mesh.refine()`; reserve `uniform_refine` for the Gmsh API.
7. Record the compiled Gmsh operation/field graph in the artifact manifest.

## Proposed UI

Model Builder:

```text
Mesh 1
  Sequence: User controlled
  Size 1 (Entire geometry)
  Size 2 (optical_core_domains)
  Boundary Proximity 1 (electrode_boundaries)
  Free Triangular 1 / Free Tetrahedral 1
  Refine 1 [disabled]
  Statistics
  Warnings
```

Settings panel for each node:

- Selection: entire geometry, named selection, manual selection.
- Entity level: domain, boundary, edge, point.
- Size: preset or custom `h_min`, `h_max`, growth, curvature.
- Proximity: near/far sizes and transition distances.
- Build Selected, Build Up To Here, Build All, Clear Mesh.
- Enable/disable, duplicate, reorder, rename.
- Generated-element count estimate and actual statistics.

Graphics interactions:

- Geometry mode picks entity IDs and creates named selections.
- Mesh mode colors elements by domain, physical group, element type, size, or
  quality.
- Clipping plane and slice controls are mandatory for 3D volume meshes.
- Low-quality histogram selection highlights corresponding elements.
- LOD rendering remains separate from actual mesh topology/statistics.

## Proposed Example Ladder

### Advanced 2D

| Example | Controls exercised | Validation |
|---|---|---|
| Parallel plate with proximity field | Distance + Threshold at electrode boundaries | Element size versus distance; capacitance convergence |
| Two cylinders | Curvature plus boundary proximity | Curved boundary error and analytic capacitance |
| Layered dielectric stack | Explicit fragmented material domains; per-domain size | Interface conformity and series capacitance |
| TFLN partial-etch modulator | Multiple domain sizes; electrode/core proximity; Min composition | Physical tags, gap resolution, mesh convergence |
| Sharp electrode corner | Box/Distance field and growth control | Minimum angle, growth ratio, field singularity convergence |
| 2D boundary-layer demonstration | BoundaryLayer field along selected curves | Layer count, first height, growth, corner behavior |
| Transfinite directional mesh | Transfinite curves/surface and recombination | Structured topology and anisotropic directional resolution |

### 3D

| Example | Controls exercised | Validation / purpose |
|---|---|---|
| Unit cube with spherical inclusion | OCC fragmentation, tagged volumes/interfaces, free tetrahedral, curvature | Volume conservation, interface conformity, element quality |
| Coaxial cylinder segment | Curved conductor/dielectric volumes, radial proximity | Analytic capacitance per length after solver support |
| Extruded parallel-plate capacitor | Layered extrusion, prisms/hexes, source/destination faces | 2D-to-3D consistency and swept topology |
| TFLN waveguide/electrode segment | Multi-volume extrusion, local core/gap refinement | Project-relevant 3D tagging and scale disparity |
| Via and pad transition | Cylinder/box Boolean fragments, surface proximity, local volume size | Small-feature and conductor-interface stress test |
| Twisted extrusion | Layer distribution, sweep/recombine | Direct analogue of Gmsh tutorial `t3` |
| Boundary-layer duct | Selected walls, layered prisms, tet interior | Boundary-layer transition and mixed elements |

## 3D Visualization Decision

The current 2D canvas renderer must not be stretched into a 3D volume viewer.
Use one of these stages:

1. Generate/export 3D MSH/VTU and inspect in Gmsh/ParaView immediately.
2. Extend the artifact parser to tetrahedron/prism/hex/pyramid topology and
   provide surface extraction plus clipping metadata.
3. Add `vtk.js` for interactive 3D surfaces, clipping planes, slices, picking,
   and scalar fields. Keep the project-specific Model Builder/settings UI owned
   by this application.

## Proposed Milestones

### M1: Mesh-Control Schema And Compiler

- Add backend-neutral operation dataclasses/schema.
- Implement global size, selection-scoped size, Distance + Threshold, and Min.
- Persist compiled field graph and resolved entity tags.
- Add deterministic tests on target sizes and physical groups.

### M2: Complex 2D Examples

- Add parallel-plate proximity, cylinder curvature, fragmented stack, TFLN,
  sharp-corner, and transfinite examples.
- Add mesh statistics and convergence harness.

### M3: Frontend Mesh Sequence Editor

- Add operation tree, settings panel, enable/reorder/build controls, and named
  selection targeting.
- Add size/quality coloring and warning nodes.

### M4: Basic 3D Gmsh Artifacts

- Add sphere-in-cube, coax, extruded capacitor, and TFLN segment generators.
- Export MSH 4.1 plus VTU and tag manifest.
- Validate in Gmsh/ParaView before in-app 3D rendering.

### M5: In-App 3D Viewer

- Add volume topology normalization and boundary-face extraction.
- Integrate vtk.js with clipping/slices/picking.
- Keep large topology parsing in workers and use binary/transferable arrays.

### M6: Adaptation And Physics-Controlled Presets

- Add wavelength/gap/curvature-based physics presets.
- Add solution-error/field-gradient-driven PostView remeshing loop.
- Separate physics-controlled generation from solver-error adaptation, matching
  the distinction in COMSOL.

## Risks And Open Questions

- Stable selections require robust OCC fragment history; bounding-box matching
  is insufficient for a production geometry tree.
- A COMSOL-like growth-rate field will need a defined approximation and tests;
  Gmsh does not expose identical semantics as one parameter.
- 3D boundary layers and mixed elements need solver-space compatibility checks,
  especially for MFEM H(curl).
- High-order curved elements require parser, visualization, and MFEM-order
  alignment; corner-only visualization is insufficient for verification.
- Mesh quality thresholds depend on element family and physics. Aspect ratio is
  not automatically bad for intentional boundary-layer elements.
- Browser JSON with nested arrays will not scale to million-element 3D meshes;
  use binary VTU or a compact typed-array artifact with transferables.

## Implementation Progress

### 2026-07-10: Mesh Sequence Slice 1

Implemented:

- Ordered backend-neutral mesh sequence schema.
- COMSOL-style named operation nodes with `type`, `selection`, `enabled`, and
  operation parameters.
- Validation for dimensional generator compatibility and size/transition
  bounds.
- Gmsh compiler for global/local `size`, `boundary_proximity`, `box_size`,
  `curvature`, `free_triangular`, `free_tetrahedral`, and `uniform_refine`.
- Explicit `Min` composition into one background field.
- Compiled operation and Gmsh field provenance in the tag-map artifact.

Generated so far:

- Parallel-plate boundary-proximity mesh: 2,316 nodes.
- Two-cylinder curvature/proximity mesh: 914 nodes.
- Sharp-corner box-refined mesh: 2,423 nodes.
- 3D sphere-in-box: 2,324 nodes and 13,033 linear tetrahedra; mean `minSICN`
  quality `0.772`, minimum `0.055`.
- 3D swept capacitor: 1,248 nodes and 1,872 linear prisms; mean `minSICN`
  quality `0.909`, minimum `0.828`.

Bug found:

- The first sharp-corner example failed to recover `signal_boundary` because
  legacy OCC entity recovery still matches micron-scale curve bounding boxes.
  The tolerance was increased from 2.5% to 5% of feature span as an immediate
  fix. This is not considered production-stable; OCC Boolean history and named
  selections must replace bounding-box recovery.
- The first HTTP `POST /api/mesh/generate` failed because Gmsh attempted to
  install signal handlers from the request worker thread. Backend Gmsh
  initialization now uses `interruptible=False`; generation is serialized with
  a process-local lock because Gmsh's global model state is not thread-safe.
- The first automated ParaView screenshot attempt used camera methods exposed
  by older/simple examples directly on the render-view proxy. ParaView 6.0 RC2
  requires camera transforms through `GetActiveCamera()`; the inspection
  script was updated accordingly.
- Screenshot inspection of the 7,282-triangle multi-electrode case found black
  holes in Mesh view. LOD sampling was filling only every second triangle,
  visually implying missing elements. Filled size maps are now allowed only at
  stride 1, and the full-detail budget increased to 10,000 triangles; decimated
  views use edge-only rendering.

Validation:

- A quantitative parallel-plate test compares triangle edges near electrodes
  with far-field edges and requires the near-field mean to be less than 55% of
  the far-field mean.
- 3D tests require tagged volumes/faces, expected tetrahedron/prism families,
  mean tetrahedral quality above `0.7`, and swept-prism minimum quality above
  `0.8`.
- The low minimum `minSICN` in the sphere-in-box mesh is retained as a quality
  investigation case rather than hidden by the mean.

## Immediate Recommendation

Implement M1 and the first four 2D examples before building the 3D viewer.
In parallel, generate the first 3D artifacts and validate them externally. The
highest-risk prerequisite is stable named-selection-to-OCC-entity mapping, not
the tetrahedral mesh call itself.

## Validation Inspection And Optimization Pass

Date: 2026-07-10

### Backend API

Added to the development server:

- `GET /api/health`
- `GET /api/examples` for config and generated-artifact discovery
- `POST /api/mesh/generate` for controlled server-side generation
- Static artifact URLs returned directly by the generation response

Generation is restricted to configs under `examples/`, writes under
`artifacts/api/`, sanitizes output names, and runs in an isolated subprocess.
Direct API generation was exercised with the parallel-plate proximity config:
2,316 nodes and 4,172 triangles were generated and returned by HTTP.

The frontend now provides a backend-populated validation `Library`, supports
deep-linked artifacts through `?artifact=...`, caches parsed artifacts in
memory, and logs fetch/parse/total timing separately.

### Frontend Screenshot Inspection

Screenshots are cached under `artifacts/inspection/screenshots/`.

| Example | Load time after optimization | Min / mean triangle quality | Inspection result |
|---|---:|---:|---|
| Parallel plate proximity | 30 ms | 0.673 / 0.939 | Smooth refinement around both electrode boundaries; tags correct |
| Two cylinders curvature | 13 ms | 0.652 / 0.940 | Curvature and proximity refinement are symmetric and conformal |
| Sharp corner box | 24 ms | 0.606 / 0.945 | Local box is effective but transition is intentionally abrupt |
| Multi-electrode hotspots | 34–46 ms | 0.579 / 0.940 | Four physical boundaries recovered; circle and lower-gap refinements coexist |

Mesh view now colors by logarithmic mean edge length instead of arbitrary
vertical position. It reports min/mean/max edge length and normalized triangle
quality.

### Load-Time Optimization

Initial screenshot measurements were approximately 519–532 ms of parse time
even for 74–210 KB meshes. Root cause: module-worker startup dominated small
files. Current policy:

- Files up to 1 MB parse synchronously; measured end-to-end loads fell to
  13–46 ms for validation meshes.
- Larger files remain off-main-thread in the mesh worker.
- Repeated backend-library loads reuse an in-memory parsed-artifact cache.
- Full-detail filled rendering is capped at 10,000 visible triangles.
- Larger views use edge-only deterministic LOD; zoom increases local detail.

Screenshot inspection caught and fixed an LOD correctness bug: filling only
sampled triangles created black holes that looked like missing elements.

### 3D Inspection

Gmsh MSH files are converted to legacy VTK for reproducible clipped ParaView
screenshots using `scripts/render_mesh_3d.py`. Quality evidence is generated by
`scripts/inspect_mesh_quality.py`.

| Example | Volume elements | minSICN summary | Inspection result |
|---|---:|---|---|
| Sphere in box | 13,033 tets | min 0.055; p01 0.378; mean 0.772 | Conformal spherical interface and correct local refinement; one pole sliver remains |
| Swept capacitor | 1,872 prisms | min 0.828; p01 0.830; mean 0.909 | Consistent 12-layer swept topology and excellent quality distribution |
| Coaxial segment | 18,755 tets | min 0.224; p01 0.365; mean 0.776 | Correct annular topology and strong inner-conductor refinement |

The sphere's worst element centroid is approximately
`(0.042, -0.024, 0.446)`, on the inclusion pole. This should become a quality
warning/highlight test for the future frontend quality inspector.

### Bugs Found During Inspection

- Gmsh initialization from the threaded API initially attempted to install
  main-thread-only signal handlers. Fixed with `interruptible=False` and a
  global generation lock.
- ParaView 6 RC2 camera methods differ from older examples. Rendering now uses
  `GetActiveCamera()`.
- Small-file worker startup added about 0.5 seconds. Fixed with size-based
  synchronous/worker routing.
- LOD fill sampling produced false holes. Fixed by allowing fill only at stride
  one and using edge-only rendering for decimated views.
- The multi-electrode OCC cut emits `BOPAlgo_AlertNotSplittableEdge`; output
  topology and physical groups validate, but OCC tolerance/history remains an
  open production risk.
- Combined API/3D regression testing triggered a native OpenMP abort after
  Gmsh was initialized/finalized in an HTTP worker and then reused later in the
  same interpreter. A Python lock did not protect against runtime contamination.
  API generation now runs each Gmsh job in a subprocess with a 120-second
  timeout; native failure is isolated from the long-running server.

## Artifact Version-Control Policy

Date: 2026-07-11

Generated files under `artifacts/` are excluded by the project-local
`.gitignore`. This includes generated Gmsh/VTK meshes, API job outputs,
inspection screenshots, and derived quality reports. These outputs currently
occupy about 26 MB and are reproducible from the tracked example configs,
backend generators, and inspection scripts.

Tracked inputs and evidence metadata remain outside `artifacts/`:

- `examples/mesh_controls/*.yaml`
- mesh-generation and inspection code under `eo_fem/` and `scripts/`
- automated validation under `tests/` and `tests-js/`
- `references/meshing/source-index.json`

If a generated file later becomes a required regression fixture, move a
minimal deterministic copy to a dedicated `tests/fixtures/` path rather than
adding a broad exception under `artifacts/`. Large golden meshes or screenshot
baselines should use Git LFS or external release/object storage if introduced.

### Milestone / TODO Update

- [x] Define and apply a project-scoped generated-artifact policy.
- [x] Keep reproducible source configs and validation tooling tracked.
- [ ] Remove already staged `artifacts/` files from the Git index before the
  next commit; ignore rules do not automatically unstage files.
- [ ] Add a clean-checkout regeneration smoke test for the example library.
