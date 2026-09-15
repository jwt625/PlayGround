# 002 — Rendering assets and external resource cache

Date: 2026-09-15
Status: asset generation, source collection, and initial validation completed as detailed below; application integration remains unverified in this workstream.

Follows [001 — optics/rendering specification](001-cbc-optics-rendering-spec.md). Active checklists remain in [asset tasks](../docs/ASSET_TASKS.md) and [coding tasks](../docs/CODING_TASKS.md). This log records delivered artifacts and evidence; it does not mark whole implementation phases complete.

## Summary

- Collected the canonical FlyBody model and generated an articulated GLB with an illustrative wing animation.
- Generated five reusable optical-bench display prefabs, exact 19-channel geometry, style defaults, and an art-direction concept.
- Cached actual MaleCNS v1.0 connectivity, annotations, neurotransmitter predictions, and two example neuron skeletons.
- Prepared an explicitly filtered graph with **164,606 nodes and 25,558,671 connection rows**.
- Validated downloads, export transforms, browser asset loading, and derived-file structure. Runtime use of these assets/data, learning, and CBC physics were not validated by this asset pass.

## 1. FlyBody source and articulated rendering asset

Source: [TuragaLab/flybody](https://github.com/TuragaLab/flybody), pinned to `d015e9bfe441bd90ae431bac24c55cb74bdbce26`. Upstream asset directory, Apache-2.0 license, README, and build script are cached under `assets/vendor/flybody/`. The refreshed inventory contains 91 files totaling 161,099,048 bytes; the initial collection had 90 files before the build script was added.

The pinned build script explicitly converts millimeters to centimeters. The exporter uses MuJoCo's compiled model and forward kinematics to resolve defaults, mesh alignment, materials, body transforms, and joints, then converts centimeters to meters.

| Artifact / property | Result |
|---|---|
| Runtime asset | [flybody-articulated.glb](../assets/generated/flybody/flybody-articulated.glb) |
| Export size | 23,005,492 bytes |
| Geometry | 85 visual meshes; 272,550 triangles |
| Hierarchy | 67 body nodes; upstream model has 103 compiled joints |
| Coordinates | Meters, +Y up, +X forward |
| Default pose | Upstream `qpos0` |
| Animation | `illustrative_wing_cycle_1Hz`; baked using MuJoCo joint kinematics |
| Converter | [export_fly_glb.py](../scripts/export_fly_glb.py) |
| Integration metadata | [joint map](../assets/generated/flybody/joint-map.json), [conversion report](../assets/generated/flybody/conversion-report.json) |

Only group-1 mesh visuals are exported; collision proxies are excluded. The wing loop is procedural slowed display motion, not measured flight kinematics or learned locomotion. The earlier inventory's 180 XML joint elements include defaults/other sections and must not be reported as articulated degrees of freedom.

### Validation evidence

- Composed body/geometry transforms agree with MuJoCo world geometry to maximum absolute error `4.44e-16` source centimeters.
- Independent GLB roundtrip through trimesh agrees on bounds within `1.74e-18` meters.
- Three.js 0.169.0 loads 85 meshes and one animation without page errors: [browser report](../assets/generated/flybody/validation/browser-report.json).
- Saved browser [side](../assets/generated/flybody/validation/side.png), [top](../assets/generated/flybody/validation/top.png), [front](../assets/generated/flybody/validation/front.png), and [wing-cycle](../assets/generated/flybody/validation/wing-cycle.png) captures.
- Side silhouette was visually compared with a [native MuJoCo render](../assets/generated/flybody/validation/mujoco-side.png). The shading differs because the exporter approximates materials using PBR.

Outstanding: matched-camera source front/top comparisons, validated resting pose, animation refinement, lower-detail variants after profiling, two-instance performance, and application integration. These checks are separate from the numerical export checks above.

## 2. Optical bench and art-direction assets

| Deliverable | Location | Status / intended use |
|---|---|---|
| Emitter, seed, splitter, phase/amplifier, console GLBs | [bench manifest](../assets/generated/bench/manifest.json) | Original prototype display geometry; named ports and bounds; roundtrip bounds checked |
| Prefab generator | [generate_bench_assets.py](../scripts/generate_bench_assets.py) | Reproducible code-native generation |
| Exact 19-channel layout | [channel-layout.json](../assets/generated/channel-layout.json) | Stable IDs, axial coordinates, positions in pitch units |
| Color/material/path defaults | [render-style.json](../assets/generated/render-style.json) | Initial art defaults, separate from calibrated scientific legends |
| Editable optical-chain reference | [cbc-19-layout.svg](../assets/reference/cbc-19-layout.svg) | Schematic with collapsed channel paths |
| Lab concept and prompt | [concept image](../assets/concepts/cbc-lab-v1.png), [prompt](../assets/concepts/cbc-lab-v1.prompt.txt) | Generated using built-in image generation; composition/material reference |

Bench prefabs use display meters and Z-up; the fly uses meters and Y-up. Integration must apply explicit frame transforms and keep enlarged mechanical display geometry separate from the simulation aperture coordinates. The prefabs do not implement actuator behavior, cable interaction, or vendor CAD dimensions.

The concept image does not reliably reproduce the exact aperture count/pattern. Its beams, dome texture, exaggerated fly scale, and target pose are illustrative. Use generated channel geometry and evaluated complex fields for the actual simulation.

## 3. Actual MaleCNS v1.0 cache

Primary source: [official MaleCNS download page](https://male-cns.janelia.org/download/). License: CC BY 4.0. Source URLs, sizes, SHA-256 hashes, cloud object generations, and available HTTP metadata are recorded in the [retained manifest](../data/manifests/malecns-v1.0.json). Documentation, release notes, and license pages are cached with the source files.

| Cached resource | Size / inspected content |
|---|---|
| `connectome-weights.feather` | 1,051,241,946 bytes; 151,856,684 segment connection rows |
| `annotations.feather` | 14,483,314 bytes; 211,577 annotated bodies |
| `neurotransmitters.feather` | 43,282,834 bytes; predictions, confidence, and consensus fields |
| `12781.swc`, `556329.swc` | 1,155,306 bytes combined; official DNge104 example pair |

Cloud data files passed size and published MD5 checks; SHA-256 hashes are retained locally. Annotation rows include glia, orphan fragments, unimportant segments, and other statuses. Neither annotation-row count nor the raw segment graph is a direct count of identified neurons.

The complete EM volume, all neuron skeletons, synapse-level locations, and database dump have not been cached. The selected resources support the initial graph/controller work and representative anatomy rendering.

### Derived graph

Selection rule:

```text
status == 'Traced' AND superclass IS NOT NULL
retain a connection only if both endpoints pass that selection
```

Results from the downloaded snapshot:

- **164,606 selected nodes**.
- **25,558,671 connection rows**.
- **124,009,893 total raw synapse weight**.

This is a documented subset, not a claim to reproduce the paper's neuron total. No transmitter signs, dynamical normalization, random edges, or learned weights were applied.

`derived/edges.u32` contains little-endian uint32 triples `(pre_index, post_index, raw_synapse_count)`, totaling 306,704,052 bytes. Indices refer to ascending original body IDs in `body-ids.npy` and matching `nodes.feather` rows. Join neurotransmitters by original body ID, never row position.

See [graph report](../assets/generated/connectome/graph-cache-report.json) and [cache instructions](../data/README.md). Raw downloads and derived bulk data live under Git-ignored `data/cache/`; the source manifest remains outside that ignored directory. The cache is local and reproducible, not bundled into a browser build.

### Real neuron rendering assets

[DNge104-skeletons.json](../assets/generated/connectome/DNge104-skeletons.json) preserves body IDs, parent-child connectivity, and SI coordinates for the two cached neurons. SWC coordinates are converted from the documented 8 nm units. One shared centering/scaling transform preserves their relative anatomy.

Browser inspection loaded **33,237 skeleton segments** without page errors: [preview](../assets/generated/connectome/DNge104-preview.png), [report](../assets/generated/connectome/browser-report.json). This depicts two actual neurons, not a complete brain or a validated controller pathway. Native EM axes are retained; anatomical registration into a fly-body mesh remains separate work.

## 4. Reproduction and tooling

Python dependencies are pinned in [requirements-assets.lock.txt](../scripts/requirements-assets.lock.txt); the isolated `.asset-venv/` is Git-ignored.

```sh
uv venv .asset-venv
uv pip install --python .asset-venv/bin/python -r scripts/requirements-assets.lock.txt
python3 scripts/cache_external_resources.py
.asset-venv/bin/python scripts/prepare_connectome_cache.py
python3 scripts/prepare_render_assets.py
.asset-venv/bin/python scripts/export_fly_glb.py
.asset-venv/bin/python scripts/generate_bench_assets.py
.asset-venv/bin/python scripts/render_fly_source.py
```

The cache downloader resumes partial transfers and validates cloud objects. Derived graph preparation streams Arrow batches instead of materializing the full graph as Python objects. Compare retained hashes/generations when refreshing a release path; a release label alone does not establish byte-identical data.

For local asset inspection, serve the repository root:

```sh
python3 -m http.server 8766 --bind 127.0.0.1
# Fly: http://127.0.0.1:8766/assets/preview/
# Neurons: http://127.0.0.1:8766/assets/preview/neurons.html
# Separate terminal, after installing project Node dependencies / Playwright browser:
node scripts/inspect_asset_browser.mjs
```

## 5. Handoff and next TODOs

- [ ] T09: integrate the fly GLB with separate animation mixers for learner and target; finish resting pose and performance checks.
- [ ] T04: connect prefab ports/anchors to actual channel selection, wiring, tip/tilt, and focus behavior.
- [ ] T06: keep intensity, wavefronts, beam envelopes, and neural activity tied to actual simulation state; assets do not replace numerical validation.
- [ ] T07: add a typed-array/streaming loader for the cached graph, preserve original body IDs, and reject missing/out-of-range endpoints.
- [ ] T07: define transmitter-to-sign assumptions, sensory/output population selection, graph normalization, and stability checks.
- [ ] T07: verify the running controller actually consumes the real graph, then train and run matched ablations. Availability of the cache alone is not integration evidence.
- [ ] A5: integrate real morphology with attribution and explicit selection labels; whole-CNS morphology/registration is not delivered here.

Application source appeared during this workstream. This devlog records asset/data evidence only; the earlier specification-only baseline is historical, and application implementation status must be assessed from its own tests and review.

## Hardware packaging follow-up

See [DevLog 005](005-hardware-assets-and-coding-handoff.md) for the manufacturer CAD/PDF cache, nine original hardware assets, compact fly console, wiring registry and phased coding handoff. Scene integration and staged foreleg choreography remain coding-agent tasks.
