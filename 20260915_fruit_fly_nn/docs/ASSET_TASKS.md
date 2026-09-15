# Rendering asset tasks

Updated: 2026-09-15. This is the asset-production companion to [CODING_TASKS.md](CODING_TASKS.md), especially T04, T06, and T09.

## Progress

- [x] A1 — Collect and inventory canonical fly source.
- [x] A2 — Generate initial art-direction and deterministic layout inputs.
- [ ] A3 — Convert and validate articulated fly GLB.
- [ ] A4 — Build reusable procedural optical-bench components.
- [ ] A5 — Integrate scientific overlays and animation assets.
- [ ] A6 — Optimize, capture, and validate final asset pack.

Checked means the stated collection/generation task is complete, not that its output is integrated in an application.

## A1 — Canonical source collection [complete]

- [x] Fetch official FlyBody repository and pin revision `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
- [x] Copy complete upstream asset directory, LICENSE, and README into `assets/vendor/flybody/`.
- [x] Generate SHA-256 inventory, mesh counts, and raw body/joint attributes.
- [x] Validate direct file references from `fruitfly.xml` resolve locally.

Deliverables: [inventory](../assets/generated/flybody-inventory.json), [raw hierarchy](../assets/generated/flybody-hierarchy.json), [license](../assets/vendor/flybody/LICENSE).

Evidence: `python3 scripts/prepare_render_assets.py`; 90 files, 85 OBJ meshes, 161,046,149 bytes. Mesh totals: 817,650 vertex records and 272,550 triangles if polygon faces are fan-triangulated. Counts include upstream duplication, not optimized GPU vertices. XML body count is 67; the reported 180 joint elements include defaults and other XML sections, not a claim of 180 articulated DOFs. No missing direct asset references.

## A2 — Art direction and exact layout [initial pack complete]

- [x] Generate lab composition concept using built-in image generation.
- [x] Save exact prompt, output, and explicit quality limitations.
- [x] Generate deterministic 19-channel geometry JSON with stable IDs.
- [x] Generate initial material/color/path-style JSON for renderer integration.
- [x] Retain editable SVG optical-chain/layout reference.

Deliverables: [concept](../assets/concepts/cbc-lab-v1.png), [prompt](../assets/concepts/cbc-lab-v1.prompt.txt), [channel layout](../assets/generated/channel-layout.json), [style defaults](../assets/generated/render-style.json), [schematic](../assets/reference/cbc-19-layout.svg).

Visual review: concept succeeds at bench materials, cable routing, console placement, and dome/section composition. Aperture count/layout and beam field are not authoritative; use the JSON geometry and physics evaluator. Target is depicted perched rather than flying, and fly scale is exaggerated. Caption explicitly says concept/not simulated data. Do not use the image as a runtime optical texture or scientific screenshot.

## A3 — Browser-ready fly [conversion and initial validation complete; integration pending]

Owner: asset preparation; coordinate importer contracts with coding agent.

- [x] Establish source centimeters from pinned `make_fruitfly.py` (explicit mm → cm conversion); export meters and compare native MuJoCo render.
- [x] Evaluate MJCF defaults, body transforms, mesh alignment, materials, and joint axes using MuJoCo 3.13.0.
- [x] Export named articulated GLB hierarchy; preserve wings, legs, head, antennae, and abdomen nodes. Export only group-1 mesh visuals; exclude collision proxies.
- [x] Export neutral qpos0 pose and illustrative 1 Hz looping wing-motion clip.
- [ ] Add validated resting learner pose.
- [x] Save browser front/side/top and wing-animation captures; inspect side silhouette against native MuJoCo render.
- [ ] Complete matched-camera front/top source comparisons and final animation review.
- [ ] Generate low/medium/high detail versions only after profiling; preserve wing silhouette and thin structures.
- [x] Record tool versions, commands, scale, axes, clips, triangle count, and attribution.

Exit: GLB loads correctly with the selected Three.js version, articulation pivots are correct, two instances render with reasonable cost, and conversion report includes comparison images. Source OBJ collection alone does not satisfy this gate.

## A4 — Procedural optical bench kit

Owner: coding agent, with asset review support. Use code-native geometry; dimensions for display are separate from optical SI coordinates.

- [ ] Emitter assembly: housing, aperture, collimator, tip/tilt mount, focus actuator, selectable phase ring.
- [ ] Seed source, splitter, phase module, amplifier module, detector icon/assembly, cable ports.
- [ ] Console with exact 19-cell layout and reusable selected-channel controls.
- [ ] Electrical/optical/observation cable curves with attachment points, channel IDs, and selection emphasis.
- [ ] Modular bench surface and posts; avoid costly individually modeled thread holes where a procedural material suffices.
- [ ] Export component preview sheet and bounds/attachment metadata.

Exit: all instances reference shared geometry/materials where appropriate, route highlighting works, and physical-array geometry is independent of enlarged display geometry.

## A5 — Scientific overlay assets

Owner: coding agent after validated optics; reviewing agent checks field-data evidence.

- [ ] Cyclic phase legend and calibrated sequential intensity legend, with linear/log labels and units.
- [ ] Hemisphere grid, target/peak/centroid markers, bucket outline, movable section handles.
- [ ] Shader-driven beam envelopes from actual axis/width; field-sampled interference sections.
- [ ] Slowed wavefront overlay driven by actual phase and explicitly labeled illustrative timing.
- [ ] Neural population shapes/activity driven by real state once biological data are integrated.
- [ ] Save baseline fixture snapshots and corresponding field arrays for visual regression.

Exit: overlays agree with saved numeric data. No stock/generated image supplies measured intensity, connectome morphology, or learned behavior.

## A6 — Final pack and validation

- [ ] Finalize scene camera presets and lighting after component integration.
- [ ] Verify contrast, legends, selection, transparency sorting, and thin wing rendering.
- [ ] Profile 19 emitters, full wiring, two flies, and measurement layers together.
- [ ] Audit source hashes, attribution, modifications, and runtime asset sizes.
- [ ] Capture overview, emitter face, selected optical path, longitudinal focus, and target-follow screenshots.
- [ ] Update coding tasks with integration paths and validated asset status.

## Reproduction

Regenerate inventories and original JSON assets:

```sh
python3 scripts/prepare_render_assets.py
```

Recover upstream sources independently:

```sh
git clone https://github.com/TuragaLab/flybody.git /tmp/flybody-source-review
git -C /tmp/flybody-source-review checkout d015e9bfe441bd90ae431bac24c55cb74bdbce26
```

Collected path is `flybody/fruitfly/assets`; upstream files are unmodified. Keep LICENSE alongside copied/derived files. Read the pinned upstream README and applicable notices before final redistribution.


## 2026-09-15 production update

### Delivered

- Articulated fly: `assets/generated/flybody/flybody-articulated.glb` (23.0 MB, 85 meshes, 67 body nodes, 272,550 triangles). Source model has 103 compiled joints; raw XML joint-element counts include defaults.
- Converter and joint map: `scripts/export_fly_glb.py`, `assets/generated/flybody/joint-map.json`.
- Conversion evidence: `assets/generated/flybody/conversion-report.json`; body/geom composition error < 1e-15 source cm, independent GLB roundtrip bounds error < 1e-17 m.
- Browser evidence: `assets/generated/flybody/validation/`; Three.js 0.169.0 loaded all 85 meshes and one animation with no page errors. Side silhouette is consistent with source; material shading differs. Still needs runtime profiling and final poses/LOD.
- Five original display prefabs: `assets/generated/bench/` — emitter, seed, splitter, phase/amplifier, 19-cell console; named attachment ports and bounds. These are prototypes, not vendor CAD or completed control mechanisms.
- Real anatomy: `assets/generated/connectome/DNge104-skeletons.json`, two actual neurons with parent-child line indices, SI coordinates, and a shared display transform.
- Actual core MaleCNS v1.0 cache and derived graph: see [data README](../data/README.md). Both source collection and explicit subset preparation completed; controller integration remains TODO.

### Next TODOs

- [ ] Integrate articulated fly into current renderer; use separate instances and animation mixers.
- [ ] Integrate selected-channel attachment points into working controls; finish movable mechanics/cables.
- [ ] Render cached neuron skeletons with source body IDs and dataset attribution; do not label two neurons as the entire brain.
- [ ] Add a typed-array loader for the derived graph and metadata; reject missing/out-of-range endpoints.
- [ ] Decide and document transmitter sign and graph normalization; benchmark actual graph dynamics before training.
- [ ] Verify active controller provenance against the cached graph; existing synthetic mode must remain clearly labeled.

### Local preview and reproduction

```sh
.asset-venv/bin/python scripts/export_fly_glb.py
.asset-venv/bin/python scripts/generate_bench_assets.py
.asset-venv/bin/python scripts/render_fly_source.py
python3 -m http.server 8766 --bind 127.0.0.1
# Open http://127.0.0.1:8766/assets/preview/
# In a second terminal:
node scripts/inspect_asset_browser.mjs
```

Python tooling versions are pinned in `scripts/requirements-assets.lock.txt`. Local preview serves only on loopback. Simulation application code is maintained separately; these assets do not imply CBC/training validation.

- Neuron preview: `assets/preview/neurons.html`; 33,237 source segments loaded without page errors; screenshot at `assets/generated/connectome/DNge104-preview.png`. Native anatomical axes are retained; display orientation is an inspection choice.
- Source inventory refreshed after caching the pinned FlyBody build script; initial A1 counts above describe the earlier 90-file snapshot.

Progress snapshot and evidence: [DevLog 002 — assets and resource cache](../DevLog/002-assets-and-resource-cache-progress.md).

## Training/scene review update — 2026-09-15

See [DevLog 004](../DevLog/004-training-review-and-live-neural-inspector.md): completed 5,000-node MaleCNS training run, saved-spec evaluation repair, real before/after browser playback, and soma-position 3D activation driven by the active controller. Prior statements that the scene only supports synthetic activity are historical. Biological signs/mapping, true intermediate-checkpoint timelapse, and remaining optics/rendering acceptance checks are still TODO. Always inspect port ownership before starting a server; reuse the existing project server when available.

## A7 — realistic hardware, console and operating-fly handoff (2026-09-15)

- [x] Research/cache manufacturer connector, adapter, collimator, phase modulator, optical amplifier and enclosure references with hashes and rights notes.
- [x] Generate nine original hardware GLBs with named ports/pivots, including compact console and foreleg contact targets.
- [x] Prepare explicit 19-channel wiring registry and phased coding-agent instructions.
- [x] Reload/validate assets and generate an offline browser contact sheet.
- [ ] Complete remaining seed/supply/junction/tray/focus-stage assets and mechanical closeup refinement.
- [ ] Coding agent integrates routing, actual controller-driven controls, and staged fly foreleg choreography.

Start with [hardware coding handoff](HARDWARE_SCENE_TASKS.md), [cache inventory](../assets/reference/hardware/README.md), and [DevLog 005](../DevLog/005-hardware-assets-and-coding-handoff.md). Asset production is complete for the listed nine prefabs; application integration remains TODO.

## A8 — corrections from live hardware review

- [ ] Rebuild the tip/tilt mount with a horizontal optical axis above a supported base.
- [ ] Generate a 43-knob console (19 phase + 19 amplitude + five selected-channel controls), with front-center fly stance and foreleg targets.
- [ ] Add table/trays/operator platform, real placeholder sockets and cable breakout assets.
- [ ] Refine pigtail terminals and remove duplicate bulkhead/socket geometry during integration.

Follow [live rendering review R1–R5](HARDWARE_RENDERING_REVIEW.md); the previous small console is a baseline asset, not the final specification.
