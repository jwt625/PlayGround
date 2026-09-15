# Rendering references and asset plan

Reviewed 2026-09-15. These sources guide implementation; they do not validate unimplemented code. The neuroscience/source claims in the initial discussion still need a separate verification pass.

## Sources read

| Source | Verified use | Implementation implication |
|---|---|---|
| [TuragaLab/flybody](https://github.com/TuragaLab/flybody) | Official repository describes an anatomically detailed MuJoCo fly and links its body-model directory; repository displays Apache-2.0 | Preferred source mesh/model. Read the actual license and asset-specific notices at a pinned revision before conversion; retain provenance. |
| [FlyBody paper](https://www.nature.com/articles/s41586-025-09029-4) | Bibliographic link confirmed through official repository; paper itself not independently reviewed in this pass | Reference for anatomy/embodiment; do not infer that its locomotion policy is a connectome controller. |
| [RP Photonics: Gaussian beams](https://www.rp-photonics.com/gaussian_beams.html) | Author-maintained technical reference read for beam parameterization, width, curvature, and phase | Implement full complex amplitude and explicitly document Gaussian/paraxial approximation. It does not establish our numerical accuracy. |
| [Three.js Data3DTexture](https://threejs.org/docs/pages/Data3DTexture.html) | Official API supports a texture from a 3D data array | Optional sampled-volume rendering route after profiling; not itself an interference solver. Start with sections and envelopes. |
| [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html) | Official glTF loader documentation | Use GLB for converted articulated assets; pin the Three.js version and test required loader extensions. |

## Created asset

[`assets/reference/cbc-19-layout.svg`](../assets/reference/cbc-19-layout.svg) is an original editable schematic: stable row ordering, hex-packed 19-emitter face, collapsed optical chain, and command connections. It is a design aid, not a final UI, complete wiring diagram, or physical field visualization. No third-party image content is embedded.

## Asset production queue

1. **Code-native optical bench:** parametric emitter housings, lenses, actuator assemblies, seed/splitter boxes, console, and selectable cable curves. Generate geometry in code so layout remains tied to channel IDs. Avoid baking schematic labels into textures.
2. **Fly mesh conversion:** fetch/pin upstream FlyBody model; inspect mesh formats, body transforms, joint axes, scale, materials, and all applicable licenses. Export reproducible GLB with articulated hierarchy and named wings/legs/head. Validate silhouette, scale, transforms, and animation against upstream rendering. Do not flatten all anatomy into one mesh before evaluating animation needs.
3. **Scientific layers:** create intensity textures and wavefront geometry from saved simulation snapshots. No generated raster can substitute for this data.
4. **Optional decorative textures:** create only after scene composition exists. Keep generated imagery out of quantitative optics and biological provenance claims. No raster asset is needed for the current handoff.

## Visual direction

Dark, restrained optical laboratory; readable turquoise optical paths and amber electrical paths, with line-style distinctions. Colorblind-safe sequential intensity colormap plus optional log view. Cyclic phase colors get their own legend. Keep labels and slice grids sharp; bloom belongs on an optional presentation layer. Show wires as traceable routes rather than a dense hairball: expand selected branches and dim others.

Camera presets: whole-system overview, emitter face, selected-channel optical path, target/section view, and longitudinal focusing view. A physical console remains visible in overview; the 2D schematic supports precise editing at any camera position.

## Asset manifest requirements

For every imported/derived asset record source URL, pinned revision, original filename, license and notices, modifications, conversion command/tool versions, physical units, orientation, animation names, vertex/triangle counts, texture sizes, and attribution string. Record original project-created assets as such. FlyBody source meshes have now been collected and inventoried at a pinned revision; an articulated GLB conversion and initial browser/source validation are now available. See [asset tasks](ASSET_TASKS.md) and [asset pack](../assets/README.md).


## Additional primary sources inspected and cached

- [Official MaleCNS v1.0 download documentation](https://male-cns.janelia.org/download/): connection/annotation/transmitter file identities, skeleton units, and CC BY 4.0 license. Local manifests and derived statistics are documented in [data/README.md](../data/README.md).
- [MuJoCo model/data API](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html): compiled model arrays and forward-kinematic transforms used for GLB export.
- FlyBody pinned build script `assets/vendor/flybody/UPSTREAM_make_fruitfly.py` explicitly rescales mm to cm; exporter applies cm to m separately.

Progress snapshot and evidence: [DevLog 002 — assets and resource cache](../DevLog/002-assets-and-resource-cache-progress.md).
