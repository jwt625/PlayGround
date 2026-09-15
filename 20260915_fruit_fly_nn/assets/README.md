# Rendering asset pack

See [asset tasks](../docs/ASSET_TASKS.md) for status and next work.

- `vendor/flybody/`: unmodified source assets from TuragaLab/flybody revision `d015e9bfe441bd90ae431bac24c55cb74bdbce26`, plus Apache-2.0 LICENSE and upstream README. Approximately 154 MiB. Unmodified source asset directory; the pinned build script is also cached for unit provenance. A converted GLB is now available under `generated/flybody/`.
- `generated/`: original deterministic channel/style JSON and generated inventories of upstream source. Rebuild with `python3 scripts/prepare_render_assets.py`.
- `reference/`: original editable SVG schematic.
- `concepts/`: built-in AI-generated composition reference and exact prompt. The pictured aperture count and optics are illustrative; not simulation output. Reviewed for composition, with limitations recorded in asset tasks.

FlyBody attribution: TuragaLab/flybody; model developed by Google DeepMind and HHMI Janelia Research Campus. Cite Vaxenburg et al., “Whole-body physics simulation of fruit fly locomotion,” Nature 643, 1312–1320 (2025), https://doi.org/10.1038/s41586-025-09029-4. Preserve upstream license and any applicable notices when distributing derivatives.


## New runtime/inspection assets

- `generated/flybody/flybody-articulated.glb`: meter-scale articulated fly, +Y up/+X forward, illustrative wing clip. See conversion report and validation captures beside it.
- `generated/bench/*.glb`: five original display prefabs; port/bounds metadata in manifest. These use display meters and Z-up; apply an explicit frame transform when mixing with the fly asset.
- `generated/connectome/DNge104-skeletons.json`: actual MaleCNS v1.0 skeletons, CC BY 4.0, source body IDs and SI coordinates preserved. Attribution and raw cache details: [data README](../data/README.md).
- `preview/index.html`: standalone fly inspection view; serve repository root locally, then open `/assets/preview/`.

Real neuron morphology preview: serve repository root and open `/assets/preview/neurons.html`. See the saved screenshot and browser report under `generated/connectome/`.

## Hardware v2

[Original hardware assets](generated/hardware-v2/manifest.json), [contact sheet](generated/hardware-v2/contact-sheet.png), and [wiring plan](generated/hardware-v2/wiring-plan.json) are available for coding-agent integration. [Manufacturer CAD/PDF cache](reference/hardware/README.md) remains separate reference material. Follow [hardware scene tasks](../docs/HARDWARE_SCENE_TASKS.md); these prefabs do not implement the scene or fly gestures.
