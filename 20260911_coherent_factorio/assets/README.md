# Fieldworks asset library

Start with the [asset gallery](index.html), [UI component kit](ui/index.html), and [asset proposal](../DevLog/002-asset-plan.md). The [manifest](manifest.json) records stable IDs, dimensions, alpha information, proposed footprints, and integration dependencies. Footprints and anchors remain provisional.

The first-outpost batch adds extractor, assembler, power unit, reference/control station, four-port junction, phase tuner, cooled dump, and two deposits. The UI kit includes 23 original SVG symbols and interactive HTML/CSS component examples. These support first-build implementation; they are not a playable game.

Generation uses the built-in image_gen tool. [Batch 02 prompts](generation-prompts-batch-02.md) record the exact requests and reference-image role. Code-native UI assets use no image generation. All raster originals are preserved without editing. The gallery includes 48/96 px display previews without modifying source files.

Rebuild catalog metadata and the gallery with `python3 assets/tools/build_catalog.py` (requires Pillow). Rebuild vector icons with `python3 assets/tools/build_ui.py`. The project .gitignore explicitly keeps asset outputs that the parent repository otherwise ignores.

## First two assets

Two provisional static assets for the [frontier experiment](../DevLog/001-gameplay-and-frontier-experiment.md), generated with the built-in image_gen tool. Original outputs are preserved unchanged with real alpha transparency, verified by reading the PNGs.

| Asset | Dimensions | State |
| --- | --- | --- |
| [Field emitter](sprites/field-emitter-v1.png) | 1254 × 1254 RGBA | Intact, idle, one orientation |
| [Armored frontier organism](sprites/armored-frontier-organism-v1.png) | 1312 × 1199 RGBA | Intact, stationary, one view |

See [exact generation prompts](generation-prompts.md) for provenance and reproduction inputs. Generative output is not deterministic.

## Art direction and integration

Oblique top-down industrial art, weathered steel, ceramic and copper, readable silhouettes, upper-left lighting. Machinery uses restrained cyan indicators; the organism uses mineral armor and amber organic material. Both are standalone cutouts without baked-in field beams. Keep beams, field intensity, targeting, warning indicators, and selection outlines dynamic in the renderer.

These are initial artwork, not finished animation sheets. Their camera angles and perceived scale are approximate, not a calibrated common projection. Before production use, standardize camera, ground anchors, tile footprints, sprite scale, and connection locations. Generate consistent directional views and damaged states after deciding how objects rotate. Do not infer physical optical phase or collision boundaries from artwork pixels.

The emitter has a tiny generated indicator marking; replace it with a renderer-controlled status display if legibility matters. Preview small-size readability in the actual game camera before accepting detail density. No engine or gameplay integration exists yet.

## Preview

Open [index.html](index.html) locally to compare both assets against a dark grid. The displayed grid and scale are illustrative, not physical units.
