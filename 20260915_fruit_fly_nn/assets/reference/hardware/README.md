# Hardware reference cache and original scene assets

Collected 2026-09-15. **5 manufacturer STEP files, 3 PDFs, 5 product-page text snapshots; 70,224,880 bytes total.** The cache contains actual files, not HTML pages renamed as CAD. File sizes/SHA-256/source URLs are in [manifest.json](manifest.json). STEP headers/terminators and PDF signatures have been checked; STEP geometry has not yet been tessellated or dimensionally inspected.

## Primary manufacturer sources

| Source | Cached artifact | Intended use and qualification |
|---|---|---|
| [Thorlabs ADAFCPM2](https://www.thorlabs.com/item/ADAFCPM2) | `vendor/ADAFCPM2-ADAFCPM2-Step.step` + page snapshot | Narrow-key FC adapter with square flange. Manufacturer describes like-to-like PC or APC mating; this is not permission to mate PC against APC. |
| [Thorlabs 30126C3](https://www.thorlabs.com/item/30126C3) | `vendor/30126C3-30126C3-Step.step` + snapshot | **FC/PC** loose connector, ceramic ferrule, 3 mm boot. Mechanical packaging reference only for the generic APC plug. Do not recolor this CAD and claim it is an APC part. |
| [Thorlabs F220APC-1550](https://www.thorlabs.com/item/F220APC-1550) | `vendor/F220APC-1550-F220APC-1550-Step.step` + snapshot | FC/APC collimator packaging reference. Our generated tip/tilt mount is a separate original schematic assembly. |
| [Thorlabs LN65S-FC](https://www.thorlabs.com/item/LN65S-FC) | `vendor/LN65S-FC-LN65S-FC-Step.step` + snapshot | Manufacturer page identifies a 10 GHz, 1525–1605 nm phase modulator with **FC/PC** connectors. It does not establish an all-PM/APC chain for this scene. |
| [Thorlabs EDFA100P](https://www.thorlabs.com/item/EDFA100P) | `vendor/EDFA100P-EDFA100P-Step.step` + snapshot | PM optical amplifier instrument packaging. Optical EDFA and electrical phase/RF driver are distinct components. |
| [Thorlabs phase-modulator lab fact](https://media.thorlabs.com/contentassets/5924f547c0be4d98a41aee67e7a83d0c/fiber_eo_phase_modulator_lab_fact.pdf?v=1116113902) | `vendor/phase-modulator-lab-fact.pdf` | Physical module/pigtail and experiment photographs, modulation context; not a dimensioned package drawing. |
| [Hammond 1455N1601 drawing](https://www.hammfg.com/files/parts/pdf/1455N1601.pdf) | `vendor/hammond-1455N1601.pdf` | Actual enclosure drawing; end panels, extrusion, assembly details. Generated enclosures do not reproduce its exact dimensions. |
| [Hammond 1455NHD1601 drawing](https://www.hammfg.com/files/parts/pdf/1455NHD1601.pdf) | `vendor/hammond-1455NHD1601.pdf` | Finned enclosure variant and mounting detail reference; distinct from 1455N1601. |

Additional researched source: [Thorlabs PM 1×8 splitters](https://www.thorlabs.com/1x8-polarization-maintaining-fiber-optic-splitters). Its page provides a useful cassette-style packaging reference; **no splitter dimensional drawing/CAD has been cached yet**. The original generated 19-output enclosure is a proposed custom device, not a stock vendor product.

## Provenance / usage

- Vendor files remain manufacturer-owned reference material. Public availability is not an established redistribution license. Keep them out of `public/`, runtime bundles and a published asset library pending appropriate rights review.
- Product text snapshots include incidental page content (such as prices); this handoff does not rely on pricing or availability.
- Product pages expose STEP downloads through browser-generated blob URLs. The manifest records durable product URLs; use the Step button to reproduce collection. Failed legacy pages returned SPA shells; those shells were removed and are not counted as resources.
- CAD PDF “Open” buttons did not yield usable cached drawings in this pass. Only the three actual PDF files listed above count as cached PDFs.
- Original GLBs were generated independently in project code with generic dimensions, without importing vendor CAD or logos. They are display assets, not exact vendor replicas or fabrication-ready designs.

## Original runtime-ready starting assets

See [hardware-v2 manifest](../../generated/hardware-v2/manifest.json), [contact sheet](../../generated/hardware-v2/contact-sheet.png), [wiring plan](../../generated/hardware-v2/wiring-plan.json), and [validation report](../../generated/hardware-v2/validation-report.json).

Nine GLBs total 248,804 bytes and 14,608 triangles before instancing. Includes FC-style plug, bulkhead, SMA plug, phase cassette, electrical phase driver, optical amplifier, custom 19-output splitter, tip/tilt collimator, and compact fly console. All use meters/+Z up and named parent-relative ports; the console includes five knob pivots and foreleg contact targets. No fly choreography or scene integration has been implemented.

Known asset limitations: simplified ferrule polish/threads/socket interiors; illustrative mounting details and dimensions; schematic motor mechanics; no focus stage/seed/power supply/driver-junction/tray assets yet. Check clearance and hardware reach before accepting closeups. Scientific field textures and neural activity remain data-driven runtime layers, not decorative cached images.

## Reproduce and validate (no server required)

```sh
node scripts/cache_hardware_references.mjs
# To retry just PDF collection:
node scripts/cache_hardware_references.mjs --pdf-only
.asset-venv/bin/python scripts/generate_hardware_assets.py
.asset-venv/bin/python scripts/prepare_hardware_handoff.py
node scripts/render_hardware_preview.mjs
```

The preview intercepts an internal browser origin and serves files directly through Playwright; it opens no listening port. Python uses the existing asset environment/lockfile. Browser preview uses the project's installed Three.js and Playwright. [Coding handoff](../../../docs/HARDWARE_SCENE_TASKS.md) contains all implementation phases and acceptance gates.
