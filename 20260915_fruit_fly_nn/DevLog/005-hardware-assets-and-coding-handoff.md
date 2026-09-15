# DevLog 005 — realistic CBC hardware assets and coding handoff

Date: 2026-09-15. Scope: planning, primary-source reference collection, original asset generation, and asset validation. **No application implementation, training run, server startup, or scene integration in this work.** Earlier application changes described in DevLog 004 remain separate.

## User requirements captured

- Realistic packaging for splitter, phase modulators and optical amplifiers; recognizable fiber connectors and connected optical paths.
- Compact console beside the controlling fly; visible wiring to phase drivers/modulators and aperture tip/tilt motors.
- Forelegs (“hands”) move over the controls; staged animation is explicitly acceptable.
- Continue 19 channels in 3+4+5+4+3, full phase/amplitude/tip/tilt/focus controls, genuine complex-field interference, intensity dome/sections, and inspectable neural activity.
- This agent prepares tasks/assets; coding agents implement the scene.

## Completed deliverables

- [Seven-phase coding handoff](../docs/HARDWARE_SCENE_TASKS.md), H0–H6, with explicit TODOs, contracts, evidence requirements and completion gates.
- [Manufacturer reference cache](../assets/reference/hardware/README.md): five STEP models (FC connector, adapter, collimator, phase modulator, EDFA), three PDFs (phase experiment reference and two Hammond enclosure drawings), five product-page text snapshots. Total 70,224,880 bytes with SHA-256 and durable source URLs.
- [Nine original GLBs](../assets/generated/hardware-v2/manifest.json): connector, bulkhead, SMA, phase cassette, electrical phase driver, optical amplifier, custom 19-way enclosure, articulated tip/tilt collimator, compact console. Total 248,804 bytes / 14,608 triangles before instancing.
- [Wiring specification](../assets/generated/hardware-v2/wiring-plan.json): 101 component/placeholder instances and 215 logical connections; all named endpoints resolve. Shared outputs mean explicit distribution junctions, not several plugs in one socket.
- [Contact sheet](../assets/generated/hardware-v2/contact-sheet.png) and reproducible collection/generation/validation scripts.

## Design decisions

1. Use original unbranded assets in the scene; retain vendor CAD separately as reference with redistribution rights unresolved.
2. Distinguish EO phase cassette, electrical phase driver, and optical EDFA. Draw command, RF, DC and optical paths separately.
3. Treat the 1×19 splitter as a custom conceptual device. Do not represent a researched 1×8 package as an actual 1×19 product.
4. Console has 19 selectors and five shared knobs. Knobs reflect current controller values; foreleg gestures are rate-limited choreography and do not generate neural activity or optical state.
5. Enlarge the fly only in display coordinates for reach. Keep mechanical display geometry distinct from solver aperture pitch; real-sized collimator mounts cannot occupy the existing tiny pitch without an explicit enlarged/exploded representation.
6. Manufacturer LN65S-FC and 30126C3 references use FC/PC. They are not all-PM FC/APC component evidence. Generated generic green-boot connector is only an illustrative APC representation.

## Validation and limitations

- Verified cached byte lengths and SHA-256, STEP file headers/terminators, PDF signatures, and every wiring endpoint. This does not validate the mechanical content of STEP files or fully parse PDFs.
- Generated GLBs reloaded through trimesh with matching bounds and retained named ports/pivots. Browser contact sheet loads all nine through Three.js GLTFLoader; see [preview validation](../assets/generated/hardware-v2/preview-validation.json).
- No application tests were needed because application code was not changed. These checks do not imply completed cable routing, valid collision clearance, learned fly motor behavior, or verified physical actuator transfer functions.
- Remaining asset tasks are explicit in the handoff: actual APC/PM connector dimensional reference, splitter drawing, seed/DC supply/driver junction/trays/focus mechanism, closeup mechanical refinement, and optional licensed vendor-CAD conversion.

## Reproduction

See [cache README](../assets/reference/hardware/README.md). Preview generation uses an intercepted offline browser origin and opens no network listening port. All server startup instructions must check occupied ports and ownership first.
