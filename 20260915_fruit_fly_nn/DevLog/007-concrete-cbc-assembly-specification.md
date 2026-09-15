# DevLog 007 — concrete assembly instructions for the coding agent

2026-09-15. Documentation only. No application code, scripts, asset geometry or runtime registry changed.

## Deliverable

[CBC assembly specification](../docs/CBC_ASSEMBLY_SPEC.md) now replaces the earlier approximate assembly instructions. It contains:

- Counted shared equipment, 19-channel components, 57 actuator children, connectors, supports and cable infrastructure.
- Table/rack/tray locations, shelf elevations, exact electronics-row assignments and canonical hex aperture positions for CH01–CH19.
- Three enclosure orientation classes and a specific horizontal-optics/vertical-support requirement for the collimator assembly.
- Port names, positions, normals and interface types, including explicit IO, motor and power distribution sockets.
- Shared cable schedule, twelve repeated per-channel cable families, and a 19-row table of IO/motor/PDU assignments.
- 241 internal runs: 58 optical, 19 RF, 40 command, 48 DC and 76 motor; one external supply-entry cord. Optical connections require 78 installed FC plugs and 78 matching equipment sockets, with no extra standalone mating sleeves.
- Named tray routes, divided cable lanes, spline lead/arrival constraints, jacket sizes, curvature requirements and aperture loop-depth assignments.
- Exact surface coordinates for 38 per-channel knobs, five near-edge shared knobs, 19 selector buttons and display; console-relative foreleg contact/stance requirements.
- Registry migration and asset assignments, including replacement of the old shared-output junction shortcuts.

## Packing correction performed in the specification

Read the existing hardware manifest and canonical channel layout. Envelope arithmetic exposed a pigtail/driver conflict in the earlier suggested cell. The revised cell rotates the cassette 90° in yaw so its pigtails face forward/rearward, moves the amplifier right, and moves the driver rearward. Documented converted footprints now fit the baseplate with reserved connector/tray corridors. This is an arithmetic check against cached asset bounds, not a mesh collision test; the coding agent must verify routed geometry.

Previous review/checklist documents now link to this specification and explicitly defer to it. Existing generated wiring JSON remains the old proposal until the coding agent migrates it; no claim is made that the revised assembly is already implemented.
