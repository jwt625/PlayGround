# Asset proposal: first outpost to planetary instrument

Date: 2026-09-11. Working proposal, based on design documents 000 and 001. Art direction remains open to feedback; this is not a commitment to generate the full catalog.

## Visual language

User feedback: prioritize chunky readability at gameplay scale, Factorio-like industrial construction, semiconductor-industry equipment forms, and recognizable real-world optical counterparts. Focus first on asset structure/dependencies and include UI elements. Exact surface styling may change later.

Retain the existing emitter's weathered steel, pale ceramic, copper thermal hardware, and restrained cyan status accents. Buildings should be identifiable by silhouette at gameplay scale: drill gantry, assembly bay, turbine cylinder, insulated reference core, four-port junction, adjustment carriage, radiator bank. Avoid making every device a rectangular box with a glowing ring.

Use an orthographic oblique top-down camera with consistent upper-left lighting. Generated perspective is approximate until checked in the renderer. Keep one fixed view for the first build; do not rotate shaded sprites arbitrarily to represent other views. Expose network connectors as renderer-owned markers, independent of decorative connectors.

Machinery is functional and repairable. Biology incorporates mineral structures and has an asymmetrical organic silhouette. Deposits lie low against the ground and must not resemble enemies. Use shape and symbols as well as color to distinguish resources, networks, status, and danger.

Core asset reward: a quiet collection of machines becomes a visibly coordinated installation. Dynamic light, moving material, target response, and commissioning feedback should express success; static art should leave room for these effects.

## First-build allocation

| Asset | Role / silhouette | Initial treatment |
| --- | --- | --- |
| Field emitter | Gimballed aperture, paired cooling blocks | Existing sprite |
| Armored frontier organism | Heavy mineral-plated stationary obstacle | Existing sprite |
| Starter extractor | Short gantry with vertical drill, side discharge | Generate now |
| Compact assembler | Open assembly bed and robust tool arm | Generate now |
| Power unit | Horizontal turbine/alternator and heat exchanger | Generate now; provisional energy technology |
| Reference and control station | Insulated precision core plus instrument cabinet | Generate now; shared building for source, monitoring, commissioning in first build |
| Four-port junction | Low cross-shaped housing and ceramic center | Generate now; physical routing markers drawn at runtime |
| Phase tuner | Linear adjustment carriage, folded enclosed route | Generate now |
| Cooled dump | Absorber housing and conspicuous radiator bank | Generate now |
| Starter ore | Low broken iron/copper-bearing rock | Generate now; provisional mixed-feedstock simplification |
| Frontier crystal ore | Low pale crystalline deposit with distinct faceted morphology | Generate now; provisional advanced feedstock |
| Electrical cable, field route, reference link | Distinct line pattern and connector glyph | Renderer/vector geometry, not generated raster |
| Material transport | Segmented lane plus moving item symbols | Renderer geometry in first build; belt sprites later |
| Monitor, controller indicators | Output/return meter, lock and test state | Renderer UI on reference/control station initially |
| Ground | Muted procedural background with placement grid toggle | Renderer geometry initially; terrain textures later |
| Inventory items | Ore, metal, ceramic, assembly, precision component | Simple readable vector symbols first; individual raster icons later |

This set supports a first-build implementation without requiring a complete animation library. The combined reference/control building is a prototype simplification, not a change to the long-term network-service distinction. Artwork does not fix recipes, machine stats, resource geology, or generator fuel.

## Asset backlog by gameplay milestone

| Priority | Family | Proposed assets | Unlock / purpose |
| --- | --- | --- | --- |
| Next | Material logistics | Belt straight/corner/end, inserter, crate, pipe, junction | Visibly automate production |
| Next | Readable operation | Machine active accents, drill rotation, assembly arm frames, status lights | Explain what the factory is doing |
| Next | Damage and recovery | Emitter wreck, burnt tuner, damaged dump, organism cracked/dead states, repair effect | Make destructive failures and frontier victory legible |
| Next | Network tools | Directional monitor, isolating switch, reference repeater, shielding wall | Diagnose and contain failures |
| Next | Terrain | Ground variants, deposit variants, exhausted patches, rubble | Make expansion and depletion visible |
| Middle | Ecology | Straylight feeder patches and growth stages, source fouling, detector fouling | Turn exposure into local spatial pressure |
| Middle | Moving hazards | Reflective swarm, moving armored organism, resonant flora | Motivate tracking, diversity, spectral choice |
| Middle | Production | Furnace, ceramic kiln, glassworks, crystal grower, cooling plant | Give advanced modules an industrial cost |
| Middle | Signal transport | Pulse source, compensation module, broad-spectrum source, spectral separator | Introduce temporal and spectral engineering |
| Later | Precision fabrication | Cleanroom shell, patterning tool, inspection/metrology head, die handling, packaging station | Reward good products/min and manufacturing yield |
| Later | Advanced networks | Mode filter, mode multiplexer, adaptive receiver, patterned steering surface | Turn earlier failure mechanisms into capacity |
| Endgame | Aperture | Element/tile/module/sector visual hierarchy, reference backbone, tracking station | Make scale readable without individual-element micromanagement |
| Endgame | Lightsail | Membrane fabrication, folding/deployment structure, sail stages, launch visuals | Planetary instrument payoff |

## Runtime effects and UI proposal

Do not bake these into generated machine sprites:

- Useful field delivery: beam envelope and target footprint computed from the modeled field; intensity visualization is an overlay, not a claim that an optical beam is always visibly glowing in air.
- Rejected/return power: directionally distinct animated route overlay and receiver/dump meter.
- Relative phase: reference-relative glyphs or hue, with numeric/shape alternatives.
- Coherence and lock: group outlines, lock acquisition animation, broken-lock state.
- Temperature: local heat tint and warning symbol; smoke/sparks only when stress warrants them.
- Commissioning: test progress, margins, pass/fail event, module capacity badge.
- Resources and combat: moving item symbols, extraction/depletion response, armor cracking, target destruction, newly available deposit marker.
- Placement and debugging: footprint ghost, connection ports, route preview, selected contributing paths, tooltips.

First-build effects can use vector/canvas geometry and particles. Animated sprite sheets should follow proven gameplay needs, especially for expensive destruction sequences.

## Asset contract

Stable IDs and relative PNG paths belong in assets/manifest.json. Record pixel dimensions and alpha coverage from inspection. Proposed tile footprints and display sizes are layout suggestions, not physical simulation units. Each initial asset has one intact static view; machine anchors and collision/port geometry must be calibrated in the renderer. Preserve original generated PNGs and exact prompts.

Review each at full resolution and at a small gameplay preview. Check transparency, object completeness, silhouette, shading consistency, and separation of machinery versus ecology. A preview gallery is an art-review tool, not a playable game. Do not claim assets are production-ready until common projection, anchors, directional views, and states are validated.

## Feedback and later art review

1. Continue detailed industrial realism, simplify toward chunkier game-scale shapes, or make technology more alien?
2. Any preferred game/film/equipment references, or disliked aspects of the existing assets?
3. Later review: should early construction look expedition-built and exposed, or like enclosed mature industrial equipment? Current proposal starts exposed and becomes cleaner as fabrication advances.

Questions 1–2 were answered: chunky readability, Factorio/semiconductor equipment, and real-world optical structures. Surface styling can evolve later; structure and dependencies take priority. Question 3 is optional for a future art review. Do not generate the entire backlog before evaluating the first outpost composition.

## Hardware references and dependencies

These are structural inspiration briefs, not verified vendor-specific product depictions. No branding or exact commercial product reproduction is intended.

| Asset | Real-world structural inspiration | Visual dependencies / future layers |
| --- | --- | --- |
| Emitter | Gimballed beam director, collimator housing, liquid-cooled source mount | Aperture aim, field footprint, thermal state |
| Reference/control station | Thermally enclosed oscillator, instrument rack, precision optical breadboard | Screen UI, lock indicator, commissioning state |
| Four-port junction | Packaged directional coupler / beam splitter with explicit output ports | Four independently addressable connection markers |
| Phase tuner | Motorized optical delay stage and fine adjustment drive | Actuator position, range indicator |
| Cooled dump | Cavity absorber and water-cooled beam dump | Absorbed-power meter, thermal warning, damage |
| Assembler | Semiconductor die-bonding workcell, gantry inspection stage | Tool motion, input tray, accepted/scrap output |
| Later fab | Load-lock chamber, wafer handler, inspection head, packaging tool | Cleanroom shell, transfer animation, yield UI |

Asset dependencies are not crafting recipes:

```text
terrain + deposits → extractor + material route → assembler + item icons
power unit + electrical links → active machinery
reference station + reference links → junction/tuner → emitter array
monitor UI + field overlays → diagnosis → tuning → commissioning UI
junction rejected port → cooled dump + thermal UI → safe operation
emitter + target effects → organism destruction → accessible frontier deposit
```

UI delivery: scalable SVG icons plus an HTML/CSS component preview containing resource counters, build selection, network inspector, tuning, commissioning, and warning states. The preview uses illustrative data only; no solver or gameplay behavior is implied. Text and numbers remain renderer-owned, not baked into raster panels.


## 2026-09-12 continuation

Milestones [006](006-world-interaction-and-onboarding.md) and [007](007-map-animation-and-technology-assets.md) supersede the original single-view/procedural-ground assumptions: terrain variants, decorative/stateful props, dedicated equipment orientations, three motion atlases and a proposed technology-art library are now delivered. The remaining backlog includes transport segment art, distinct machine damage states, precision manufacturing buildings and final registered/alpha animation production. See 007 for exact current scope and review results.
