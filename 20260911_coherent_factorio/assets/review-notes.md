# First-batch review notes

## Accepted for prototype exploration

- Extractor, assembler, generator, and reference station have distinguishable major forms and share the initial emitter's material family.
- Junction explicitly exposes four ports; useful/rejected roles must come from network state, not permanent color painted onto the sprite.
- Dump has a strong radiator silhouette, making its heat-management purpose visible.
- Resource art is a separate object layer; remaining quantity and depletion need renderer state or later variants.
- UI was rendered in headless Chrome and visually inspected. SVG XML and JavaScript syntax were checked. Preview interactions are illustrative and are not connected to a solver.

## Follow-up before final production art

- The generated sprites retain substantial fine surface detail despite the chunkier prompt. Review in the 48/96 px gallery and simplify further if noise competes with shape.
- Elevation and projection vary, particularly on the four-port junction. Calibrate common camera and ground anchors before a rotation batch.
- Some cutout edges contain fine colored fringes. Check on actual terrain and use a targeted image edit if they remain visible; source PNGs have been preserved unchanged.
- Static cyan indicators should eventually become separate controllable layers. Do not interpret painted light as operational state.
- Stationary organisms and deposits need clearly different map markers and hit/selection behavior in addition to their silhouettes.
- Existing emitter/organism and this batch have one intact view each. Damage, destruction, active motion, and directional views are explicitly unfinished.

Next asset batch should prioritize material transport, damaged equipment/organism states, and dedicated monitoring equipment after an outpost composition review. Avoid generating a full late-game facility set before confirming game-scale readability.
