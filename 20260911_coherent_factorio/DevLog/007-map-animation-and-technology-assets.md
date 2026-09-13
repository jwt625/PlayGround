# Map, animation and technology asset continuation

Started 2026-09-12 after reviewing commit `5c65d26` (update assets). Working tree was clean; the previous interaction work is committed. The runtime has no research tree/economy yet. Technology art below is a proposed progression visualization, not an implemented unlock system.

## Batch and progress

- [x] A1 — Recover current repo state; close previous milestone validation; read progression and asset contracts.
- [x] A2 — Generate terrain variants and stateful map dressing; preserve prompts/provenance.
- [x] A3 — Generate fixed-camera animation atlases; validate orientation/frame layout and integrate production/movement playback.
- [x] A4 — Generate technology-node illustrations and science-pack assets aligned with the design progression.
- [x] A5 — Build an interactive asset/technology review surface; clearly distinguish currently represented systems from proposed future capabilities.
- [x] A6 — Inspect small-scale rendering and animation, run relevant checks, record remaining quality gaps and next steps.

## Asset contract

- Use built-in image generation. Known output limitation: the tool has returned opaque RGB despite alpha requests. Request deliberate dark matte backgrounds for sprite assets; retain source PNGs and use the existing runtime blending strategy. Do not claim production alpha cutouts.
- Map textures: four seamless square substrates (dust, fractured bedrock, industrial paving, crystal-bearing ground), with shared lighting and restrained contrast. Decoration atlas: exhausted ore, depleted crystal remnants, scrap, defeated guardian rubble. Decorations never add hidden collisions.
- Animations: 4×4 source atlases, rows are four physical ground-plane orientations, columns are four motion phases. Fixed camera, scale, base anchor and lighting; only moving mechanisms/legs change. Do not rotate a flat image for orientation. Pause, power loss and stalled production must halt corresponding motion.
- Texture, sprite, node and frame mappings live in presentation/asset metadata. No image knowledge in the simulation. Existing physical ports remain authoritative.
- Technology art: two 3×3 sheets of equipment illustrations plus one 4×2 sheet of eight industrial science packages. UI labels, dependencies, lock/selection states and progress remain native HTML/SVG/CSS, not baked into art.

## Proposed technology illustration inventory

Foundations: mechanical automation; thermal processing; electrical distribution; instrumented automation; perimeter defense; high-frequency interconnects; dense interconnects; precision materials; thermal management.

Precision/frontier: precision references; controlled field networks; field metrology; automatic calibration; semiconductor fabrication; precision patterning; decorrelated sources; planetary aperture; lightsail fabrication.

Science package families from design section 30: mechanical, electrical, materials, precision, semiconductor, field, fabrication, planetary engineering. These do not establish recipes or costs.

## Review criteria

- Terrain variants should read at map scale without overwhelming machines/routes.
- Animation base anchors should remain steady; inspect frame-to-frame drift, clipping and camera consistency. If generation cannot hold registration, mark as provisional and prefer safe bounded playback over pretending smooth production animation.
- Technology illustrations must have distinct silhouettes at 48–96 px. The proposed dependency graph is a review aid; do not gate existing tutorial equipment or alter simulation fidelity behind research.
- Preserve the direct-camera decision and initial investigative ecology. No avatar or research-economy expansion is implied by this asset pass.

- Generated and saved `assets/map/dust-basin-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/map/fractured-bedrock-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/map/industrial-paving-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/map/crystal-substrate-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/map/depletion-debris-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/animations/extractor-cycle-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/animations/assembler-cycle-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/animations/crawler-walk-v2.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/technology/technology-foundations-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/technology/technology-precision-v1.png`; review pending. Source recorded in expansion-provenance.json.

- Generated and saved `assets/technology/industrial-science-packs-v1.png`; review pending. Source recorded in expansion-provenance.json.


## First output review

All eleven requested sources are saved. The map and animation sheets are 1254×1254 RGB; the science package atlas is 1774×887 RGB (4×2 square cells). The three animation sheets have the requested 4×4 structure, physical orientation changes and distinct motion phases. Some base/scale drift remains between generated frames and will be assessed in playback. The crawler now has clear rear views.

The runtime has a cached terrain composition with soft patch transitions and state-based depleted deposits / generic wrecks / guardian remains. Machine atlas frames consume production progress; crawler gait consumes movement distance. No simulation behavior or save schema changed.

Technology art is mapped to eighteen proposed capability nodes and eight science packages in `assets/technology/catalog.json`. The review surface distinguishes current-slice, partial and proposed capabilities; baseline diagnostics and existing equipment are not research-gated.


## Registration correction

Playback review exposed clipped upper mechanisms in the last rows: the generator produced uneven vertical spacing despite the equal-cell prompt. Added explicit source rectangles and foundation anchors for the 32 machine frames in `assets/animations/frame-map.json`. The read-only measurement script detects dark gutters and stationary base tips; source PNG pixels remain unchanged. Runtime and review playback both consume this mapping. The crawler remains on a regular grid because its leg motion should not be registered to a changing foot tip.


## Integrated review checkpoint

The three new browser scenarios pass: technology dependencies/status/search, preview frame stepping/four orientations, and real production pixels advancing then freezing on pause and electrical disconnection. Production build passes. Reviewed technology graph, science packages, repeated terrain swatches, debris and the live outpost. Native technology previews retain dark details; map sprites keep the established matte compositing strategy. Final combined regression and documentation checks follow.


## Final validation and next work

- `npm run build`: TypeScript and production bundle pass.
- `npm run test:browser`: all 11 scenarios pass (46.1 s), including the entire outpost tutorial/defense/commissioning recovery and the three new asset/presentation scenarios.
- `git diff --check -- .`: clean. Existing simulation behavior and save schema were not changed in this pass.
- Browser screenshots reviewed: `technology-tree-review.png`, `map-animation-review.png`, `animated-terrain-outpost.png`, plus the established outpost scenarios.
- Outputs and exact prompts: [asset handoff](../assets/expansion/README.md). PNGs remain in `assets/map`, `assets/animations`, and `assets/technology`; source paths and dimensions are recorded in `assets/expansion-provenance.json`.

Next priorities: production alpha cutouts / smoother registered loops, belt straight/corner/end and inserter motion assets, individual damaged machine states, then in-game research data/contracts after the proposed dependency graph is reviewed. No need to generate the whole long-term building backlog before testing these silhouettes and gameplay pacing.
