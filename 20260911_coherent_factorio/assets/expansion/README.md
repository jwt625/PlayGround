# PHIELDWORKS map, animation and technology pass

Generated with the built-in image generation tool on 2026-09-12. Eleven source PNGs are delivered in the repository:

| Family | Location | Content |
| --- | --- | --- |
| Map | `assets/map/` | Four terrain textures; four depletion/debris states in a 2×2 atlas |
| Animation | `assets/animations/` | Extractor, assembler and crawler; four orientations × four phases each (48 frames) |
| Technology | `assets/technology/` | Eighteen equipment illustrations in two 3×3 sheets; eight industrial science packages in one 4×2 sheet |

[Exact prompts](../expansion-prompts.json) · [source provenance, dimensions and layouts](../expansion-provenance.json) · [milestone and review notes](../../DevLog/007-map-animation-and-technology-assets.md).

With `npm run dev` running, open:

- [Map and animation review](http://127.0.0.1:5173/assets/expansion/index.html): play/pause, frame stepping, four orientation rows, terrain repetition tests and debris states.
- [Technology tree review](http://127.0.0.1:5173/assets/technology/index.html): proposed dependency graph, selectable nodes, search, zoom, existing-slice versus proposed capability descriptions and science package art.
- [Playable outpost](http://127.0.0.1:5173): integrated terrain blends, production animation, crawler gait and depletion/wreck states.

These review pages are development artifacts served from the repo; they are not research gameplay or part of the production game bundle. No research costs, crafting recipes or unlock enforcement were introduced.

## Integration contracts

`src/terrain.ts` caches blended texture patches. Decoration never changes collision or resource availability. Existing `src/sim` definitions remain authoritative.

`src/presentation.ts` selects machine animations. Frames follow production progress, so power loss and backpressure hold a frame. `src/renderer.ts` advances crawler gait from distance traveled, so stopped creatures do not walk in place. Pausing simulation stops motion.

Generated machine sheets do not have perfectly even row spacing. [frame-map.json](../animations/frame-map.json) supplies explicit source rectangles and fixed foundation-tip anchors for 32 machine frames. `python3 assets/tools/measure_animation_frames.py` reproduces the metadata using Pillow; it reads but never rewrites source PNGs. The crawler uses a regular 4×4 grid and preserves its gait instead of anchoring to moving foot tips.

## Remaining art quality work

- All new sources are RGB dark-matte images. They are not production alpha cutouts. Runtime lighten blending can show terrain through dark metal; native source previews retain those details.
- Source frame spacing is corrected by metadata; small geometry/scale differences and four-frame motion remain visible at enlarged preview sizes. Hand-authored or 3D-rendered registered loops would improve smoothness.
- Terrain sources are tileable candidates inspected in repeated previews; repetition remains visible at large scale. The game uses subdued blended patches to keep routes readable.
- Technology nodes are visual proposals. The full progression still needs recipes, costs, capability gates and balance, while keeping baseline error visibility available.
