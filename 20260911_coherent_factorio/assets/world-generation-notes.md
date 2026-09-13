# PHIELDWORKS world asset pass — 2026-09-12

Generated with the built-in image generation tool. Local originals were retained; selected PNGs were copied to `assets/world/`. No external generation service is used by the game.

Exact initial prompts: [world-generation-prompts.json](world-generation-prompts.json), [sentry-generation-prompt.json](sentry-generation-prompt.json).
Exact executed matte-sheet prompts: [world-executed-prompts.json](world-executed-prompts.json), [props correction](props-correction-prompt.json).
Source-to-deliverable mapping: [world-provenance.json](world-provenance.json).

## Delivery and mapping

- `terrain-basalt-v1.png`: full square terrain tile; repeated at eight world tiles, subdued in rendering.
- `*-turnaround-v2.png`: eight original equipment kinds, four separately redrawn views in a 2×2 atlas. TL=orientation 0, TR=1, BL=2, BR=3. Camera and lighting stay fixed; surfaces and geometry change with ground-plane orientation.
- `perimeter-sentry-v1.png`: ninth equipment kind, same four-cell layout.
- `terrain-props-v2.png`: rocks, scrub, crates, cable spool. Decorative, noncolliding; scatter avoids machinery, deposits and route samples. V1 retained as an iteration but not rendered.
- `investigator-crawler-v1.png`: four separately redrawn heading views. Movement is simulated, with heading-frame changes; this is not a walk-cycle animation.

`src/presentation.ts` maps equipment definitions to art. The renderer selects source rectangles; no canvas rotation or mirroring is used for machinery. Core footprints, ports and routing never read an image or DOM state. Scale and offset remain presentation parameters.

## Validation and limitations

The built-in generator returned opaque RGB despite requests for true alpha. A background-extraction retry also returned RGB. Painted-checkerboard candidates were rejected. Final selected sheets use dark matte backgrounds; the renderer uses lighten compositing against terrain to avoid rectangular matte patches. This preserves the generated originals but can allow terrain to show through darker metal details. True alpha cutouts and final port-to-art alignment remain art-polish tasks. Original transparent single-view sprites remain useful for palette icons.

Terrain has no collision effect. Decorative clutter cannot block building or walking. All generated art is provisional: per-orientation anchors, animation, damage variants and calibrated photonic component illustrations need later review.
