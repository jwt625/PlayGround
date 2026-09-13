# PHIELDWORKS — first playable outpost

A desktop-browser factory experiment about turning an industrial network into a controlled wave system. Build a second emitter branch, tune its phase, clear a resource frontier, and commission a reusable outpost.

## Run

Use Node.js 22.12+ (tested here with Node 23.7) and npm.

```sh
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. No API key, account, backend, or external service is required. The game uses generated assets bundled locally. Click Tutorial for the guided first outpost. The development server binds to localhost.

```sh
npm test              # simulation and world tests
npm run build         # TypeScript check + production bundle in dist/
npm run preview       # serve the production build locally
npm run test:browser  # end-to-end browser tests
```

Browser tests use installed Google Chrome at its standard macOS path when available. Otherwise run `npx playwright install chromium` once to install Playwright's browser. Test artifacts are written under test-results/ and are ignored by Git.

## First objective

The expedition starts with a working extractor, assembler, power unit, reference source, junction, emitter, and dump. Resources are finite. The starter assembler produces replacement construction parts automatically.

1. Open **Tutorial** for the guided route. Inspect the reserve ore patch and assembler. Build a **perimeter sentry at (22, 10)** and wire generator BUS OUT → sentry POWER IN. Its 8-tile range covers the eastern field branches. The first powered sentry starts a 30-second grace period; afterward sustained stray field can agitate crawlers. Add defenses as your outpost grows.
2. Build a **phase tuner at (17, 12)** and a **second emitter at (20, 12)**. Choose **Wire**, then connect the generator **BUS OUT** to the emitter **POWER IN**. The map hint displays cursor coordinates.
3. Select **Field link**. Connect junction **D** (lower-right port) to tuner **IN**, then tuner **OUT** to the new emitter **IN**.
4. Inspect the tuner. Adjust relative phase while watching **On target**. More than **32 field units** damages the armored organism; mismatched phase sends more radiation elsewhere.
5. Enable **Automatic phase control** to follow thermal drift. Clear the target and unlock the crystal deposit.
6. Run the **20-second acceptance test**, then record a blueprint. Build a generator and extractor near the frontier crystal, and wire their power ports to harvest it.
7. Accumulate the blueprint cost shown in the commissioning panel (**88 assemblies** with one sentry in the suggested layout). Pan east and place it near **(38, 3)** over the remote ore patch. Other configurations have different costs/footprints and may require another location. Placed modules need new local tuning and commissioning.

The field manual (`?`) is available in-game. **1–9** select buildings; **R** rotates a build or disconnected machine; **D** toggles cardinal/diagonal routing while a transport tool is active; **B** toggles nominal bend radius 0/0.5; click terrain between source and destination ports to add waypoints; **Esc/right click** cancels; **Space** pauses; **F** toggles the field overlay; **mouse wheel** zooms; **middle mouse/Alt-drag/WASD/arrow keys** pan; **Home** recenters; **Q** cancels construction or picks the inspected machine. The minimap supports click/drag and arrow-key navigation; **Map** fits the whole sector, and **+ / −** zoom. Click machines, deposits or creatures to inspect them. The selected object appears in the sidebar. Equipment supports disconnect, repair and recovery actions. **Diagnostics** exposes current faults, remedies, object location and detected/resolved/action history; **Tutorial** replays onboarding. The field manual also closes when its backdrop is clicked. The inspector scrolls independently of the map.

Save/Load uses this browser's local storage. New expedition replaces the running world but preserves the manual save. Save schema v3 preserves wildlife behavior/exposure, defense readiness, the grace timer, event history, machinery orientation, route geometry, in-flight inventory, link identities, and blueprint. V2 saves gain an initially peaceful ecology. V1 saves migrate to grid routes and explicit wires when the old layout has sufficient port clearance; otherwise loading reports an invalid save. Loading restores machinery, routes, inventory, and blueprint; qualification must be re-tested. Time does not advance while the tab is hidden or a modal is open.

## Implemented simulation

- Renderer-independent fixed-step world, resource inventories, recipe consumption, construction costs, material belts with discrete packets, and explicit grid power wires.
- Half-tile grid routes with physical port positions/normals, obstacle avoidance, optional diagonals, and quarter-turn footprints. Belts travel at 2 tiles/s with half-tile spacing and receiver backpressure; disconnecting a loaded belt converts its packets to scrap.
- Complex-amplitude, bidirectional port network: matched sources, unitary four-port hybrids, tuners, weak emitter reflections, lossy propagation, and absorptive terminations.
- Small dense pivoted solve of the scattering network per independent source group; powers add between independent groups.
- Route-length phase and attenuation plus an effective bend-radius penalty. The ledger separates propagation loss and bend radiation without double-counting link loss.
- Explicit source/heat/radiation/open-port/link-loss ledger. Singular networks fail with a diagnostic, not NaN propagation.
- Phase-sensitive bounded target-mode projection, thermal drift, local search phase control, target damage, trips, destructive overheating, and repairs.
- Deterministic moving crawlers, field-source investigation, obstacle avoidance, gated aggression, equipment damage and powered perimeter sentries. The armored frontier guardian stays anchored.
- Acceptance testing and blueprint replication with fresh identities and local qualification requirements.

## Deliberate limits

This is a short first-loop prototype, not the balanced 30–45 minute scenario or a full factory game. Electrical wires use generator-to-load routes with 240-unit capacity per generator; fuel, wire resistance, poles, and multi-source buses remain abstracted. Assemblies and mined crystal enter shared stock; ore travels on belts. Paths persist as half-tile segments, but construction costs, individual segment editing, inserters, splitters, and bridges remain future work. Crossings and shared runs represent insulated/separate layers and never create implicit junctions. One source supplies the initial coherent group. Tile phase is a compressed effective model, not literal optical path length. The target model is a normalized mode projection, not a full array-factor/diffraction simulation; visual beams and focus rings are illustrative diagnostics. There is no pulse dispersion, partial coherence, stochastic environmental coupling, ecology growth, or FDTD yet.

The prototype caps builds at 40 machines and 128 field ports. Equipment uses separately redrawn four-view atlas frames, with fixed camera and lighting. Presentation mappings accept dedicated sheets, scale and offsets; canvas rotation is not used. The generator returned opaque RGB sheets, so this pass uses matte art and lighten compositing, which can show terrain through dark details. True alpha cutouts, animation and final port-to-art alignment remain polish tasks. Bend radii are effective parameters on polylines, not exact arc geometry or calibrated material physics; damaged equipment is indicated by opacity/status, not unique wreck sprites. Blueprint capture currently includes the entire outpost. Human playtesting and larger-network performance work remain necessary.

## Project map

- `src/sim/definitions.ts`, `geometry.ts`: equipment definitions, typed physical ports, rotation, route metrics
- `src/sim/routing.ts`: obstacle-aware cardinal/diagonal paths and geometry validation
- `src/presentation.ts`: replaceable art mapping, orientation variants and visual offsets
- `src/sim/complex.ts`: complex algebra and linear solve
- `src/sim/network.ts`: scattering-network composition and power accounting
- `src/sim/world.ts`: economy, equipment, target, heat/control, commissioning, saves
- `src/sim/ecology.ts`, `diagnostics.ts`: wildlife/defense and actionable world diagnostics
- `src/ui/`: catalog requirements, minimap and contextual onboarding
- `src/renderer.ts`: textured canvas map, atlas rendering and overlays
- `src/main.ts`, `src/style.css`: interaction and UI
- [Milestones and TODOs](DevLog/003-implementation-milestones.md)
- [Ports/routing architecture and next TODOs](DevLog/005-ports-routing-and-presentation.md)
- [World interaction/onboarding milestone and playtest checklist](DevLog/006-world-interaction-and-onboarding.md)
- [Generated world art and exact prompt set](assets/world-generation-notes.md), [turnaround gallery](assets/world/index.html)
- [Implementation journal](DevLog/004-implementation-journal.md)
- [Asset proposal](DevLog/002-asset-plan.md), [asset gallery](assets/index.html)

Tooling references used during implementation: [Vite guide](https://vite.dev/guide/), [Playwright test configuration](https://playwright.dev/docs/test-configuration), [Node test runner](https://nodejs.org/api/test.html).
