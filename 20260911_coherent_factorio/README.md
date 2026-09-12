# Fieldworks — first playable outpost

A desktop-browser factory experiment about turning an industrial network into a controlled wave system. Build a second emitter branch, tune its phase, clear a resource frontier, and commission a reusable outpost.

## Run

Use Node.js 22.12+ (tested here with Node 23.7) and npm.

```sh
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. No API key, account, backend, or external service is required. The game uses the existing local assets. The development server binds to localhost.

```sh
npm test              # simulation and world tests
npm run build         # TypeScript check + production bundle in dist/
npm run preview       # serve the production build locally
npm run test:browser  # end-to-end browser tests
```

Browser tests use installed Google Chrome at its standard macOS path when available. Otherwise run `npx playwright install chromium` once to install Playwright's browser. Test artifacts are written under test-results/ and are ignored by Git.

## First objective

The expedition starts with a working extractor, assembler, power unit, reference source, junction, emitter, and dump. Resources are finite. The starter assembler produces replacement construction parts automatically.

1. Build a **phase tuner at (17, 12)** and a **second emitter at (20, 12)**. The map hint displays cursor coordinates.
2. Select **Field link**. Connect junction **D** (lower-right port) to tuner **IN**, then tuner **OUT** to the new emitter **IN**.
3. Inspect the tuner. Adjust relative phase while watching **On target**. More than **32 field units** damages the armored organism; mismatched phase sends more radiation elsewhere.
4. Enable **Automatic phase control** to follow thermal drift. Clear the target and unlock the crystal deposit.
5. Run the **20-second acceptance test**, then record a blueprint. Build a generator and extractor near the frontier crystal to harvest it.
6. Accumulate **80 assemblies** for the suggested outpost blueprint. Pan east and place it near **(38, 3)** over the remote ore patch. Other configurations have different costs/footprints and may require another location. Placed modules need new local tuning and commissioning.

The field manual (`?`) is available in-game. **1–8** select buildings; **Esc/right click** cancels; **Space** pauses; **F** toggles the field overlay; **mouse wheel** zooms; **middle mouse/Alt-drag/arrow keys** pan; **Home** recenters. Click a machine to inspect, disconnect routes, repair, or recover it. The inspector scrolls independently of the map.

Save/Load uses this browser's local storage. New expedition replaces the running world but preserves the manual save. Loading restores machinery, routes, inventory, and blueprint; qualification must be re-tested. Time does not advance while the tab is hidden or a modal is open.

## Implemented simulation

- Renderer-independent fixed-step world, resource inventories, recipe consumption, construction costs, material routes, and spatial power coverage.
- Complex-amplitude, bidirectional port network: matched sources, unitary four-port hybrids, tuners, weak emitter reflections, lossy propagation, and absorptive terminations.
- Small dense pivoted solve of the scattering network per independent source group; powers add between independent groups.
- Explicit source/heat/radiation/open-port/link-loss ledger. Singular networks fail with a diagnostic, not NaN propagation.
- Phase-sensitive bounded target-mode projection, thermal drift, local search phase control, target damage, trips, destructive overheating, and repairs.
- Acceptance testing and blueprint replication with fresh identities and local qualification requirements.

## Deliberate limits

This is a short first-loop prototype, not the balanced 30–45 minute scenario or a full factory game. The local 16-tile power bus abstracts electrical wiring and generator fuel. Assemblies enter shared construction stock; material routes are not individual belt tiles. One source supplies the initial coherent group. Tile phase is a compressed effective model, not literal optical path length. The target model is a normalized mode projection, not a full array-factor/diffraction simulation; visual beams and focus rings are illustrative diagnostics. There is no pulse dispersion, partial coherence, stochastic environmental coupling, ecology growth, or FDTD yet.

The prototype caps builds at 40 machines and 128 field ports. Sprites are intact single-view art; damaged equipment is indicated by opacity/status, not unique wreck sprites. Blueprint capture currently includes the entire outpost. Human playtesting and larger-network performance work remain necessary.

## Project map

- `src/sim/complex.ts`: complex algebra and linear solve
- `src/sim/network.ts`: scattering-network composition and power accounting
- `src/sim/world.ts`: economy, equipment, target, heat/control, commissioning, saves
- `src/renderer.ts`: canvas map and overlays
- `src/main.ts`, `src/style.css`: interaction and UI
- [Milestones and TODOs](DevLog/003-implementation-milestones.md)
- [Implementation journal](DevLog/004-implementation-journal.md)
- [Asset proposal](DevLog/002-asset-plan.md), [asset gallery](assets/index.html)

Tooling references used during implementation: [Vite guide](https://vite.dev/guide/), [Playwright test configuration](https://playwright.dev/docs/test-configuration), [Node test runner](https://nodejs.org/api/test.html).
