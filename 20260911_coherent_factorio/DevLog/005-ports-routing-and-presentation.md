# Ports, transport, and presentation separation

Started 2026-09-12. Continued from local Codex rollout 01a09422-253b-7911-96fd-eea0e41fd004 and DevLogs 000–004.

## Milestones

- [x] P1: Move sprite mapping outside simulation; explicit physical port contracts and quarter-turn geometry.
- [x] P2: Persist obstacle-aware grid paths, validate endpoint direction and route geometry, support optional diagonals.
- [x] P3: Drive field phase/loss from route geometry; expose bend radius penalties and separate radiation accounting.
- [x] P4: Material packets with travel time, spacing, backpressure, and conservation on disconnect.
- [x] P5: Route/rotation UI, save migration, blueprint geometry, headless and browser regression checks.

## Decisions

Simulation owns footprint, orientation, typed ports, route centerlines, inventories, and physical model parameters. Presentation owns sprites, anchors, camera and hit testing; physical port positions remain authoritative. Field port indices retain their scattering-matrix order; indices are scoped by transport type. Material and power interfaces use explicit input/output roles. Quarter turns rotate physical geometry; artwork may supply orientation variants independently.

Half-tile routing lattice accommodates multiple ports per two-tile device. Buildings stay on whole tiles. Routes leave and enter along port normals. Cardinal routing is the default; optional 45-degree segments use the same geometry validation. Crossings are insulated/grade-separated in this prototype, never implicit junctions; route sharing/crossing costs and dedicated bridge/splitter elements remain follow-ups. Building interiors block routes and routes block subsequent building placement.

Waveguide centerlines have a nominal bend radius with locally constrained effective radius. The initial bend penalty is a tunable game model, not material-specific electromagnetic data. Phase uses centerline length in effective radians/tile. Radiation and straight propagation loss subdivide existing link loss, preserving the energy balance. Wires now use explicit generator-to-load routes. Each generator supplies 240 units and permits output fan-out; each load permits one input feed. Generator overload shuts its connected loads down independently of other generators. Fuel, resistance, ampacity, poles and multi-source buses remain abstracted.

## Follow-ups

- Dedicated rotated sprite views and adjustable sprite/port presentation anchors.
- Route segment construction costs, crossing layers, bridges, splitters, and per-tile editing.
- Material-specific bend calibration, true arc geometry/clearance, optical delay and thermal environment.
- Wire resistance/capacity, poles, and multi-source connected-component power allocation.

## Progress — 2026-09-12

P1–P4 implemented. Simulation modules have no sprite/DOM/canvas imports. `presentation.ts` maps art IDs, optional orientation variants, scale and offsets independently. Core geometry defines rotated port normals and footprints. Renderer previews and hit targets consume this geometry. Existing art rotates in canvas as a temporary fallback, not a newly generated oblique camera view.

P5 implementation complete; headless and browser regression checks passed, and starter/rotated-layout screenshots were visually reviewed. R rotates build previews or disconnected equipment; D toggles diagonal routing; B selects nominal radius 0 or 0.5; terrain clicks add waypoints. Ports show outward normals. Gold packets represent actual in-flight inventory. Purple routes carry electrical supply. The inspector reports route length, bend count and tight corners; the energy ledger splits propagation loss and bend radiation.

Fixed issues found during validation: inventory tests now include in-flight ore; blueprint bounds include external port stubs; blueprint placement validates all translated routes before committing; v2 load preserves link identities and packet positions. V1 migration autoroutes old links, supplies explicit wires to formerly covered loads, and clears qualification. A physically unroutable legacy layout is rejected rather than silently losing a connection. Legacy blueprint migration adds a clearance margin.

The original full gameplay browser test passes with the added emitter wire. Target thresholds and thermal behavior were retained. Reflected energy can trip the reference source slightly earlier with the changed path lengths; the protection regression checks returned power immediately and cooling over the subsequent interval.

## Effective route model (not a physical calibration)

For polyline length L, straight power-loss exponent is 0.012 L and phase is 0.31 L rad. For each direction change theta, effective radius is the smaller of the requested radius and half the shorter neighboring straight length divided by tan(theta/2). Half-length allocation prevents adjacent corner fillets from claiming the same straight run. The bend exponent is 0.001 × (theta / (pi/2)) × [1 + 8 max(0, 1 - R/0.5)^2]. Amplitude transmission is exp[-(straight exponent + bend exponent)/2]. The link-loss ledger is split in proportion to these exponents (an aggregate allocation, not spatially resolved heating/radiation).

This version measures the lattice polyline, does not subtract fillet lengths, and does not simulate exact arcs, bend reflections or material/wavelength dependence. The parameters are intentionally replaceable. Orange markers denote radii below the nominal 0.5-tile threshold. Crossings and shared runs currently represent separate insulated/layered routes at zero extra cost.


## Validation completed

- `npm test`: 30 passing headless tests, including legacy migration, stable link IDs, packet conservation/backpressure, port normals/rotation, diagonal corner clearance, electrical disconnection/overload, energy accounting, commissioning, and atomic blueprint failure.
- `npm run build`: TypeScript and production bundle pass.
- `npm run test:browser`: three browser scenarios cover the full outpost loop, invalid placement/disconnection/manual, and rotated equipment with diagonal waypoints, sharp bends, material-port reconnection and persistence.
- Reviewed `test-results/initial-outpost.png` and `test-results/grid-routing-and-rotation.png`. Route types and rotated ports are visible; dedicated rotated artwork remains intentionally deferred.

Next implementation priority: editable transport segments and explicit crossing/splitter pieces, followed by calibrated route profiles and dedicated orientation art. Human playtesting and scaling benchmarks remain open; automated scenario success does not establish balance or large-factory performance.


### Superseded by world-interaction milestone

[006](006-world-interaction-and-onboarding.md) replaces the temporary canvas rotation with nine dedicated four-view equipment atlases, introduces save schema v3 ecology, and adds direct-camera navigation, onboarding and diagnostics. The remaining segment editing/calibrated bend work above is still deferred.
