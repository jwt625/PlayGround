# CBC hardware scene: coding-agent handoff

Status: **specification and standalone assets ready; integration not performed.** Updated 2026-09-15.

## Goal and scope

Make the existing 19-channel CBC scene look like a connected optical experiment: enclosed splitter, distinct phase modulators and optical amplifiers, recognizable fiber connectors, motorized tip/tilt collimators, and a compact console beside the controlling fly. Preserve the existing real controller/field data. The fly's forelegs may perform staged operating gestures.

Read [rendering specification](../DevLog/001-cbc-optics-rendering-spec.md), [coding tasks](CODING_TASKS.md), and [hardware asset inventory](../assets/reference/hardware/README.md) first. This document refines the hardware/embodiment work; it does not mark existing optics or training gates complete. Do not retrain merely to change packaging.

## Required architecture

```text
Seed → custom PM 1×19 splitter
  CH01…CH19 → phase cassette → PM optical amplifier → fiber collimator → free-space CBC

Fly → compact console → phase-command bus → electrical phase driver → RF/bias cable → phase cassette
                     → motor-command bus → aperture driver junction → tip/tilt motors
                     → amplitude command → amplifier setpoint (illustrative actuator mapping)
                     → focus command → focus actuator (asset still needed)
Power supply → console, electrical drivers, optical amplifiers, aperture driver junction
```

This is a proposed conceptual topology, not a vendor-qualified BOM. Amplitude-to-amplifier control is a display mapping until a documented device transfer function is implemented; do not invent bandwidth or hide saturation. Phase drive voltage requires the configured Vπ/bias convention before showing a physical voltage. A UI value of zero amplitude is not evidence of infinite extinction in an EDFA. The RF driver is electrically distinct from the optical amplifier.

The 1×19 enclosure is a **custom schematic device**. The researched stock PM 1×8 package is a packaging reference. Do not substitute a 1×32 distribution with unused ports without explicitly changing the power budget. Every modeled branch must retain the existing CH01–CH19 identity and 3+4+5+4+3 ordering.

### Mechanical and optical coordinate contract

- New GLBs use meters, +Z up. Port node local +Z points outward; port positions are relative to their declared parent. Connect mating planes with opposing normals and align keys. Apply one explicit up-axis transform at the imported asset root if needed.
- Never scale physical aperture pitch, wavelength, target distance, beam waist, or phase to fit the hardware. Use an explicit `displayTransform`/exploded bench layout. Distinguish mechanically enlarged emitter housings from SI solver positions in the inspector.
- The 45 mm wide illustrative tip/tilt mount cannot physically occupy a sub-millimeter channel pitch. Show an enlarged aperture assembly or transition to an optical-coordinate inset; do not conceal overlapping housings or imply a buildable compact packing.
- The 160 × 110 mm console and millimeter-scale fly require an intentional presentation scale. Uniformly enlarge the fly in its display group until its forelegs can reach the controls; expose “fly enlarged for display.” Keep anatomical dataset coordinates unchanged.
- No baked vendor logos; no internal electronics/optical paths invented as verified manufacturing detail. Optional cutaways must be labeled schematic.

## Available assets and contracts

Path prefix: `assets/generated/hardware-v2/`. All are original generic illustrative geometry; all nine reload and preserve their port/pivot nodes. See `manifest.json` for exact bounds, dimensions, triangles, and port metadata. These are improved starting assets, not finished engineering CAD.

| GLB | Purpose | Named interfaces |
|---|---|---|
| `fc-apc-plug.glb` | Green boot, coupling nut, key, simplified ceramic ferrule | `port_mating`, `port_cable_exit` |
| `fc-bulkhead.glb` | Flange/socket adapter | `port_front`, `port_rear` |
| `sma-plug.glb` | Distinct gold RF connector | `port_mating`, `port_cable_exit` |
| `splitter-19.glb` | Custom enclosure with 19 output sockets | `port_input`, `port_CH01`…`port_CH19` |
| `phase-cassette.glb` | Enclosed EO component, pigtails, RF input | `port_optical_in`, `port_optical_out`, `port_rf` |
| `phase-driver.glb` | Electrical driver enclosure | `port_command`, `port_dc_power`, `port_rf_out` |
| `optical-amplifier.glb` | Generic optical amplifier instrument with heatsink | `port_optical_in`, `port_optical_out`, `port_command`, `port_dc_power` |
| `tiptilt-collimator.glb` | Collimator, yoke, two schematic motors | `tip_pivot` (X), child `tilt_pivot` (Y), `port_fiber_in`, `port_emission`, `port_motor_bus` |
| `fly-console.glb` | Compact channel selector and five shared knobs | CHxx selector meshes; `knob_phase/amplitude/tip/tilt/focus`; `touch_*`; `fly_stance`, `rest_left/right`; three rear ports |

`port_optical_in/out` on the phase cassette are **pigtail exits**, not bulkhead mating sockets. Add a fiber segment and connector at the free end. Port kinds and channel IDs are in the manifest; never infer compatibility from cable color. APC/PC ferrule geometry, threads, fastener holes, motor internals and mounting clearances remain simplified. The placeholder focus knob has no focus mechanism yet.

## Phase H0 — baseline, contracts, integration plan

- [x] Read current source and existing uncommitted work before editing. Record current scene/training behavior and capture baseline screenshots.
- [x] Inspect listening ports and process ownership before launching any preview; reuse the existing project server or choose a verified free port with strict-port behavior.
- [x] Write the typed component/port/channel/cable registry and separate display transforms from solver coordinates.
- [x] Map all 19 stable IDs to assets, readout outputs, selectors and schematic branches.
- [x] Select the existing app's importer/asset publication pattern; do not ship the 66+ MB vendor STEP reference cache to the browser.
- [x] Confirm rendering budget with one module and one cable before instancing 19 branches.

**Gate:** baseline retained; registry has no duplicate IDs; importing models does not change any solver input or output.

Status 2026-09-15 (coding agent): H0 complete. `src/renderer/hardware/registry.ts` loads the
generated manifest + wiring plan into a typed registry; `tests/hardwareRegistry.test.ts`
(5 tests) checks all nine components, 19 unique channel mappings, the CH01 optical path,
zero structural errors (pigtail-assembly warnings only), and display/solver separation.
Ran against the existing dev server on 127.0.0.1:5173. Baseline screenshots are under
`test-results/` (gitignored). H1/H2 first pass: `src/renderer/hardware/hardwareScene.ts`
loads the nine GLBs via a Vite glob (vendor STEP never imported), places 100 modules, and
routes 195 sampled cables from the wiring plan behind an opt-in **Hardware bench (v2)**
toggle. Remaining H1/H2 work (mated connectors, labels, tube geometry, bend checks,
per-channel tracing gate) is not yet done.

### Integration status 2026-09-15 (coding agent, second pass)

All nine generated `hardware-v2` GLBs are now instantiated in the live scene,
shown by default (the **Hardware bench (v2)** checkbox can hide it):

- 78 asset modules: 1 splitter-19, 1 fly-console (enlarged for the display-scale
  fly), 19x phase-cassette, 19x optical-amplifier, 19x phase-driver, 19x
  tiptilt-collimator, plus placeholders for asset-needed nodes (seed, supply,
  focus stages, junctions).
- 172 mated connector instances: fc-apc-plug at FC/APC and PM-pigtail ports,
  sma-plug at RF ports, and fc-bulkhead feedthroughs on every splitter channel
  output. Connectors are children of the GLB port nodes, so they follow motion.
- 215 routed cables from `wiring-plan.json`, colour- and style-separated by
  optical / RF / control / power; selecting a channel dims unrelated cables.
- Channel ID sprite labels at each aperture collimator.
- Nested `tip_pivot` (X) / `tilt_pivot` (Y) driven from the actual per-channel
  pointing with a documented display gain; focus stage asset still pending.

Coverage is asserted by the browser test `hardware bench loads on demand and
routes cables` (9 component kinds, >70 modules, >100 connectors, zero load
failures). Still open: H1 mated-connector closeups and clearances, H2 tube
cables with bend checks and the CH01/CH10/CH19 tracing gate, H3 console knob
binding, H4 fly foreleg gestures, H5 focus stage, H6 evidence capture.

## Phase H1 — enclosure and connector integration

- [ ] Import new assets behind an opt-in scene mode, then replace crude v1 boxes after acceptance. Reuse materials/geometries across instances.
- [ ] Lay out seed and splitter, 19 channel modules grouped in trays, and the aperture with mounting surfaces and believable clearances. The seed can initially reuse the existing prefab, clearly tracked as a remaining asset task.
- [ ] Keep phase cassettes, electrical drivers, and optical amplifiers separately identifiable. Allow a selected-channel exploded view to make routing readable.
- [ ] Mate FC plugs to compatible bulkheads by transforms; hide/protect the ferrule when connected. Orient key references consistently for PM connections.
- [ ] Add readable runtime labels, channel IDs and optical IN/OUT arrows; keep labels outside static textures.
- [ ] Add separate DC/control/RF sockets where absent; leave no unexplained powered module. Use source drawings for future detail upgrades; current models are illustrative.

**Gate:** closeups show lids/seams/feet/connectors and distinct RF/optical/control interfaces; zero floating cable ends; no FC/PC-to-FC/APC mates; no gross enclosure intersections.

## Phase H2 — routed optical and electrical harnesses

- [ ] Consume `wiring-plan.json` to instantiate explicit edges and branch/junction identities. Add seed/supply/junction placeholders with registered ports until their detailed assets exist.
- [ ] Route fibers from seed → splitter → phase → amplifier → each collimator; route command lines console → driver banks and aperture motors, including visible rear-console connectors.
- [ ] Use smooth cable curves constrained by port tangents; add strain relief and service loops. Do not animate light as particles traveling through electrical wires.
- [ ] Implement adjustable bend-radius/clearance checks. The supplied visual radii are presentation defaults, not manufacturer-qualified minimum bend radii.
- [ ] Bundle parallel wires into conduits/trays with explicit breakout nodes, retaining per-channel graph edges. Selected-channel view exposes its entire path and dims unrelated branches.
- [ ] Parent moving-end anchors to the tip/tilt hierarchy; update endpoint position and tangent after every pose change. Avoid rigid cables detaching from moving optics.
- [ ] Keep fibers/control cables/RF/DC distinguishable by labels and line style as well as color.

**Gate:** CH01, CH10 and CH19 can each be traced end-to-end in 3D and schematic; cables remain attached during full allowed tip/tilt sweep; automated registry check catches missing/incorrect endpoints.

## Phase H3 — compact operating console

- [ ] Place `fly-console` next to the fly, reachable in the overview composition. Add a 19-channel selector arranged 3+4+5+4+3, five shared knobs, and a small selected-channel/status readout.
- [ ] Bind phase, amplitude, tip, tilt and focus knobs to the selected channel's actual current values; use the same state store as the 2D schematic. Preserve selection across camera changes.
- [ ] Give phase a cyclic indication; use bounded position mappings for other knobs and show units/ranges. Distinguish current measured/commanded values if they differ.
- [ ] Expose existing manual/controller/replay ownership explicitly. Manual edits must not silently overwrite replay/model outputs; display which source owns control.
- [ ] Highlight the selected route and affected aperture assembly when hovering/selecting controls. Provide reset and keyboard/2D equivalents in the application UI.
- [ ] Show physical cables leaving the console rear toward phase-driver banks and aperture motor junctions; console is a controller, not an optical splitter.

**Gate:** rotating any control produces the same parameter update as the 2D editor; replay changes knob indicators; no decorative knob falsely implies an implemented actuator.

## Phase H4 — controlling fly foreleg animation (staged is approved)

- [ ] Reuse `flybody-articulated.glb` plus `joint-map.json`; inspect actual left/right foreleg node names and joint axes before selecting animation tracks. Do not rotate whole wings/legs as a substitute for articulated foreleg motion.
- [ ] Build deterministic clips/states: idle/rest, reach selector, reach knob, touch/turn, retract. Use `touch_*` and `rest_*` target nodes on the console.
- [ ] Use a short reach/contact/retract sequence (initial art-direction range 0.3–1 s per stage, tunable). Blend smoothly and maintain contact during the turn. IK is optional; authored keyframes are acceptable.
- [ ] Keep remaining legs supporting the fly; avoid hand/body penetration of the panel and large whole-body motion. Limit targets to the reachable zone after display scaling.
- [ ] Trigger gestures on selection or control activity, with rate limiting during fast replay. Knob state continues to follow controller outputs even when the gesture is still catching up.
- [ ] Label “staged operating gesture” in the inspector; do not present it as a learned motor policy or infer neural activation from gesture phase.
- [ ] Keep neural activation sourced from the existing saved/controller reservoir states. Foreleg choreography must not modify the reservoir, readout or optics.
- [ ] Respect pause, reduced motion and speed controls; dispose independent animation mixers when swapping fly instances.

**Gate:** demonstrable left/right reach, contact and retract; console value remains correct at every animation frame; fixed input sequence produces repeatable gestures; pause/reduced-motion modes retain usable controls.

## Phase H5 — mechanical motion and scientific overlays

- [ ] Drive nested tip/tilt pivots from actual per-channel pointing with a documented axis/order convention. Transform emission direction consistently with the solver-to-display mapping.
- [ ] Model focus change with a named movable lens/focus stage once the asset exists; preserve the Gaussian/q-parameter behavior in the solver. Do not use a brightness change as a focusing stand-in.
- [ ] Preserve complex-field interference and the translucent 2π dome with movable intensity sections around the power-distribution centroid, as required by the main optics spec.
- [ ] Keep diagrammatic beam envelopes separate from computed intensity; changing phase must affect interference, not falsely change an individual beam's power.
- [ ] Keep neuron activity visible/inspectable without obscuring fly controls or implying that staged foreleg motion is a biological prediction.

**Gate:** one-channel tilt, focus and phase perturbations behave correctly; change only display asset scale and verify numerical intensity and training metrics stay identical.

## Phase H6 — review evidence and handoff completion

- [ ] Add useful port/registry tests and a scene smoke check; reuse existing optics tests rather than duplicating formulas in tests.
- [ ] Capture overview, selected-channel route, connector closeup, console closeup and aperture-motion views; record a short console/foreleg demonstration.
- [ ] Record asset bytes, draw calls, triangle count and frame time on a named machine/browser; optimize visible cable tessellation and repeated hardware before adding decorative textures.
- [ ] Verify no missing textures/assets, browser console errors, or resources fetched from vendor sites at runtime.
- [ ] Update the relevant devlog with what is integrated vs still a placeholder. Do not mark this document complete just because assets exist.

**Completion:** user can inspect a wired 19-channel system, operate shared controls beside an animated fly, and trace each command to its physical device while retaining actual optics and neural-state provenance.

## Asset follow-up queue (separate from application work)

- [x] Cache manufacturer connector, adapter, phase modulator, collimator and EDFA STEP files.
- [x] Cache phase-modulator lab reference and two enclosure drawings; retain hashes/URLs/rights notes.
- [x] Generate nine original GLBs with named ports and controls plus contact sheet and wiring contract.
- [ ] Obtain an explicitly APC/PM loose-connector dimensional reference; cached 30126C3 is FC/PC.
- [ ] Obtain a direct splitter dimensional drawing/CAD; the 1×8 manufacturer web reference is not yet cached as a drawing.
- [ ] Generate seed instrument, DC supply, aperture driver junction, mounting trays and focus stage assets.
- [ ] Refine hollow connector geometry, ferrule polish angle, screws/holes, gimbal mechanics, cable strain relief and full clearance validation for hero closeups.
- [ ] If vendor-derived runtime meshes are desired, establish redistribution permission and tessellate/decimate with tool version, units and bounds checks. Current original assets avoid that dependency.
