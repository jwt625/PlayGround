# DevLog 008 — hardware/fly feedback fixes

2026-09-15. Implementation pass addressing user feedback on the live scene. Simulation-only; solver optics, controller behavior and hardware renderer stay separated.

## Feedback and changes

1. **Target-fly orientation.** The moving target previously yawed only and treated
   the model's forward as +Z, so the head pointed 90° off its trajectory. It now
   takes its full orientation from the trajectory tangent plus world up: local +X
   (head) maps to the tangent and local +Y (dorsal) is projected from world up,
   with a vertical-tangent fallback. `orientationFromTangent` in
   `src/renderer/flyActors.ts` is exported and unit-tested
   (`tests/flyOrientation.test.ts`).
2. **Control fly in front of the panel.** The learner fly is now placed from the
   console/platform stance (OP-PLATFORM top Y=20, foot centroid X=-1220, Z=+30
   mechanical mm), its feet resting on the platform via its own bounding box, and
   a smaller dedicated display scale (operator length 8 vs target 16) so it fits
   the 520 mm console. It is exposed as `operatorDisplay` and asserted in the
   browser test to sit on the console centerline and in front of the panel face.
3. **Oversized panel controls.** `consolePanel.ts` built knob/pointer geometry in
   raw millimetres while positions used the panel's `unitsPerMm`, producing knobs
   ~20× too large. Geometry is now converted with the same `unitsPerMm`; channel
   knobs are 12 mm diameter, selected knobs 18 mm, selector buttons 7 mm
   (spec §8.2). Reference: `test-results/feedback-console-panel.png`.
4. **Labels, enclosure colors, fiber connectors.** Shared instruments and row
   PDUs now carry minimal nameplate sprites. Procedural enclosures use a dark
   graphite palette (metalness/roughness) instead of one flat blue-gray. Captive
   PM pigtails get a green strain-relief boot plus ferrule at the free end, and
   the existing FC/APC and SMA plugs are now at correct size. Reference:
   `test-results/feedback-hardware-bench.png`,
   `test-results/feedback-channel-connectors.png`.
5. **Retired filler geometry.** Removed the code-native placeholder bench
   (seed/splitter boxes, per-channel modulator boxes, emitter cylinders, phase
   rings, fiber/electrical lines) from `src/renderer/bench.ts`; it is now only
   the field-derived beam-envelope layer. The superseded five-knob
   `fly-console` GLB is no longer instantiated; the console is the spec'd
   procedural 520×300 body with the 43-knob panel.

### Corrected scale defect (root cause of 3/4)

Generated GLBs are authored in **metres** (up +Z) while placement is in mm.
`hardwareScene.ts` scaled GLB clones and connector plugs by `SCENE_UNITS_PER_MM`
(0.046) instead of `1000 × SCENE_UNITS_PER_MM` (46.15), so equipment and
connectors rendered ~1000× too small and were effectively invisible. Fixed with
an explicit `ASSET_SCALE`. Procedural equipment boxes now also put their bottom
on the declared support Y instead of centering on it.

## Validation

Machine: macOS, Node v23.7.0; browser checks against the running dev server.

| Command | Result |
|---|---|
| `npx tsc --noEmit` | clean |
| `npm test` | 48 passed (12 files), including 4 new orientation tests |
| `npm run build` | vite build ok |
| `CBC_INSPECTOR_URL=http://127.0.0.1:5173 npx playwright test` | 10 passed, including new `console operator fly sits in front of the panel` |
| `git diff --check` | clean |

Runtime metrics (evidence `DevLog/evidence/008-hardware-feedback-metrics.json`):
112 placed nodes, 77 asset modules, 154 mated connectors, 242 routed cables,
43 knobs, 19 selectors, 7 GLB kinds, 0 failures. Routing audit still reports a
sub-30 mm minimum sampled radius and clearance violations (unchanged open item).

Screenshots (gitignored `test-results/`): `feedback-hardware-bench.png`,
`feedback-console-fly.png`, `feedback-console-panel.png`,
`feedback-channel-connectors.png`.

## Still open

- Routing radius/clearance gate (R3) remains unmet: min sampled radius ~1 mm and
  ~12k inflated-clearance hits. Reported, not claimed.
- Console is a flat supported body, not yet the spec's wedge/pedestal; knob
  labels/scales and status lamps are not yet rendered.
- Pigtail boots are procedural strain relief, not dimensioned vendor connectors.
- Foreleg contact is still staged, not measured against the actual `touch_*`
  knob transforms.

## Verification needed (not claimed)

- Visual: confirm channel head-tangent orientation across a full fly-target
  circuit in motion, and that connectors read as FC/APC vs SMA at close range.
- Visual: confirm the operator fly does not intersect the console body at the
  enlarged stance during the foreleg gesture.

---

## Follow-up (2026-09-15): why the blue knobs are still, and foreleg operation + FX

**Reported:** the blue (amplitude) knobs never turn. **Cause (not a bug):** the
controller action space is piston/phase only. `analyticSteeringCommands` uses
`uniformCommands`, so `amplitude` is 1.0 on every channel, and the connectome
readout only adds a piston correction. `resolveActual` then scales by the hidden
gain, but with `hiddenGainScale = 0` the actual amplitude is constant. The blue
knobs are correctly bound to actual amplitude and therefore do not move. Making
them move would require adding amplitude to the action space, which is a
planning decision (`docs/CODING_AGENT_NEXT_PASS.md`), not a renderer fix. The
orange phase knobs do turn because their piston changes every control step.

**Added:** the operator fly now works the controls that are actually moving.

- `HardwareScene.updateConsole` tracks per-frame piston change and
  `activePhaseKnobs(count)` returns the hottest phase knobs with their world
  positions (`src/renderer/hardware/hardwareScene.ts`).
- `FlyActors.setForelegTargets` maps two hot knobs to the two foreleg chains.
  `aimChain` orients each articulated joint (femur → tibia → tarsus) at the
  target using per-joint rest quaternions, a scrubbing wobble around the knob
  centre, and a small supported body shift (`src/renderer/flyActors.ts`).
- **Flashy FX:** `createScene` now renders through an `EffectComposer` with an
  `AfterimagePass` (damping driven by measured foreleg joint speed) plus a mild
  `UnrealBloomPass`, toggled by the new **Motion FX** checkbox
  (`src/renderer/scene.ts`, `index.html`). Presentation-only; it does not feed
  the solver.

Evidence: `test-results/feedback-console-forelegs.png`,
`test-results/feedback-motionfx-overview.png`; metrics `activeKnobs` and
`flyForelegMotion` are exposed on `window.__cbc`.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run build` ok;
`npm run test:browser` 11 passed, including
`operator forelegs animate toward the turning phase knobs`.

Still open: the fly's reach is illustrative and does not contact the knobs with
measured IK; amplitude control remains out of the action space.

---

## Follow-up 2 (2026-09-15): reach-and-traverse, localized trails, dome off, tighter beams, layout, labels

**Forelegs now reach and travel between knobs.** The learner stance moved from
Z=+30 to Z=-40 mm and leans ~0.9 units toward the panel while operating, so the
joint-by-joint aim can actually extend to the near row. `setForelegTargets` keeps
a stable ordered working set of the hottest phase channels and cycles through it
on a 0.9 s dwell, with a small operating scrub on the resident knob. Amplitude
remains untouched (see the earlier note).

**Whole-scene motion blur removed.** Global `AfterimagePass`/`UnrealBloomPass`
were deleted (they smeared beams and overlays). Instead the arms get two additive
tip streaks built from an 18-point tarsus-tip history, so the effect is local to
the fly; the checkbox is now **Arm trails** (`src/renderer/scene.ts` reverted to a
direct renderer; `src/renderer/flyActors.ts` `updateArmTrails`).

**Dome hidden by default.** `show-dome` is unchecked and `dome.group.visible =
false` at startup.

**Beams 2x tighter (presentation).** A physical trial (`launchRadius_m` 0.18 →
0.36 mm) was rejected: the tiled-array power test showed launch-plane overlap
exceeding 192%, so it is not a valid fixture. Instead `DISPLAY.beamTighten = 0.5`
is applied consistently to the envelope cones and the measured section, which
samples a 2x wider physical window onto the same display plane. The solver
fixture is unchanged; the beam-rendering test inverts the display factor to keep
checking the physical Gaussian radius.

**Seed laser and 1x19 splitter moved** from under the bench supply to the open
table in front-right of the console (mechanical `(-780, 20, 150)` and
`(-780, 20, -60)`) so they read in the default and hardware cameras.

**Labels no longer clipped.** The nameplate canvas is now sized from
`measureText` (plus padding) and the sprite width follows the canvas aspect, so
long labels such as "Command chassis" and "Motor controller" fit.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run build` ok;
`npm run test:browser` 11 passed. Still open: contact is illustrative, not
measured IK; `beamTighten` changes the section plot's displayed spatial scale
(documented, presentation only).

---

## Follow-up 3 (2026-09-15): real reach solve, cone length, enclosure, layout, wrapping, neural scale

- **Forelegs now actually reach.** The per-joint aim was replaced with a CCD
  (cyclic coordinate descent) solve over the whole T1 chain (coxa → femur →
  tibia → tarsus1–4 → claw), and the chain root is lengthened by a declared
  `FORELEG_STRETCH = 1.9`. The operator now targets the five reachable
  front-row shared knobs (which mirror the hottest channel via a new
  `fallbackChannel`), instead of the far rear phase knobs. Measured claw-to-knob
  distance is ~0.35 units and is asserted below 0.6 in the browser test
  (`operator forelegs reach and animate at the active controls`).
  Diagnostics: `forelegDiagnostics` on `window.__cbc`.
- **Beam cones reach the plot plane again.** The `beamTighten` factor had been
  applied to the axial propagation term as well as the transverse width, halving
  the cone length. Only the transverse offset is tightened now; axial reach is
  full `DISPLAY.targetDistance`, and the beam test recovers the axial parameter
  accordingly.
- **Console body no longer intersects the panel.** The rectangular body rose
  above the low front edge of the 15° sloped panel. It is now a wedge whose top
  face is parallel to the panel (`consoleBodyGeometry`), so nothing protrudes in
  front of the panel.
- **Splitter and seed rotated 180°** about world Y (`yawRad`), with their port
  offsets/normals flipped, so the seed emits toward the splitter and the 19
  outputs face the channel rows: natural fiber flow.
- **Bench supply sits on the table.** The two floating left-rack shelf plates
  were removed and PSU moved to table level.
- **HUD note wraps** via `max-width` on `#hud`.
- **Neural 3D view** is 2x smaller (scale 21 instead of 42) and moved next to
  the operator fly at `(-56, 6, 4)`; the "Inspect 3D neurons" camera targets its
  live centre.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run build` ok;
`npm run test:browser` 11 passed. Evidence:
`test-results/feedback-console-reach.png`,
`test-results/feedback-hardware-layout.png`,
`test-results/feedback-target-plane.png`.

---

## Follow-up 4 (2026-09-15): orientation math fix and 10x faster forelegs

- **Orientation B/C were wrong (splitter upside down).** `orientationRotation`
  built B as `Euler(-90°, 180°, 0)` and C as `Euler(-90°, 90°, 0)`. With XYZ
  order that rotates the asset about its own Y *before* lifting +Z, mapping the
  lid to world -Y — upside down. It is now `A` (a -90° X rotation) followed by a
  world-Y yaw: `qY(180°)·A` for B and `qY(90°)·A` for C, exactly as the spec
  defines. This fixes the splitter and also un-tilts the 19 phase cassettes,
  whose ports are authored in that world frame. The splitter additionally uses
  orientation A so its 19 outputs face -Z toward the channel rows (the requested
  natural fiber flow) with the lid up; its ports were checked against the asset
  manifest (`+Y` output bank → world -Z, `-Y` input → +Z).
- **Forelegs 10x faster.** Knob dwell dropped from 0.9 s to 0.09 s and the
  operating scrub speed raised 10x, so the limbs travel between active controls
  ten times as fast. CCD keeps the claws on target at the higher rate.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run test:browser`
11 passed. Evidence: `test-results/feedback-splitter-seed.png`,
`test-results/feedback-console-fast-arms.png`.

---

## Follow-up 5 (2026-09-15): 3x thicker wires

The 242 routed cables were 1-pixel `THREE.Line`s (WebGL ignores
`LineBasicMaterial.linewidth`). They are now real tubes whose jacket radii are
3x the previous values (optical 3 mm, RF 4.5, motor 3.75, command/DC 6, external
12 mm; `WIRE_RADIUS_MM`). The selected-channel highlight is 1.7x the new base
radius. Same draw-call count as before (one object per run), so no measurable
load change. Evidence: `test-results/feedback-thicker-wires.png`.

---

## Follow-up 6 (2026-09-15): whole-panel operation, live array indicator, 3x faster actions

- **The fly now works all 43 knobs, not five.** `HardwareScene.allKnobTargets()`
  returns every console knob; the operator hovers just above/behind the current
  control (`applyOperatingStance`) so the stretching CCD forelegs can reach the
  rear rows too. The working set is no longer capped at 6. The stance assertion
  became "stays over the console region" and a new test verifies 43 active knobs
  and claw contact within 0.9 units (`operator forelegs work all 43 console
  knobs`).
- **Array indicator is now live.** The 19-button hex selector was a static
  selected-channel lamp. Each button is now colored by that channel's actual
  phase (`phaseColor(piston_rad)`) and brightened by its amplitude, with the
  selected channel pushed brighter. Evidence:
  `test-results/feedback-all-knobs-array-indicator.png`.
- **Actions 3x faster again.** Dwell `0.09 s → 0.03 s`, scrub speed 3x, and the
  hover lerp raised so the body keeps up.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run test:browser`
11 passed.

---

## Follow-up 7 (2026-09-15): camera pose readout, copy, trajectories, collapsible panels

- **Live camera pose** in small grey text (bottom-right Camera panel): world
  `pos`, orbit `target`, `fov`, and `yaw`/`pitch` in degrees, refreshed each
  frame (even while the simulation is paused) from the actual `PerspectiveCamera`
  and `OrbitControls.target`. The same object is exposed as
  `window.__cbc.getMetrics().camera`.
- **Copy pose** button writes a rounded JSON pose to the clipboard (with an
  `execCommand` fallback), ready to paste back as keyframes.
- **Preset trajectories** (`overview pan`, `console orbit`, `array dive`,
  `target sweep`) selectable from a dropdown with Play/Stop. Playback eases
  piecewise between keyframes, drives `position`/`target`/optional `fov`,
  suspends OrbitControls, and auto-stops at the end. Camera presets stop playback
  first.
- **Collapsible panels.** The three viewer panels (metrics HUD, Inspect scene,
  Saved MaleCNS training) each get a corner toggle that collapses them to their
  anchored edge; the new Camera panel collapses the same way.

Files: `index.html`, `src/main.ts`, `src/renderer/scene.ts` (skip
`controls.update()` while disabled). New browser tests:
`camera pose readout, copy and trajectory playback` and
`viewer panels collapse toward their anchored edge`.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run test:browser`
13 passed. Evidence: `test-results/feedback-camera-ui.png`.

---

## Follow-up 8 (2026-09-15): authored 5-key cubic trajectory

- Added the **authored 5-key fly-in** trajectory from the supplied keyframes.
  It is a `CatmullRomCurve3` (centripetal) through the five positions and five
  targets, sampled with a global **smootherstep** time warp, so the path is C1
  through every keyframe and velocity is zero at the first and last frame.
  Duration 12 s; fov 50 throughout.
- **Fixed camera re-aim during playback.** Playback set `camera.position` and
  `controls.target` but never updated the camera quaternion, so yaw/pitch (and
  the rendered view) stayed frozen while the camera translated. Both the cubic
  and piecewise playback paths now call `camera.lookAt(target)`.
- The cubic browser test now also asserts yaw changes by >30° across playback.

Validation: `npx tsc --noEmit` clean; `npm test` 48 passed; `npm run test:browser`
14 passed.

### Follow-up 8b: second authored tour

Added **authored 6-key orbit** (duration 16 s, cubic): same first two and last
keys as the 5-key fly-in, with the three supplied orbit keys inserted between,
target `(-32.685, 6.824, -47.588)` shared by most keys. Smooth start/stop and
C1 through all six keys, same as the 5-key version. The camera pose test now
asserts both authored trajectories are present.
