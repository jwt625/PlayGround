# Hardware rendering review — corrective coding tasks

**Detailed implementation source of truth:** [CBC assembly specification](CBC_ASSEMBLY_SPEC.md). Use its counted BOM, placement/orientation tables, physical port schedule and connection schedule instead of the preliminary coordinates below.

**Next-pass checklist:** [coding-agent delivery brief](CODING_AGENT_NEXT_PASS.md). It distinguishes user requirements from proposed dimensions/control counts and defines three reviewable implementation passes.

2026-09-15. **Priority correction to `HARDWARE_SCENE_TASKS.md`.** Inspected the running app at the existing `127.0.0.1:5173` server, its hardware camera, and current implementation. No app code modified. The user now requires more knobs and the operating fly directly in front of the panel; the former five-knob-only recommendation is superseded.

## Evidence and findings

Screenshots: [overview](../DevLog/evidence/hardware-review/current-overview.png), [hardware preset](../DevLog/evidence/hardware-review/current-hardware.png), [isolated hardware](../DevLog/evidence/hardware-review/hardware-isolated.png).

| Priority | Finding | Code/evidence | Required correction |
|---|---|---|---|
| P0 | Inconsistent up axes: hardware lies/orients independently of the apparent floor | `scene.ts`: default Y-up camera, but GridHelper rotated into XY; `hardwareScene.ts`: Z-up GLBs cloned without basis conversion | Establish a Y-up world, horizontal XZ table and vertical XY aperture; use explicit per-asset transforms. |
| P0 | Hardware aperture is not the required hex array | `buildLayout`: `ii % 5`, `floor(ii/5)` yields rows 5+5+5+4 | Consume `channel-layout.json`; retain exact CH identity and 3+4+5+4+3 geometry. |
| P0 | Console scale changes with unrelated state | Console initial `placeAsset(..., 1.9)` uses an absolute scale; `setSelectedChannel` changes it to `HARDWARE_SCALE * 1.9` | Store an immutable base transform per instance. Highlight with outline/emissive color, never resize hardware on selection. |
| P0 | Fly is disconnected from panel and cannot operate it | Console at (-70,-6,4), learner at (-30,-18,-6); fixed +π/2 Y rotation; learner has no animation mixer | Attach fly placement to console front/stance targets, face panel, add articulated foreleg gestures. |
| P0 | Modules float in rows; no actual supporting tray/table | `trayPosition` spreads equipment in XY while each GLB has Z-up feet | Put enclosure feet on a real horizontal tray/table and provide a supported aperture frame. |
| P1 | Cabling forms a crossing fan above the experiment | Every route uses one quadratic with midpoint +10 world-Y | Use port-tangent leads, explicit tray lanes/breakouts, piecewise splines and service loops. |
| P1 | Cable attachment bypasses connector geometry | `worldPortPosition` returns equipment port, not attached plug `port_cable_exit`; placeholders fall back to box centers | Resolve terminal cable-exit anchors; require real placeholder port nodes. Never silently connect to equipment centers. |
| P1 | Wrong pigtail treatment and duplicate splitter sockets | `mateConnectors` mounts FC plug directly on PM-pigtail; adds bulkhead to splitter that already includes socket geometry | Pigtail → flexible fiber → free-end plug. Existing panel socket → one plug. Define socket ownership explicitly. |
| P1 | Full transforms aren't respected by cable vertices | World positions are inserted directly into geometry parented under `group` | Transform all evaluated points into cable-container local space before building geometry. |
| P1 | Apparent completeness exceeds mechanical realism | “78 modules / 172 connectors / 215 cables” counts instances, not correct mating/support/clearance | Gate on correctness evidence below; counts alone do not pass. |
| P1 | Hardware inspection is obscured | Hardware preset retains intensity plane, beams and neural cloud; labels overlap; oversized corner HUDs consume viewport | Hardware preset should fit bench bounds in usable viewport, suppress scientific overlays reversibly, and offer compact HUD. |
| P2 | Two competing apertures are visible | Legacy beam-emitter rings and displaced hardware mounts coexist | Make one clearly registered assembly, or explicitly separate optical inset and mechanical view. Do not suggest beams originate from a disconnected second array. |

The original v2 meshes are intentionally low-detail starting assets. Importing them does not fix layout, orientation, enclosure support or cable topology. Those must precede decorative detail.

## R1 — freeze coordinate and mounting conventions first

- [ ] World: +Y up, +Z optical propagation, X horizontal. Aperture is XY at Z≈0. Table is XZ below the optical axis. Unrotate the floor grid or replace it with an actual table aligned to this basis.
- [ ] General Z-up hardware import: map `(xa, ya, za)` to `(xa, za, -ya)` with root `Rx(-π/2)` plus uniform meter-to-display scaling. Module +Z lid normal must become world +Y. Rotate modules in yaw only to present ports toward routing lanes.
- [ ] **Mount exception:** generic tip/tilt asset emits along its own +Z, the same axis as its base normal. A blanket import transform would point its beam upward. Rebuild the mount as a horizontal collimator on a vertical support: base normal +Y world, optical rest axis +Z world. Do not rotate the entire base upright to correct the beam axis. Split base/support and optical pivot assembly transforms.
- [ ] Fly GLB is Y-up and source-forward +X. Preserve up; rotate forward toward the console using a derived heading, not an assumed +Z-forward convention. Existing target-flight heading code also assumes +Z; audit separately.
- [ ] Add temporary arrows for world up, module top/front, optical axis, port normal and fly forward; capture top/front/side orthographic evidence before removing debug arrows.
- [ ] All placement uses stored `baseMatrix`; selection cannot change scale or position. Declare exaggerated motor-angle gain in the inspector (current `MOTION_GAIN=25`) and separate it from actual solver pointing.

**Gate:** lids horizontal, feet supported, aperture axes forward, fly upright, console scale identical before/after channel selection. Numerical solver coordinates unchanged.

## R2 — pack equipment into a coherent bench

Proposed **mechanical display meters**, scaled uniformly at one root. Coordinates below are a layout starting point, not optical SI positions or a qualified manufacturing arrangement. Table top Y=0; modules are raised by foot/bounding-box offsets, not placed with body centers on the surface.

| Zone | Initial location/envelope in world meters | Packing/orientation |
|---|---|---|
| Table | X ∈ [-1.2,1.2], Z ∈ [-1.4,0.2] | Flat top, thickness/support legs; optical area toward +Z. |
| Shared instruments | Left rear, X≈-0.95, Z≈-1.15 | Seed, splitter and DC supply on distinct supported shelves; optical connector faces toward adjacent optical tray; power exits rear. |
| Channel trays | Seven columns X=-0.60…+0.60 at 0.20 spacing, three rows Z=-1.05,-0.75,-0.45 | 19 labeled channel cells (7+7+5 is acceptable for electronics only). Each cell contains amplifier plus separate cassette/driver; do not use this rectangular ordering for the aperture. |
| Per-channel cell | 0.20 wide × 0.28 deep | Amplifier (~0.16×0.13) in rear portion; phase cassette and driver side by side in remaining region after clearance checks. Cable/service corridors between rows; enlarge pitch if connectors collide. |
| Aperture frame | Center X=0, Z≈0, optical axis Y≈0.30 | Supported vertical 19-hex plate; illustrative mount pitch at least 0.055 m for the 0.045 m starting mount. Feed fibers from rear (-Z), emit forward (+Z). |
| Console | Left front, X≈-0.90, Z≈-0.20 | Front edge faces +Z, top tilted about 15° toward operator; give it an independent pedestal/clearance zone. |
| Operating fly | On panel centerline X≈-0.90, immediately beyond front edge (+Z) | Faces -Z toward panel; stance height from support contacts and foreleg reach. Not beside the console, not buried in neural cloud. |

- [ ] Implement top-down placement from a configuration table; show enclosure/connector/service-loop bounding boxes while arranging.
- [ ] Use 10–15 mm enclosure separation and ≥30 mm initial connector/cable exit clearance as **art-direction starting values**; actual bend/connector envelopes take precedence. Reserve cable corridors explicitly rather than letting routing find gaps afterward.
- [ ] Set feet just above table surface; add mounting ears/bolts where visible. Put fan/heatsink surfaces into open space; do not cover them with trays or cables.
- [ ] Keep phase and optical amplifier order visible in each selected-channel path. Group driver electronics beside the phase cassette, not as a floating third wall of boxes.
- [ ] Allocate one rear power spine with distribution blocks; one command backbone; optical row troughs; short local RF runs. No universal central point where every cable meets.
- [ ] Use stable channel labels on module front panels and small aperture labels; avoid large depth-test-disabled sprites covering neighboring channels.

**Gate:** top/side views show no gross intersections, no unsupported modules, and a clear seed→splitter→phase→amplifier→aperture chain.

## R3 — connector and spline routing implementation contract

For each logical edge retain endpoint equipment IDs **and** its actual cable terminal transforms. An equipment port can own a socket, pigtail exit or free-space aperture; these are not interchangeable.

### Endpoint geometry

- [ ] A socket gets one compatible plug. Use the mating-node transforms (`Tplug = Tsocket × Rmate × inverse(TplugMating)`) rather than assuming every mating point is the mesh origin. Oppose local +Z normals and preserve key orientation.
- [ ] Cable begins at attached plug's `port_cable_exit`, downstream of boot/strain relief. Pigtail cables instead begin at the cassette's pigtail exit and terminate at a free-end plug. Do not double up geometry on the splitter's built-in bulkheads.
- [ ] Implement explicit face-mounted port anchors on supply/junction/seed placeholders. Multiway logical edges require physical fan-out blocks with distinct sockets, or one modeled multicore harness and documented breakout mapping.
- [ ] Socket dimensions, cable diameters and module sizes share one mechanical scale. Keep control/RF/DC connectors distinct; do not attach FC plugs to electrical ports.

### Spline construction

1. Evaluate source cable exit `p0` and outward cable tangent `n0`; target exit `p1` and outward tangent `n1` in world space after current poses.
2. Create straight boot leads `a=p0+L0*n0`, `b=p1+L1*n1`. Choose L from boot length/obstacles, initially 10–30 mm for these display assets. The arrival derivative is **-n1**, not +n1.
3. Route `a` through designated row lane → trunk lane → channel breakout → `b`; never use one unconditional elevated midpoint. Each path carries a lane ID and stable channel offset.
4. Use cubic Bézier lead-in/lead-out segments (first/last handles along terminal tangents), with a C1-continuous piecewise cubic or centripetal Catmull–Rom interior. Catmull–Rom alone does not guarantee terminal tangency or collision avoidance. Round tray corners using explicit handles.
5. Add a shallow U/Ω service loop near a moving collimator and a short supported drop to its rear connector. Preserve route ordering so channels do not braid when selected or animated.
6. Transform sampled points into cable-parent local coordinates. Reuse geometry buffers; update only routes with moved endpoints. Recompute bounds when needed. Do not allocate 215 curves/geometries each frame.

### Radius, clearance and drawing

- [ ] Start with existing visual bend radii: fiber 30 mm, RF 20 mm, control/DC 25 mm; these are not certified vendor minimums. Check sampled curvature radius `R=|r'|³/|r'×r''|` (straight segments have infinite R); refine sampling near tight corners.
- [ ] Check cable centerline segments against inflated equipment bounds: inflate by cable radius plus a declared visual clearance (initially 2 mm). Exempt only the declared terminal insertion/lead corridor, not entire endpoint enclosures.
- [ ] Mesh selected/near cables as tubes (6–8 radial sides, adaptively sampled centerline; initial 32–96 segments per route). Use a lighter line representation at distance. Screen-space line width must not imply mechanical cable thickness.
- [ ] Bundle long shared runs into sleeves with visible strain-relieved breakout points. Preserve all logical edges in the registry/selection inspector even when a bundle is rendered as one tube.
- [ ] Apply deterministic lane offsets by CH order; cross cables only in explicit separated-height bridges. Keep power runs visually separate from fibers, and local RF runs short.

**Gate:** CH01/CH10/CH19 routes have correct terminal tangents, no equipment penetration, no sharp turns, and remain attached throughout tip/tilt motion. Report minimum sampled radius and clearance violations. Shared source sockets cannot contain overlapping plugs.

## R4 — replace five-knob console with a richer panel

**43 physical knobs:** 19 phase knobs + 19 amplitude knobs, plus five larger selected-channel knobs (phase, amplitude, tip, tilt, focus). The shared phase/amplitude knobs are fine-adjust mirrors of the selected channel, not independent state. Keep a compact 19-position hex selector and channel/status display.

- [ ] Generate a new console asset/version; preserve v2 for baseline comparison. Proposed panel footprint **0.52 × 0.30 m**, body depth 30–45 mm, tilted top 15°. New dimensions intentionally supersede 160×110 mm prototype.
- [ ] Arrange channel strips in two banks of 10 and 9. Each strip has two knobs, channel label, phase ring and amplitude indicator; initial strip pitch 32 mm, knob diameter 12–14 mm. Use the remaining right area for hex selector/status. Lay out actual clearance rectangles before meshing.
- [ ] Place the five larger selected-channel knobs along the near/front edge within foreleg reach (initial centers X=-0.16,-0.08,0,+0.08,+0.16 m in panel coordinates; verify against the enlarged fly rig). Add encoder pointer marks, labeled scales, collars and panel fasteners. Do not rely on color alone.
- [ ] All 43 knobs bind to one existing controller state. In replay, physical knob indicators move from saved commands. Manual edits respect existing ownership rules. Phase wraps continuously; bounded knobs stop at documented ranges.
- [ ] Route command/power harnesses from the **rear** panel into the left cable spine. Front edge remains clear for the fly's legs. Cable labels distinguish phase-bank command, motor command and power.
- [ ] Provide runtime text labels and named `knob_CHxx_phase`, `knob_CHxx_amplitude`, `knob_selected_*`, `touch_*`, `fly_stance`, support/contact anchors. Panel animation cannot overwrite actual command state.

**Gate:** all 19 phase/amplitude pairs are visible in console closeup, labels legible, selected-channel controls synchronized, and no knobs are ornamental unless explicitly disabled/labeled.

## R5 — fly directly in front, articulated legs operating controls

- [ ] Derive fly stance from console front-center, not a hardcoded world coordinate. The fly's head/forelegs face the near control row, thorax remains outside the enclosure, hind/middle legs rest on a small operator platform.
- [ ] Uniformly scale fly for display and fit foreleg reach from its articulated skeleton. Use actual contact targets to choose spacing; don't stretch limbs to meet a panel placed too far away.
- [ ] Add left/right foreleg rest→reach→touch/turn→retract clips or IK. Maintain contact during knob rotation; animate femur/tibia/tarsus chains, not only root leg rotation. Alternate legs or use one to select while the other turns.
- [ ] Forelegs operate the reachable selected-control row. Remote per-channel knob values can follow the controller without pretending the fly touched every knob. If needed, small staged lateral stance shifts reach edge controls; avoid teleportation.
- [ ] Keep torso mostly stable, supporting legs planted and wings at rest. Gesture rates are limited so high-frequency updates don't cause jitter. Pause/reduced-motion stops choreography while retaining correct controls.
- [ ] Explicitly label staged operating animation; use actual reservoir activity for neural visualization. Position neural overlay as a toggled inset/head-relative view so it doesn't hide the panel and foreleg contact.

**Gate:** console closeup video shows both forelegs reaching/contacting controls without penetration; the fly is in front in top and side views; gesture timing does not alter solver/controller values.

## Delivery order and evidence

1. [ ] **R1 + R2:** basis, frozen transforms, supported packing and exact hex aperture. Deliver top/front/side screenshots first.
2. [ ] **R3:** terminal-aware routed harnesses. Deliver selected-channel cable closeups and clearance/radius report.
3. [ ] **R4 + R5:** new 43-knob console and operating fly. Deliver closeup plus short animation clip.
4. [ ] Fix hardware camera framing and overlay defaults; review ordinary overview, hardware and optical modes at 1600×1000 and narrower viewport. Restore previous scientific visibility settings on exit.
5. [ ] Run existing optics/controller regressions and focused transform/routing tests. Asset instance counts alone are not acceptance criteria.

Review is feedback for coding agents, not an implementation-completion report. Current cached vendor references and original asset generation scripts remain available; refine assets as necessary, but do not spend time on photoreal textures before these structural corrections pass.
