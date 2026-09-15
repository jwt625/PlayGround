# Coding agent: next rendering pass

**Implementation source of truth:** [CBC assembly specification](CBC_ASSEMBLY_SPEC.md). It supplies equipment counts, every channel's placement, orientations, physical port coordinates, distribution assignments and the 241-run internal cable schedule. This brief is only the review sequence; where its approximate dimensions differ, the assembly specification wins.

## Objective

Make the hardware read as a supported, connected optical experiment, with the controlling fly directly in front of a usable physical panel. Fix orientation and composition before adding detail. This is reviewer feedback and implementation guidance; no implementation is supplied here.

Detailed reference: [rendering review R1–R5](HARDWARE_RENDERING_REVIEW.md). Baseline evidence: [isolated hardware](../DevLog/evidence/hardware-review/hardware-isolated.png).

## Requirements versus recommendations

**User requirements:** realistic equipment packaging, correct orientation/layout, detailed connections and spline cabling, more control knobs, and the fly directly in front of the panel with animated legs. Retain the approved 19-channel hex array and actual optical/controller behavior. Staged leg animation is acceptable.

**Reviewer recommendations:** 43 knobs, a 0.52 × 0.30 m panel, the proposed bench coordinates, tray dimensions, routing radii, and 15° panel tilt. These are starting designs, not user-specified dimensions or validated mechanical clearances. Adjust them to make the assembly fit and remain readable; explain material deviations in the implementation report. Do not force equipment into an overlapping layout to match a suggested coordinate.

## Pass 1 — orientation, packing, and framing

- [ ] Establish one world-up convention and explicit import transforms. Show the table horizontally, equipment lids upward, and the aperture vertically with beams leaving its front.
- [ ] Correct the tip/tilt model's relationship between its base and optical axis. A correctly oriented lens on an upright base is still an incorrect mount.
- [ ] Use the canonical 3+4+5+4+3 aperture layout. Rectangular electronics trays are fine; a rectangular optical aperture is not.
- [ ] Add table/trays/aperture supports. Position equipment from foot contact and actual bounds, not mesh origins.
- [ ] Remove channel-selection scaling of equipment. Keep the console at one consistent size.
- [ ] Establish an operator composition now: panel and fly together at the bench's front-left, fly centered on the panel's front edge, facing its controls. Set aside an unobstructed camera view of the forelegs.
- [ ] Fit the hardware camera to equipment bounds within the space left by the HUD. Temporarily hide beam/section/neural overlays for hardware inspection, restoring their previous visibility afterward.

**Submit:** top, front and side views with temporary orientation arrows; one clean three-quarter view. Include the full tabletop, aperture support, console and fly. Do not submit only a distant image where incorrect orientations cannot be seen.

**Reject if:** equipment floats, feet point sideways, two unregistered apertures compete visually, the fly faces away from the panel, or the scientific plane hides the bench.

## Pass 2 — finish one channel before repeating nineteen

Use CH10 as the first complete example, then validate CH01 and CH19 after replication.

- [ ] Show separate phase cassette, electrical driver, optical amplifier and collimator mount, with readable IN/OUT/channel labels.
- [ ] Trace optical input through each actual terminal in the correct order. Show cassette pigtails, compatible plug/socket mating, and connector boots. Avoid duplicate bulkheads.
- [ ] Trace electrical command from panel through the driver to the phase cassette; trace aperture command through the motor driver to the two tip/tilt actuators. Provide explicit power distribution.
- [ ] Connect cable geometry to terminal cable-exit anchors. No cable may emerge from an enclosure center or start inside a boot.
- [ ] Route straight tangent leads into tray lanes, rounded corners, channel breakouts and a service loop at moving optics. A single curved arch between equipment centers is not sufficient.
- [ ] Keep module lids and heat sinks clear. Support long cable runs and make any crossing deliberate, with vertical separation.
- [ ] Sweep tip/tilt and check both cable attachment and service-loop deformation. Preserve actual channel IDs during highlighting and motion.

**Submit:** selected-channel overview, connector closeup, rear-aperture closeup, and motion clip. Report routing-radius and equipment-clearance violations, including unresolved exceptions.

**Reject if:** the path is only indicated by color, a plug sits directly on a pigtail exit, cables penetrate cases, several plugs share one socket, or endpoints detach when optics move.

## Pass 3 — panel and operating fly

- [ ] Provide visibly more controls than the five-knob prototype. Recommended arrangement: 19 phase/amplitude knob pairs plus five larger selected-channel controls, with a compact hex channel selector.
- [ ] Add physical knob pointers, collars, labels and scales. Bind all duplicate controls to the same state; do not invent additional independent parameters.
- [ ] Keep a reachable control row at the front. Put connector harness exits at the rear, away from the forelegs.
- [ ] Derive fly placement from panel stance/contact targets. Enlarge the fly uniformly for display; do not stretch individual limbs to compensate for poor placement.
- [ ] Animate articulated foreleg reach, contact, turn and retract, with the other legs supported. The fly should appear to operate the controls, not wave nearby.
- [ ] Maintain contact during the turn and keep head/body out of the enclosure. Use reachable shared controls for gestures; remote knob indicators may follow controller values without a gesture for every update.
- [ ] Keep staged animation separate from actual neural activity and optical commands. Preserve pause and reduced-motion behavior.

**Submit:** a console closeup still and a short continuous clip showing both forelegs, knob indications, and support contacts. Include a side view proving that the fly is in front and can reach the surface.

**Reject if:** the fly is beside the panel, hands pass through it, all legs float, knobs are decorative while claiming control, or neural activity is synthesized from the gesture animation.

## Review evidence and claim calibration

- Baseline screenshots confirm cluttered composition, disconnected operator placement and overlapping labels. The source review identifies specific causes, including the rectangular aperture placement and inconsistent scale paths.
- The console's 1.9 versus 114 scale paths are a code-level inconsistency; a 60× jump was not measured in an interactive reproduction. Add a selection/no-selection check before claiming the runtime symptom is fixed.
- World-space cable points under a group are a latent transform defect when the group has a nonidentity transform. They need not be visibly wrong while the group is identity. Verify with a translated/rotated parent.
- Counts of models, connectors and logical cables do not demonstrate correct assembly or routing. Report visual and geometric acceptance evidence separately from counts.

## Completion report format

For each pass, list completed checklist items, evidence links, remaining placeholders and any changed design assumptions. Clearly separate assets loaded, behavior implemented and behavior verified. Preserve the existing scientific regression checks. Check occupied ports before starting a preview; reuse the project server where appropriate.

The reviewer will assess the next supplied implementation/evidence. This document does not claim a newer implementation has been inspected or that feedback has been delivered through a separate agent-messaging system.

---

## Implementation status (coding agent, 2026-09-15)

Implemented against `CBC_ASSEMBLY_SPEC.md`:

- **Equipment/packing (Pass 1, partial):** typed registry in
  `src/renderer/hardware/assemblySpec.ts` (shared instruments, 19 cells,
  canonical 3+4+5+4+3 aperture, port schedule, 241-run cable schedule);
  `hardwareScene.ts` builds the supported bench, registers the aperture to the
  optical array origin, and uses single per-instance base transforms (no
  selection-time scaling).
- **Cabling (Pass 2, partial):** terminal-aware plugs/leads, family lanes, Ω
  service loops, selected-channel tube geometry. Routing audit is exposed but
  **fails the 30 mm radius / clearance gate** (min radius ~2.67 mm).
- **Console (Pass 3, mostly):** 43-knob panel + 19-button hex selector + display,
  bound to actual command state.
- **Operator fly (Pass 3, partial):** placed in front of the console on the
  platform, wings at rest, staged foreleg gesture on the six T1 joints.

Not done: orientation-arrow evidence and clean Pass 1 framing; CH10 labels and
the CH01/CH10/CH19 motion-attachment gate; routing-radius/clearance compliance;
foreleg contact measured against real `touch_*` targets and a gesture clip; H6
evidence bundle. Counts and audits are implemented; visual/geometric acceptance
is not claimed.
