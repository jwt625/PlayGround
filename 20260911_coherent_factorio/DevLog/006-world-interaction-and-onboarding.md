# PHIELDWORKS — world interaction and first expedition

Started 2026-09-12. User decisions: retain direct camera control; tutorial uses the established starter outpost; creatures initially investigate, with aggression gated behind basic defenses and sustained field leakage. Equipment orientations must be separately redrawn with a fixed oblique camera, never screen-rotated artwork.

## Milestones

- [x] W1: Resource/creature selection and inspectors; PHIELDWORKS branding; backdrop dismissal.
- [x] W2: Textured terrain, decorative scatter, orientation sprite sheets; document generation prompts and provenance.
- [x] W3: Consistent camera pan/zoom, navigation buttons, minimap and viewport indicator.
- [x] W4: Categorized build palette, costs/inputs/outputs/service requirements, contextual first-run tutorial with skip/replay.
- [x] W5: Persistent actionable diagnostics with object focus; clear separation from action history.
- [x] W6: Moving investigating creatures, obstacle avoidance, explicit defense readiness, leakage-triggered aggression and basic defense.
- [ ] W7: Save compatibility, simulation/browser validation, screenshot review and playtest instructions.

## Design contracts

- Resources expose quantity, extraction conditions, locked frontier status and an extractor action. Selection is typed and independent of machinery IDs.
- Decorative props have no collisions; resource patches, machines, creatures and frontier guardian remain selectable gameplay objects. Terrain scatter must not imply invisible obstacles.
- Camera is independent of overlays and simulation. Wheel zoom anchors at cursor; buttons and minimap support discovery. WASD/arrow keys pan; middle/Alt-drag pan; Home recenters. This milestone does not imply RTS unit selection, train scheduling, robots or every Factorio shortcut.
- Baseline errors are always visible. Future instrumentation upgrades can add precision, trends and automatic diagnosis without hiding basic requirements.
- Onboarding floats near the relevant control, completes from observable world/UI state, and can be skipped/replayed. Avoid a modal at every step.
- Existing bend radiation makes an immediate aggression trigger unfair. Gate aggression behind an explicit defense deployment milestone, then use sustained leakage exposure with readable warnings and a grace period. Save this state. Investigation remains visible before that gate.
- New sprite sheets contain four separately rendered orientations at fixed camera/light, selected by source rectangle. No canvas rotation or mirroring of machinery sprites. Generated images remain replaceable presentation assets.

## Asset batch

Terrain: seamless muted basalt/gravel substrate with low contrast. Dressing: transparent atlas of rocks, sparse alien scrub, crates and cable spools. Creature: investigating crawler. Machinery: one four-view turnaround sheet per existing equipment kind, using original sprite as identity reference. Exact prompts and outputs go in assets/world-generation-notes.md.

## Deferred

Playable characters; complex combat/ecology; physical terrain collision beyond machines; terrain painting; material-calibrated radiation ecology; upgraded diagnostic instruments; full orientation-specific animated machinery.

## Implementation checkpoint

- Implemented schema v3 with bounded ecology state and v1/v2 migration. Creatures patrol, investigate field sources, avoid machine interiors, flee close sentries, then attack after the explicit gate. Sentries require an actual wire and apply conventional defense damage independently of the field energy ledger. Wildlife damage invalidates qualification.
- Exposed typed resource/creature/guardian selection, auto-revealed inspectors, reserve-patch extraction action, active diagnostics with object focus and detected/resolved/action history. Existing events survive save/load. Basic diagnostics remain available from the start.
- Added cursor-anchored wheel zoom, buttons, full-sector fit, minimap drag/keyboard navigation, WASD, Q pick/cancel, and category keyboard navigation. New camera inputs do not mutate simulation.
- Added eleven tutorial lessons around the operating starter outpost; progress uses actual UI/world state, with Back, Skip, Replay and area focus. Suggested first sentry at (22, 10) covers the eastern field branches.
- Generated nine equipment turnaround sheets, one terrain tile, a four-prop atlas and a four-direction crawler atlas. These are fixed-camera redraws selected by atlas rectangle. No equipment sprite rotation or mirroring remains.
- Asset-generation limitation discovered and recorded: current built-in outputs were opaque RGB despite transparent prompts. Rejected painted checkerboard variants; generated dark matte versions instead. Runtime lighten compositing integrates dark-backed sheets without modifying generated source pixels. These are prototype art assets, not finished alpha-cutout production sprites.
- Initial validation: 35 simulation tests pass; original three browser gameplay scenarios pass. Expanded browser coverage is running for resources, navigation, ecology, diagnostics, and onboarding. One subpixel camera assertion was corrected to allow browser pointer-coordinate rounding (<1px).


## Human playtest checklist

1. Open `http://127.0.0.1:5173` and click **Tutorial** to replay the first-run outpost flow. Verify that floating panels leave their target controls accessible; Skip/Replay should work without resetting the factory.
2. Inspect the reserve deposit at (3–7, 17–20), then the assembler. Check remaining ore, explicit power/recipe requirements and the extractor action. Inspector selection should be brought into view.
3. Zoom at a chosen port with the wheel, pan with WASD/middle-drag, travel using the minimap, and press Home. Map fits all 64×36 tiles. Field overlay changes should preserve camera position.
4. Choose categories and compare equipment requirements. Place a disconnected phase tuner and press R through all four orientations: the camera stays upright while machine geometry, footprint and ports change. Check [the full turnaround gallery](../assets/world/index.html) at higher resolution.
5. Watch small crawlers move and investigate. The large armored guardian is intentionally stationary. Build a sentry at (22, 10), wire BUS OUT → POWER IN, and inspect a crawler's gate/exposure/grace fields. Extend defense coverage if equipment is exposed.
6. Build an unpowered machine, open Diagnostics, and use Locate & inspect. Connect power and confirm the fault clears and appears in history. Tight field bends, exposed ports, heating and damaged equipment also have remedies.
7. Complete the second field branch and tune/control it. Run commissioning. A creature may land a hit before a sentry kills it, invalidating the test; allow threats to clear, repair damaged equipment, and retry. This is an intended recoverable gameplay event.
8. Save/load and verify routes, rotations, inventory and ecology resume. Qualification intentionally requires a fresh test. Open the manual and click outside it to dismiss.

## Next iterations

- Tune investigation range, exposure thresholds, grace timing and defensive coverage using human pacing; currently these are explicit gameplay constants, not calibrated radiation biology.
- Obtain production alpha cutouts, improve per-view port/art registration, add walk cycles and machinery damage/operating variants. Current matte-composited art can lose dark details against textured ground.
- Build segment editing, route selection, splitters/crossing-layer controls and clearer radius previews; stored transport paths and core ports are ready for this.
- Add optional diagnostics trends/precision/automation as progression without hiding baseline errors. Characters, teams and RTS commands remain deferred under the accepted direct-camera decision.
