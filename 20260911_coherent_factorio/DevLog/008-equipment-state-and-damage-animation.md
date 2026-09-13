# Equipment state and damage animation

Started 2026-09-12. Continue the user's request for readable power/activity/damage states. Preserve the uncommitted map/animation/technology work from 007.

## Work plan

- [ ] S1: Define state matrix and presentation-only state resolver for all nine equipment kinds and transported items.
- [ ] S2: Generate nine four-orientation condition atlases with dark LEDs/screens: intact/off, light damage, severe damage, wreck.
- [ ] S3: Generate damaged extractor/assembler motion loops, generator operation and sentry firing; generate distinct transported-item art.
- [ ] S4: Integrate power/activity/thermal/damage selection, state-aware lights/effects and item/belt feedback; keep physics authoritative.
- [ ] S5: Interactive state review, transition checks, screenshot/animation review and regression validation.

## Orthogonal state matrix

| Axis | States | Presentation rule |
| --- | --- | --- |
| Integrity | intact = 100%; light damage = 65–99%; severe = 1–64%; destroyed = 0% | Separate geometry/surface assets, not opacity alone. Damage does not silently change core production efficiency. |
| Electrical service | powered / no power / passive | Unpowered chassis has dark native LEDs/screens and frozen mechanisms. Passive junction/tuner do not acquire imaginary electrical power requirements. |
| Work | running / ready / input-starved / output-blocked / no field / uncooled | Powered-idle keeps status illumination but stops production mechanisms. Incoming field and missing cooling can coexist on a dump. |
| Thermal | nominal / hot (>65°C) / tripped / destructive (>105°C) | Heat and trip overlays remain independent of integrity; cold damaged machines do not emit smoke. Tripped machinery stops moving. |
| Combat | sentry ready / firing | Movement/recoil and flashes require real shots; do not fire idly. |
| Items | ore / assemblies / crystal / scrap | Distinct material silhouettes. Actual belt packets determine movement and queuing; no invented per-item quality state. |

## Source and rendering contract

- Nine 4×4 condition atlases: rows = intact/off, light damage/off, severe damage/off, wreck; columns = four physically redrawn orientations. No baked illuminated LEDs, active beams, smoke or fire.
- Operating machinery reuses or adds four-direction motion sheets. Damaged extractor/assembler loops retain their damage while operating; no pristine moving machine over a damaged body.
- Dynamic status lamps, port illumination, heat shimmer/smoke and firing effects remain renderer-owned, as established in the asset plan. Power loss removes emitted light instead of merely reducing opacity.
- State resolution reads core world state; no presentation imports in simulation, no image-derived collisions/ports, no new research gate or damage physics.
- Source PNGs remain unmodified. Explicit source rectangles/anchors may correct generated grid spacing. Dark-matte RGB remains a known production-art limitation.

## Acceptance cases

Power off → no lit LEDs and no work motion. Powered but no ore → lit ready indicator, stationary arm. Output full → queued belt and blocked marker, stationary drill. Partial damage → persistent visibly changed chassis, with power independent of damage. Repair → intact art. Heat trip → alarm/stalled machinery; cooldown does not erase physical damage. Destroyed → kind-specific wreck. Sentry ready → no recoil; actual shot → firing response. Simulation pause → all time-based motion/effects hold.

- Generated `assets/states/extractor-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/assembler-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/generator-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/reference-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/junction-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/tuner-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/emitter-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/dump-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/sentry-condition-v1.png`; source recorded in state-provenance.json; review pending.

- Generated `assets/states/extractor-light-cycle-v1.png`; source recorded in state-provenance.json; review pending.
