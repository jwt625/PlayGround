# DevLog 006 — live hardware rendering review

2026-09-15. Inspected existing port 5173 after checking listening processes; used isolated browser sessions and opened no server. Captured overview, hardware preset and isolated-hardware evidence in `DevLog/evidence/hardware-review/`. Read current scene, hardware placement/cabling, and fly actor code. No application implementation changed.

[Corrective coding-agent handoff](../docs/HARDWARE_RENDERING_REVIEW.md) now takes priority over the earlier compact five-knob specification.

Confirmed problems: mixed world/asset up axes; hardware aperture packed as 5+5+5+4 instead of the required hex arrangement; console initial scale 1.9 versus later 114; modules without supporting trays; learner fly at unrelated fixed coordinates with no operating animation; midpoint-only cable arches; equipment-center fallbacks; cables attached before connector boots; direct plug attachment to pigtail exits; duplicate splitter bulkheads; scientific overlays and HUD obscuring equipment inspection.

Specified corrections: consistent Y-up scene, per-asset orientation including rebuilt horizontal tip/tilt optics, supported 19-channel equipment cells, actual hex aperture, immutable base transforms, endpoint-tangent spline leads, tray lanes and breakout bundles, service loops, sampled bend-radius/clearance checks, 43-knob panel, and fly stance derived from panel front-center with articulated foreleg gestures. The animation remains staged; the controller and neural activity remain data-driven.

Next coding-agent delivery: coordinate/packing screenshots first; then cabling closeups/report; then console/foreleg video. Instance counts alone do not establish correct hardware integration. Asset refinement tasks are listed in A8 of the asset plan.

Follow-up: added [next-pass delivery brief](../docs/CODING_AGENT_NEXT_PASS.md), separating user requirements from reviewer-proposed knob counts/dimensions and defining rejection criteria for each pass. Clarified that the scale discrepancy is source-confirmed rather than a measured interactive jump, and that the cable parent-space issue is conditional on a nonidentity parent transform. This follow-up changed documentation only; no code or assets were generated or modified.
