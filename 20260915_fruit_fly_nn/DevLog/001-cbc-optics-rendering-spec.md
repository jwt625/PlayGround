# 001 — CBC optics and rendering decisions

Date: 2026-09-15
Status: approved direction; numerical defaults below are implementation proposals.

This addendum supersedes conflicting defaults in [000](000-initial-discussion.md). The original connectome, learning, reproducibility, ablation, and video requirements remain in force. Everything is simulated; no physical laser control interface.

## Confirmed decisions

- Default array: 19 channels, hexagonally packed, rows 3 + 4 + 5 + 4 + 3. Keep geometry configurable.
- Optical layout: common coherent seed → splitter → per-channel phase modulator and amplifier → collimation/pointing/focus optics → tiled free-space array.
- Each channel supports piston phase, amplitude, independent tip/tilt, and focus. Initial fly training controls piston phase; later curricula expand control.
- Target ultimately moves in a 3D volume; initial training uses a fixed-distance plane.
- Both a physical 3D control console and an interactive screen-space schematic, with explicit wiring and optical paths.
- Translucent beams plus movable intensity cross-sections near the power distribution's center of gravity on a forward 2π dome.
- This agent maintains specifications/tasks, reviews implementation and validation evidence, and prepares rendering assets/references. A separate coding-agent handoff is in `docs/CODING_TASKS.md`.

## Geometry, units, and channel identity

Use SI units internally. Array lies in XY; forward propagation is +Z. Camera/display scaling must not change optical coordinates.

Hex axial coordinates: integer q,r with max(|q|, |r|, |q+r|) ≤ 2.

    x = pitch * (q + r/2)
    y = pitch * sqrt(3)/2 * r
    z = 0

Assign CH01…CH19 in rows from +Y to −Y, then ascending X. Store IDs explicitly and use the same mapping in commands, wires, geometry, logs, and learned outputs.

Proposed first numerical fixture: wavelength 1550 nm, pitch 0.5 mm, launch 1/e² intensity radius 0.18 mm, nominal target Z = 1 m, training steering limited initially to ±2°. These are simulation defaults to validate, not vendor specifications. State aperture truncation and test grid convergence before accepting them. Mechanical housing geometry may be enlarged for readability; label this display scaling.

## Optical contract

One source of truth supplies fields, intensity maps, metrics, beam envelopes, and actuator telemetry. Never animate beams toward the target independently of actuator state.

Use phasors with physical field Re{U exp(−iωt)}, propagation exp(+ikz). In each channel's launch aperture, include:

    U_n(x,y,0) = amplitude_envelope_n(x,y)
                 * exp{i[piston_n + k(sx_n*dx + sy_n*dy)
                          − k*C_n*(dx²+dy²)/2]}

Here dx,dy are relative to that emitter; C is signed convergence in m⁻¹ (C=0 is collimated, C>0 adds converging curvature). Tip/tilt is parameterized by a normalized propagation direction; the UI may show angular controls. Do not conflate piston, phase slope, and curvature. A curvature command is not identical to the final waist distance for a finite Gaussian beam; compute the propagated waist using the complex beam parameter.

Normalize launch amplitude to declared channel power and aperture convention. Sum complex fields before squaring:

    U(r) = sum_n U_n(r)
    I(r) ∝ |U(r)|²

Use a coherent common polarization for V1. Scalar/paraxial approximation and angle limits must be documented. A fast Gaussian-beam evaluator must include envelope spreading, phase curvature, propagation phase, and Gouy phase, with piston referenced consistently to the launch aperture. Cross-check it against an independently implemented angular-spectrum or Fresnel propagation reference within the declared domain. Finite-aperture truncation belongs in the reference; report where the Gaussian approximation diverges.

For forward angular spectra use exp[−ik(sx*x + sy*y)] in the aperture transform with this phasor convention. Thus positive launch phase slope steers toward positive transverse direction. Establish signs through tests rather than copying the original discussion's alternative convention. For finite targets, align propagated channel phases at the target; geometric −k*path length alone omits Gaussian phase terms.

Hidden piston offsets, drift, gain errors, quantization, limits, and optional coupling act between commands and actual actuator state. Manual controls and learned controllers use the same command path. Hidden state is ground-truth telemetry only, never implicit learner input.

## Dome and center of gravity

Interpret 2π as a forward hemisphere in solid angle, centered on the array. Default dome colors represent far-field angular power density dP/dΩ, not a claim that a nearby spherical surface is in the far field. A separate finite-radius sphere mode may be added, but must use propagated fields and normal flux and be labeled distinctly.

For unit direction s and angular power density W(s):

    P = integral_hemisphere W(s) dΩ
    m = integral_hemisphere s*W(s) dΩ / P
    s_c = m / |m|

Use proper solid-angle quadrature: uniform azimuth/elevation pixels do not have equal weights. Display both s_c and |m| (directional concentration). Mark centroid unavailable if P or |m| is below tolerance. The centroid can fall between lobes: show the strongest-lobe marker separately. No peak snapping disguised as a centroid.

The entire dome is a navigation/display surface. Mask unsupported directions explicitly; do not extrapolate a paraxial model across grazing angles or renormalize a cropped patch as if it contained all emitted power. Expose angular coverage and captured-power estimates. Implement a suitable wide-angle reference before claiming quantitatively valid full-hemisphere power.

## Movable cross-sections

- Default section center: R*s_c; normal: s_c. R is an explicit range slider, initially target range.
- User can change range, translate, rotate, resize, freeze, and reset the section.
- Optional longitudinal section spans s_c and a stable perpendicular axis, exposing convergence and divergence.
- Follow modes: power centroid (default), strongest lobe, target, manual. Label the active mode.
- Samples on each section come from the complex-field evaluator at actual world coordinates. The dome texture must not be stretched onto the section.
- Show target, angular centroid projection, local plane intensity centroid, peak, and PIB bucket as distinct markers. Include raw numeric intensity and linear/log display options.
- Camera-follow smoothing may improve readability, but cannot alter physics, metric values, or logged centroid.

## Board, schematic, and behavior

Draw a master seed, splitter tree, 19 numbered optical branches, phase/amplitude stages, pointing/focus actuators, emitters, and a simulated detector/observation return. Distinguish optical fibers, electrical command wiring, and observation/data paths through line style and labels as well as color.

The physical console has 19 phase/status cells and a selected-channel inspector for all controls. The schematic exposes the same channels without crowding the 3D bench. Selecting any emitter, wire, or board cell highlights the complete corresponding path in both views. Include a collapse/expand control for the 19 branches.

Manual, analytic reference, SPGD, and connectome modes must be clearly identified. Manual takeover pauses that channel's automatic command ownership; resuming automation is explicit. Reset/randomize/disable/solo must be reproducible and logged. Solo inspection must distinguish visual isolation from actually disabling other emitters.

Animated electrical pulses reflect real command updates; neural activity reflects simulation state. Illustrative propagation delays are labeled. Optical carrier oscillations at 1550 nm are not realtime-visible: optional wavefront/phase overlays use labeled slowed time. Translucent envelopes aid inspection; overlap brightness must not be presented as interference unless it is evaluated from the summed field. False-color visible beams represent the simulated infrared light.

## Metrics and controls

Retain PIB, Strehl, pointing error, phase error, reward, and training telemetry. Report angular error on the dome and transverse/range error at finite targets with units. Compute metrics from linear physical samples, before tone mapping or bloom.

PIB integrates physical flux over a declared target-plane bucket and divides by a declared total-power reference; report finite numerical-window capture separately. Strehl reference uses the same power, geometry, target, aperture, and allowed actuators; changing focus or steering must not silently change its denominator. Phase RMS removes global piston and compares against target-specific propagated phase alignment.

Expose command versus actual values and hidden-error visibility separately. A global piston change must leave all intensity views invariant.

## Completion sequence

1. CPU optics reference, fixture configuration, and numerical validation.
2. Fast field evaluator and SPGD demonstrated without elaborate art.
3. Functional board/schematic and 19-channel scene with measured intensity sections.
4. Dome, centroid following, finite-range steering/focus, visual validation.
5. Connectome and learning integrations following the original experiment, then full controls/3D tracking curriculum.
6. Licensed fly assets, scene polish, reproducible capture and final videos.

See the task document for evidence required at each gate. No code or numerical validation existed at this specification's creation.

## Asset and data progress follow-up

See [002 — rendering assets and external resource cache](002-assets-and-resource-cache-progress.md) for generated fly/bench assets, actual MaleCNS v1.0 downloads, explicit graph filtering, validation evidence, and remaining integration tasks. The specification above remains the acceptance contract; collected assets and data do not by themselves complete application phases.
