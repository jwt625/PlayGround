# R6 port construction, channel identity and overlap repair

> **Implementation follow-up:** [R6-P1 geometry repair and compute plan](R6P1_geometry_repair_and_compute_plan.md) identifies material-composition, uniformity and termination defects in the first port implementation. Correct them before transfer/reciprocity measurements. The channel and overlap definitions below remain applicable; the follow-up defines the immediate diagnostic and compute allocation.

Review date: 2026-09-19. Reviewed `V1_R6a_results.md`, `r6_reciprocal.py`, `validate_port.py`, source/port implementations, provenance, and the repaired isolated-mode and reverse-field figures. Documentation only; no solver runs or implementation edits.

**Direction: build an effectively isolated uniform receiving-guide port, repair incident-power and Gaussian-overlap extraction, then repeat R6. Hz remains primary; retain one Ez control. Proceed independently with T02 digitization now.** These are complementary workstreams. No further orchestration approval is needed for the bounded port construction and checks below.

## 1. What is accepted and what remains provisional

The isolated-profile rendering defects are fixed in the inspected image: the core is shaded on the position axis and the spurious tail-to-core diagonals are gone. The reported same-aperture isolated-guide result is a useful normalization check. However, `validate_port.py:part_a` specifically uses an axis-aligned Ez guide at resolution 100; it does not by itself validate both oblique polarizations at resolution 25. Carry that distinction into the report.

The new same-aperture coupler comparison removes the previous aperture mismatch. The 8.6% residual remains unexplained/incompletely decomposed, and the receiving-guide identity is still open. Band 2's index alone does not establish its spatial radiation character; export its fields too. Preserve 0.70% as a candidate-channel result until identification passes.

The reverse figures do show a broad emitted field, apparently concentrated near the small-gap/upstream termination. That is a useful qualitative observation about the current launch. It does not yet establish the emission pattern of a pure receiving-guide excitation, nor a quantitative Gaussian mismatch.

## 2. Concrete problems in the current reciprocal calculation

| Current implementation | Consequence | Required correction |
|---|---|---|
| Source at s=31; input monitor at s=27 | Propagation occurs through a changing, radiating wedge before the supposed input measurement. The local modal bases differ. | Put the input monitor between source and device in the same uniform port. Calibrate the launched channel there. |
| `abs(native_flux)` used as input | Net flux includes opposing directions and other channels; its absolute value is not incident guide power. | Use the identified incoming modal coefficient, with a source-only reference and separate outgoing coefficient. |
| Scalar Ez/Hz overlap with a real Gaussian | Omits the calibrated beam phase curvature and electromagnetic power weighting; particularly unsuitable for a broad angular spectrum. | Use calibrated complex E/H and a power-normalized electromagnetic overlap. |
| Pixel mask `Sx < 0`; sum only negative local flux | Local net Poynting sign is not a directional-wave decomposition. Interference/backflow and clipping make this a nonlinear power selection. | Separate incoming/outgoing angular spectra using E/H in homogeneous silicon, then integrate outgoing power and overlap. |
| Sampling the original source plane and entire cell-height DFT column | Forward reference fields on a current sheet are ambiguous; some samples lie in PML. | Use a common source-free, homogeneous-Si reference plane and a finite aperture entirely outside PML. |

The small Ez coefficient (0.40) consequently does **not** by itself prove that band 1 at the source is not the guide mode. It establishes failure to identify/normalize a matching channel after propagation to a different cross section. Likewise, `M≈0.08` is currently a scalar-profile statistic, not an accepted Gaussian-channel power fraction. Retire its physical interpretation until recomputed.

The source pulse bandwidth differs between forward and reverse scripts. This is permissible if each frequency-domain input is independently calibrated, but raw DFT powers must not be reused between runs or polarizations. Use a separate incident reference for each branch and numerical configuration.

Meep normalizes propagating modal coefficients to power and defines overlaps using both electric and magnetic fields. Use its documented conventions consistently rather than adding an extra factor of two or a cosine correction. [Meep mode decomposition and overlap definition](https://meep.readthedocs.io/en/latest/Mode_Decomposition/).

## 3. Channel definition and a reproducible localization rule

The channel is the fundamental of the isolated **air / Si(262 nm) / oxide** strip, at the actual output-guide orientation and chosen polarization. Band number is a solver index to record after identification, not the definition.

Use guide-normal coordinate q with q=0 at the strip midplane. Start with a fixed observation window **|q| ≤1.5 µm** for each polarization and every candidate. This is a proposed diagnostic convention, not a paper parameter.

1. Export complex E/H and dielectric profiles on the same physical coordinates. Rotate vector components into a common guide frame and compare modes traveling in the same direction; reverse the reference consistently when necessary.
2. Report `Gamma_strip = integral_strip U / integral_window U`, with `U proportional to epsilon|E|² + mu|H|²` for these nondispersive materials. Report substrate and oxide/air participation in the same window separately. Use actual material masks and consistent interface quadrature.
3. Compute normalized complex full-field similarity to the isolated fundamental. One suitable identity metric is the squared normalized inner product of `(sqrt(epsilon_ref) E, sqrt(mu_ref) H)`, with the **same positive reference weights** in both norms. This is an identity metric, not transmitted power. Use the electromagnetic flux overlap for power.
4. At the effectively isolated port, require squared similarity **≥0.99**, strip participation within **0.01 absolute** of the isolated reference, and convergence of neff toward the isolated reference (target |delta neff| ≤0.005 after spatial convergence). Compare numerical references at matched discretization as well as against the analytic limit.
5. Enlarge the identity window to **|q| ≤2.0 µm**, independently enlarge the solver domain, and check that the selected field, participation and identity remain stable. At the isolated port, use 0.001 absolute as the initial window-stability target for Gamma and squared similarity; increase the window if needed. Final coupling must also satisfy its own aperture-convergence criterion.

**Do not impose a universal “more than 50% of energy in silicon” threshold.** Ez and Hz have different confinement, and loading changes the distribution. Reference-relative identity and independent domain convergence are stronger than an arbitrary absolute localization cutoff.

For exploratory loaded-wedge modes, report Gamma and overlap and track fields between nearby gaps, refining steps where branches mix. Do not apply the isolated-port 0.99 threshold to strongly loaded resonances or normalize leaky modes over an expanding global substrate. If there is no unique guide-associated branch, report the mixing rather than forcing a label. Full loaded-resonance identification is not required to start R6 with a validated isolated port.

## 4. Approved port construction: named numerical continuation

Name the variant **isolated-output-port-v1**. Keep the nominal interaction, original physical output reference at s=31, 262 nm thickness and 53.5° guide orientation unchanged. Retain the original reconstruction as a separate configuration.

- Begin modifications only downstream of the existing physical guide extent, **s=33 µm** for the nominal configuration. Continue the straight strip without changing its thickness or air-side cladding.
- Smoothly recede the substrate/extend the oxide into a uniform section with interfaces parallel to the guide. Start with **3 µm oxide clearance**, measured normally from the strip's substrate-facing surface. This is a numerical isolation parameter, not inferred paper geometry. Check **4 µm** as an independent isolation test.
- Start with an **8 µm transition length** along the guide and test 16 µm. Preserve the strip axis while changing only the downstream support. Export all vertices and coordinate conventions; no abrupt material step at the splice.
- Place both launch and incident-mode monitor wholly within the resulting uniform section. Start with the monitor at least 3 µm from the transition end, the source another 2 µm farther downstream, and at least 3 µm non-PML clearance beyond the source. Extend all port materials consistently into PML. Increase separations if required by the sensitivity tests.
- Size source/modal apertures from the converged isolated-mode tails, rather than reusing a 12 µm vertical line that intersects substrate. The |q|≤1.5 µm initial window corresponds to a vertical-line half-span of about **2.52 µm** at 53.5°. Enlarge to the |q|≤2 µm check while keeping the full eigenmode aperture inside air/strip/oxide and outside PML.
- Rerun **both directions in exactly this same geometry**. Do not compare reverse isolated-port efficiency with the historical forward loaded-port number.

The new port defines a clean scattering channel, but its continuation may introduce loss or feedback. Reciprocity of the modified structure alone does not validate the original paper reference. Report the transfer between s=31 and the isolated port; do not remove an arbitrary scalar loss factor when reflections or multiple channels are present.

Acceptance: source-only launch purity ≥99% as an initial check, incoming-channel normalization reproducible within 1%, same-aperture native/modal agreement ≤1% in the pure-guide reference for **both** polarizations, and final target-channel contamination ≤1e-4 of input power. Transition length, oxide clearance, monitor/source position and PML changes must alter eta by ≤`max(1e-4 absolute, 5% relative)` for this diagnostic. Record actual differences. If the port transfer is not stable/negligible, retain it as a named diagnostic geometry and do not call its efficiency the original pre-bend result.

## 5. Gaussian channel and minimal checks before the full pair

Use **x=-10 µm** as an initial common reference plane for the nominal geometry: it is downstream of the original forward source at x=-12 and upstream of the interaction. Verify homogeneous silicon and PML clearance over the entire chosen aperture in the sampled dielectric. Change the plane if those conditions fail.

- Save the actual forward-reference complex E/H at 1550 nm on that plane for each polarization. Normalize its incoming power. Build the reverse target from its time reverse: for the usual phasor convention, E becomes E* and H becomes -H*, with the propagation direction reversed.
- Extract outgoing propagating components from the reverse E/H angular spectrum. Use the appropriate polarization-dependent plane-wave admittance and normal-flux weights; treat evanescent components separately. Do not select directions by real-space flux sign.
- Compute a complex, power-normalized overlap over the same converged aperture/basis. The target is the calibrated incident beam; fitting a new center, angle or waist is a separate later diagnostic.
- Before a full coupler run, recover unit target overlap for the calibrated target itself and recover known amplitudes from a superposition of incoming and outgoing waves. Check aperture and reference-plane invariance. These checks specifically detect sign, phase, admittance and direction-separation errors.
- Normalize reverse target power by the incident receiving-guide channel at the uniform port. Normalize forward guided power by its own calibrated incident Gaussian. Save complex amplitudes, individual directions and raw fields in immutable run records.

Then require direct/reverse agreement within **max(1e-4 absolute, 5% of the larger efficiency)**. First debug this on the same coarse mesh and fixed geometry; only after extraction is consistent undertake the 50/75-per-µm spatial-convergence runs. Mesh convergence remains necessary for physical acceptance even if reciprocity passes on a coarse grid.

## 6. Execution order and stopping condition

| Task | Completion evidence |
|---|---|
| R6-P1: isolated port and channel | Dielectric/complex-mode plots, reference-relative metrics, branch identity, Hz/Ez straight-oblique source calibration, continuation geometry manifest. |
| R6-P2: overlap/normalization repair | Direction-separation checks, unit-target check, phase-correct Gaussian reference, same-aperture power accounting and raw arrays. |
| R6-P3: new matched pair | Forward/reverse in isolated-output-port-v1, input/output budgets, reciprocity residual and continuation sensitivity. No reuse of the old channel denominator. |
| R6-P4: physical acceptance | Two finer meshes and remaining PML/time/aperture tests; then decide compact mismatch study versus a mechanism-directed calibration. |
| T02: independent work | Digitized paper targets, source pixel coordinates, uncertainty and overlays on cached panels. No dependency on R6. |

The existing provenance Markdown has six header columns but five populated cells (mesh/time combined), and does not list the reverse figure hashes shown in their titles. Repair the manifest, link every image to its effective configuration/raw record, and avoid treating a displayed hash alone as complete provenance.

Continue through routine implementation/debugging without another geometry-policy stop. Return for an orchestration decision if no stable isolated-port transfer can be obtained within the declared continuation family, or if a consistent, validated reciprocal pair exposes a physical mismatch requiring changes inside the preserved coupling region. Report those findings with the failed sensitivity data. No broad efficiency optimization is part of this authorization.
