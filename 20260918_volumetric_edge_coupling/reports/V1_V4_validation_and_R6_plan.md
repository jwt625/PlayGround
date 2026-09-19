# Figure validation and next-step decision

Review date: 2026-09-18. Scope: the four `reports/figs/fig_*.png` images, cached paper Fig. 2, their plotting/geometry/port/mode implementations, current nominal configuration, and available result records. This is a visual, analytic and source-code review; no new FDTD runs or implementation changes were made.

**Decision: choose (c), reciprocal R6 first, with a short port-validation prerequisite. Prioritize Hz and retain one matched Ez control.** Keep 262 nm, 53.5°, the nominal wedge and source fixed for this diagnostic. A complete complex-beta solver is not a prerequisite for R6 if an effectively isolated receiving-guide port can be validated. The low-coupling result and the claimed loaded-mode phase match remain provisional. Do not start a broad geometry search or describe the existing scan as a validated sub-1% ceiling.

## 1. Gate decisions

| Gate | Verdict | Evidence and limitation |
|---|---|---|
| Geometry | **Partial pass** | Correct local Fig. 2 material ordering, cavity-side guide, wedge opening downstream, source axis y=11 µm, analytic normal thickness 262 nm. Impact gap **63.77 nm**. Actual sampled dielectric, source extent, monitors and inner PML boundaries are missing from the figure. |
| Field flow | **Qualitative pass only** | Down-left reflected lobe agrees qualitatively with vector reflection. Guide localization, reflected angle and absence of direct/radiative contamination are not established quantitatively. The white guide outline obscures the putative narrow jet. |
| Loaded-mode identity | **Not passed: evidence missing** | Exported loaded-mode E/H profiles and guide/substrate participation are absent. The supplied profile plot is isolated analytic slab data, and has two plotting defects. A real MPB band number or index crossing is insufficient identity evidence. |
| Flux/modal self-consistency | **Not passed: different apertures** | The custom line and native modal monitor differ in position, orientation, extent and sampled material. Their difference cannot be assigned solely to cladding radiation. Compare flux and modal decomposition on the identical native monitor first. |

Credit for resolved implementation issues: the current slab solver uses an undivided residual and rejects roots with large normalized residual; the current port code indexes wavevectors by band and frequency and divides by the corresponding frequency. Those changes address the two specific bugs in the preceding review. Their regression/convergence records are still separate acceptance evidence.

## 2. Geometry: what is actually established

For the current nominal configuration, s runs down the 54.74° sidewall and q points from silicon into oxide. The analytic gap is

`g(s) = 0.020 + (s - 9.0) tan(1.24°)` in micrometres.

| Quantity | Analytic value |
|---|---:|
| Beam-axis impact s | 11.02211 µm |
| Beam-axis impact (x,y) | (6.36293, 11.00000) µm |
| Gap at impact, measured along sidewall normal | **63.7694 nm** |
| Gap slope | 21.6455 nm/µm |
| Gap at upstream stack start, s=9 | 20 nm |
| Gap at nominal interaction end, s=25 | 366.33 nm |
| Gap at output reference, s=31 | 496.20 nm |
| Upstream strip start y | 12.65113 µm |

The guide polygon offsets its outer face along the guide normal by 262 nm, so the thickness convention is correct. The positive nominal gap rules out analytic guide/substrate overlap over the modeled strip. The source axis and wedge direction agree with the local coupling portion of paper Fig. 2. This is an infinite inclined-substrate reduction; the finite cavity floor and planar output bend are not modeled. The y=0 label is a reference, not a simulated floor.

The upstream strip edge is only 1.65 µm above the beam axis, versus w0=5 µm. Thus a substantial part of the footprint encounters the sidewall before the strip begins. This is a physical finite-overlap hypothesis, distinct from numerical source clipping. The current gap plot starts at s=9 and hides that part of the footprint. Extend its horizontal range to show the full beam and explicitly mark regions without a guide; do not extrapolate the gap into fictitious negative-gap material.

At resolution 25/µm, the nominal cell pitch is 40 nm: the impact gap spans 1.59 cells, the starting gap 0.50 cell, and the strip 6.55 cells. Subpixel geometry can represent smaller distances, but a single such mesh does not establish converged coupling or nanometre tolerances.

The geometry image shows an outer cell rectangle but not the PML's inner edge, full source span, or measurement apertures. Export a sampled dielectric zoom with analytic boundaries, the actual source line, both port apertures and shaded PML bands. The absorbing guide continuation should intentionally enter PML; the source, interaction region and measurement planes should remain outside it. Do not apply a blanket prohibition on guide/PML intersection.

## 3. Field and figure provenance

For horizontal incidence and sidewall normal n=(sin(alpha), cos(alpha)),

`k_ref = k_in - 2 (k_in dot n) n = (cos(2 alpha), -sin(2 alpha))`

gives **(-0.333478, -0.942758)**, or **-109.48°** from +x. The down-left reflected lobe is consistent with this direction. The schematic's vertical arrow is not a quantitative reflection prediction. A bare Si/oxide reference and a power-weighted reflected angular spectrum should establish numerical agreement, away from the interaction/interference region.

The Poynting figure also shows appreciable cavity-side radiation. However, its bright guide-aligned line is partly a white geometry overlay; it cannot independently prove a confined output channel. Add a near-guide zoom with thin boundaries or a second panel without outlines, signed guide-parallel flux, and a port field/profile comparison. A Poynting magnitude image alone cannot distinguish guided power from nearby radiation.

The figure reproduction command in `figures_index.md` uses `configs/nominal.yaml`, whose current source branch is **ez**. The image itself does not identify polarization. It therefore cannot be treated as Hz evidence without a saved effective configuration establishing its branch. This does not establish which historical configuration generated the PNG; that provenance is missing.

The figure script saves PNGs, not the underlying complex fields or an immutable run record, and it does not compute the modal number quoted beside the Poynting result. The available `results/` files do not contain a traceable new loaded-index table or the stated 12-case calibration. Preserve the reported numbers as reported exploratory results until the raw arrays, configuration and run identifiers are delivered together.

Two concrete defects in `fig_mode_profiles.png` need correction:

1. Core shading is vertical along field amplitude, although position is on the vertical axis. Shade y in [-0.131, +0.131] µm horizontally.
2. `slab_profile` concatenates an increasing top-cladding coordinate segment with a decreasing core segment. The plotting line connects the distant tail directly to the core, creating the visible spurious diagonals. Reorder coordinates and their field/material arrays consistently, or plot properly ordered contiguous segments.

Label the scalar profiles explicitly as normalized |Ez| and |Hz|, with air/core/oxide boundaries. These plots validate isolated fundamental shapes only; they do not replace actual loaded-mode profiles.

## 4. Why the new index table is not yet a mechanism confirmation

The supplied indices are candidates for a loading-dependent branch. They do not yet prove that the same receiving-guide resonance is tracked across positions. The real-mode computation includes substrate channels; finite transverse computational domains discretize that continuum. Meep's modal decomposition describes propagating modes of a translationally invariant cross section and unit-power modal coefficients; it is not automatically an outgoing-wave complex-beta resonance calculation. See [Meep mode-decomposition theory and normalization](https://meep.readthedocs.io/en/latest/Mode_Decomposition/) and [MPB's eigenproblem](https://mpb.readthedocs.io/en/latest/Introduction/).

In particular, `scripts/r5_loading.py` claims continuity tracking in its description but actually selects the index nearest the isolated-guide index independently at each gap. That is not field-overlap branch tracking. Export the profiles and vary the transverse domain before interpreting such a sequence. For a local parallel-layer approximation, record its coordinate frame and approximation to the actual nonparallel wedge.

Even if the quoted Hz indices belong to the correct resonance, a local crossing does **not** eliminate phase mismatch over the finite illuminated region. Using the reported values and the guide-projected target 2.070, the local detuning k0(neff-2.070) changes from about -0.085 rad/µm at s=11 to +0.507 rad/µm at s=16. Coupling depends on the coherent integral of excitation, accumulated detuning and subsequent leakage along the guide. Local phase matching, useful coupling strength, low escape loss and Gaussian wavefront matching are separate conditions.

Therefore replace “this confirms prism loading and phase matching is not the limiter” with: **“The exploratory real-mode indices suggest a possible loading-induced local crossing for Hz; branch identity, leakage and accumulated phase remain unvalidated.”** Hz remains the prioritized working hypothesis, not a confirmed translation of the paper's TE label.

## 5. Power comparison: the apertures are different

For the nominal geometry at s=31, in paper coordinates:

| Measurement | Actual aperture |
|---|---|
| Custom line (`extraction_line`) | Starts from the sidewall point (17.896, -5.313) µm, sampled at offsets 0.05 to 12 µm along the guide normal; line length 11.95 µm. Mostly samples oxide/guide/cavity-side material. |
| Native modal monitor (`add_oblique_mode_monitor`) | Vertical line at approximately x=18.408 µm, centered at y=-4.951 µm, spanning ±6 µm. Includes substantial substrate as well as the strip and cladding. |

The 2.3% net line flux and 0.5–0.8% selected-band forward power are therefore not an aperture-controlled comparison. They also compare net power with forward power. Use native signed flux on the **same mode monitor** and compare with the sum of identified forward-minus-backward channel powers. Retain individual forward and backward powers. A selected-band sum is not necessarily a complete modal basis; residual flux is not automatically absorption or numerical error.

If the custom oblique line remains a diagnostic, first validate it against native flux on the same physical aperture. Comparing different oblique/vertical sections additionally requires accounting for flux through their connecting boundaries; a cosine correction alone cannot remove radiative side leakage. Until identity passes, use “selected MPB-band power” and “net aperture flux,” not “guided coupling.”

## 6. Coding handoff: bounded next tasks and completion criteria

All numerical tolerances below are proposed review criteria, not paper claims. Keep these tasks small and preserve the original T01–T15 numbering.

### V0 — Freeze evidence and repair diagnostic figures

- [ ] Save the effective configuration, branch, geometry manifest, solver versions, mesh, runtime, smoothing policy and run ID for every new figure/table/scan point. Save raw complex E/H, fluxes and complex modal coefficients, not only squared magnitudes.
- [ ] Correct the isolated-profile plotting defects; export whole-cell and sampled-gap geometry with source, ports and PML boundaries.
- [ ] Export all 12 calibration cases as machine-readable records, including failures and both forward/backward coefficients. Do not silently associate them with the Poynting PNG.
- [ ] Include the complete footprint and upstream termination in the gap/field panels, and label Hz/Ez explicitly.

**Done when:** every plotted number is reproducible from a named immutable record, the profile plot has correct interfaces and no connecting artifacts, and source/interaction/measurement clearance is numerically reported. Mark prior narrative-only measurements as provisional until linked.

### V1 — Validate a receiving-guide port and common-aperture accounting

- [ ] Export dielectric and complex modal profiles at the current output, at the near-impact candidate, and at an effectively isolated receiving-guide reference. Report guide/substrate participation in a fixed finite observation window.
- [ ] Track candidate identity by field overlap and convergence toward the isolated fundamental, with transverse-domain changes independent of mesh refinement. Do not select by band number or proximity to a target index.
- [ ] Establish an effectively isolated, uniform guide launch/measurement port for R6. Any continuation used to reach it must preserve the physical interaction geometry and pass a continuation-length/placement sensitivity check. Record both the physical output reference and the numerical port; quantify any transfer loss between them.
- [ ] In a straight-guide reference, verify the intended incoming mode, propagation sign and power normalization. Target at least 99% identified launch-channel purity; quantify contamination of the much smaller coupling observable separately.
- [ ] On an identical native aperture save total signed flux, identified forward/backward guide power, selected other modes and residual flux. Reconcile the custom integration separately on the same aperture.

**Done when:** a physical receiving-guide channel is identifiable and stable against domain/port choices, launch normalization is checked, and the same-aperture pure-guide flux discrepancy is ≤1% at the selected mesh. The full loaded complex-beta R5 calculation may remain open; R6 must not launch an unidentified substrate band.

### R6a — Reciprocal mechanism diagnostic, 1550 nm

- [ ] Run nominal **Hz first**, launching the validated receiving-guide mode from large gap toward the small-gap interaction region. Keep thickness, angles, wedge, beam target and source-height convention fixed. Run one matched Ez control after the Hz setup checks pass.
- [ ] Measure the actual incoming guide power, guide reflection, radiation into silicon, radiation into air/other channels, and a closed-boundary energy budget without double counting internal collection planes.
- [ ] On a plane in homogeneous silicon, save outgoing complex E/H over an aperture converged for the emitted field. Separate propagation directions before computing overlap.
- [ ] Construct the reciprocal Gaussian target from the calibrated forward illumination at the same reference plane, including its phase curvature, waist, polarization, position and angle. Use the time-reversed E/H convention consistently; do not substitute an intensity overlap or a refitted Gaussian.
- [ ] Compute a power-normalized complex electromagnetic overlap, `eta_reverse = P_target_Gaussian / P_guide_in`. Record collection-plane outgoing power separately from total silicon radiation.
- [ ] Report `f_collection = P_collection / P_guide_in` and `M = P_target_Gaussian / P_collection`, so `eta_reverse = f_collection * M`. Plot emitted amplitude, unwrapped phase where amplitude is significant, angular spectrum and target residual. Collect the spatial emission distribution along the wedge.
- [ ] Run the corresponding forward Gaussian-to-guide case using the same port pair and geometry. Compare the single identified channel in both directions; do not compare a one-mode reverse result with an unqualified multi-band forward sum.

**Done when:** forward/reverse efficiencies agree within `max(1e-4 absolute, 5% of the larger efficiency)`, the global power-budget residual is ≤1% of incident power, and the target overlap is stable against collection aperture/position. An overlap above collected power or a reciprocity failure blocks physical interpretation.

### R6b — Numerical acceptance and decision

- [ ] Treat resolution 25/µm as exploratory. Use at least two finer meshes (e.g. 50 and 75/µm, then higher if needed) and demonstrate stabilization of both efficiency and emitted phase/shape.
- [ ] Check run duration, output continuation, PML clearance/thickness and port aperture. Separately quantify smoothing/subcell-placement sensitivity of the thin gap.
- [ ] Require the final numerical changes in eta to be ≤`max(1e-4 absolute, 5% relative)` for this low-efficiency diagnostic. Report the actual changes and any unmet original stricter production criteria. A 1-percentage-point global energy check alone cannot validate a 0.5% coupling signal.
- [ ] Bound source/port contamination of the target-channel efficiency at the 1e-4 absolute scale or better. Retain unresolved uncertainty explicitly rather than reporting a precise ceiling.

**Done when:** the forward/reverse pair, output identity, emitted-field interpretation and numerical uncertainty support the same physical conclusion. R6 completion does not automatically complete the paper reproduction or ±7 nm tolerance validation.

## 7. Decision after R6

| Validated finding | Next action |
|---|---|
| Forward and reciprocal eta both low; emitted field and power budget explain the loss | Choose (a): accept low coupling **for this declared reconstruction** and produce a compact spectrum and a few perturbations. Do not claim a family-wide ceiling or disprove the paper. |
| Appreciable power reaches the collection plane but complex Gaussian overlap is poor | Consider (b), limited to the measured mismatch: beam position/focus, emission angle/phase, or gap-dependent emission envelope. Name the hypothesis and predeclare its bounds. |
| Weak silicon collection, with energy retained in the guide, reflected or radiated elsewhere | Use spatial leakage and reflection to select a bounded interaction-length/gap-profile/termination diagnostic. Full R5 complex-beta/leakage modeling becomes useful here. |
| Large forward/reverse discrepancy or unstable mode identity | Repair source/port/normalization before any physical calibration. |

Reciprocity by itself is a consistency check, not independent proof of geometry fidelity; the emitted field and overlap decomposition provide the mechanism information. No geometry fitting is authorized solely to reach 88%. Keep nominal and alternative-family results distinct and retain the original reported thickness/angles.

**T02 digitization should proceed now.** It depends on the cached paper, not on accepting a nominal observable. Defer claims of reproduced Fig. 3/4/5 curves until the simulation gates pass; there is no reason to defer assembling and validating the target dataset.
