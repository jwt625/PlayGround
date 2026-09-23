# Implementation TODOs and completion criteria

Start here after reading [PAPER_REVIEW.md](PAPER_REVIEW.md) and [SIMULATION_SPEC.md](SIMULATION_SPEC.md). This file is the handoff to the coding agent. Implementation now exists; unchecked boxes record acceptance still to be evidenced, rather than implying that no implementation work has occurred.

**Current orchestration priority (2026-09-22):** execute [NEXT_SESSION.md](NEXT_SESSION.md): repair harness termination/calibration/persistence/field coverage → initialization checks → two coarse C0/C1 runs plus matched incident reference → field/flux diagnosis and actual-port identity. The pair has not yet run; res-8 smoke outputs are not physical acceptance. Full-domain 50/75 stays deferred; T02 continues independently. The [R6 port/channel/overlap contract](reports/R6_port_and_overlap_decision.md) remains in force.

**Latest R5/R6 allocation:** Hz first, with one matched Ez reciprocal control. Current code addresses the previous root-residual and band/frequency indexing defects; their regression evidence and loaded-mode localization remain to be validated. Do not require completion of the full outgoing-wave R5 solver before R6 if an effectively isolated output port is validated. Hz loading-induced phase matching and the paper's TE mapping remain hypotheses. See the [earlier branch review](reports/R5_R6_branch_priority.md) for context.

**Scope:** reproduce the paper's single-channel electromagnetic simulations, with Meep as the default backend. Complete a defensible 2D result first. Full paper coverage additionally requires 3D width and transverse-offset runs. Exact agreement is conditional on recovering or declaring missing inputs. A solver that faithfully disagrees with the paper is preferable to an undocumented fitted curve.

**Roles:** coding agent implements, tests and runs; orchestration/visual reviewer inspects geometry and fields, checks evidence labels and comparison claims, and tracks unresolved questions. Review artifacts are required; a human permission stop is not required for ordinary local implementation. If a needed assumption cannot be established, keep independent work moving and mark the dependent result conditional.

## T01 — Establish environment and run-record contract

Dependencies: none.

- [ ] Install Meep/MPB in an isolated environment using the official supported package route. Record platform, package/build versions, dependency lock, numerical precision, and material data provenance.
- [ ] Verify a minimal 2D run and access to the mode solver; then test Gaussian source classes actually present in the installed build.
- [ ] Define a validated parameter schema with units, dimensionality, named geometry/source hypotheses, and reported/assumed/fitted labels.
- [ ] Define immutable run directories with effective configuration, input hash, solver version, runtime, peak RSS, completion status, raw flux/mode data, and warnings.
- [ ] Implement a dry-run resource estimate, bounded worker count, failure reporting, and run-level resume. Do not build a web app or generic workflow framework.

Deliverables: environment lock; getting-started command; configuration schema; small smoke-run record; measured initialization/run memory and throughput.

**Done when:** a clean shell can reproduce the smoke run; changing a physical or numerical parameter changes the run identity; incomplete/failed outputs cannot be mistaken for valid results; no global Python environment changes are required.

## T02 — Create the paper target dataset

Dependencies: cached PDF/figures already present. Can proceed independently of solver setup.

- [ ] Encode Tables 1–3 as a reference dataset, preserving reported precision and page/table provenance.
- [ ] Digitize all three Fig. 3 curves; all six Fig. 4 curves; the Fig. 5(a–c) curves; and five reflection points from Fig. 5(d).
- [ ] Calibrate pixel-to-axis transforms separately for each panel; distinguish solid/dashed curves and exclude legends/insets/occluded segments.
- [ ] Save original pixel coordinates, transformed values, estimated readout errors and point visibility flags. Do not fill hidden portions as measured data.
- [ ] Cross-check a subset of manually read points against extraction. Overlay recovered samples on the original embedded image for visual review.
- [ ] Store table claims separately from curve estimates; log any conflicts without altering either source.

Deliverables: target CSV/JSON tables; digitization metadata; image overlays; a discrepancy ledger.

**Done when:** every target is traceable to a page/panel or a named proposed criterion; overlays track visible curves within line-width/readout uncertainty; table values retain their original meanings. Digitized visual estimates are never described as author raw data.

## T03 — Resolve the physical model before the full coupler

Dependencies: T01.

- [ ] Implement explicit Ez and Hz polarization configurations and map each to the paper coordinate frame.
- [ ] Solve the nominal 262 nm receiving-guide modes for air/oxide and oxide/oxide surroundings, plus relevant substrate-loaded cross sections.
- [ ] Calculate nominal TIR angle, critical angle, evanescent decay and approximate incident tangential momentum using the actual chosen indices.
- [ ] Document the main polarization/cladding hypothesis and the reason for it; preserve alternate diagnostics if the paper is ambiguous.
- [ ] Establish a provisional source-focus convention and gap/geometry family from the missing-input ledger. Do not pretend the gap was supplied by the paper.

Deliverables: mode-index table versus thickness/wavelength; mode-profile plots; concise model-decision note with unresolved items.

**Done when:** the launch medium gives physical TIR, mode identities are explicit, plausible phase matching is assessed, and any remaining ambiguity is represented in configuration rather than buried in code. A mismatch is investigated rather than cured by silently replacing 262 nm or 53.5°.

## T04 — Implement geometry and visual inspection artifacts

Dependencies: T01, T03.

- [ ] Build substrate, oxide wedge and finite-thickness guide from the common geometry contract.
- [ ] Support the source-height reference, guide-angle pivot, physical normal thickness, finite interaction endpoints and output reference plane.
- [ ] Add preflight checks for negative gap, overlapping solids, clipped source, PML overlap and monitor placement.
- [ ] Export whole-device material maps, a magnified wedge/guide view, g(s), source axis and waist, monitors, boundary/PML extents, and coordinate axes in µm.
- [ ] Show analytic boundaries and sampled/subpixel geometry together; include examples at nominal and perturbed thickness/angle.
- [ ] Export a geometry manifest sufficient to reproduce all vertices/material regions without reading plotting code.

Deliverables: `geometry_full`, `geometry_gap_zoom`, and `gap_profile` images; geometry manifest; validation failures for deliberately invalid inputs.

**Done when:** visual review confirms the Fig. 2 topology, beam inside silicon, correct wedge sign, 262 nm measured normally, source y = 11 µm from the base, and a pre-bend output. Visual similarity alone is insufficient: numerical coordinates and separation checks must agree.

## T05 — Validate illumination, TIR and absorbing boundaries

Dependencies: T01, T03; simplified geometry from T04 as needed.

- [ ] Run a homogeneous-silicon source reference at 1500, 1550 and 1600 nm; measure waist, center, propagation angle, wavefront and incident power.
- [ ] Check tilted source cases and the effect of temporal bandwidth on launch direction/waist.
- [ ] Validate planar-interface Fresnel response below critical incidence for both polarizations and TIR above critical incidence.
- [ ] For a monochromatic plane-wave or properly angular-spectrum-accounted reference, compare oxide amplitude decay with analytic κ. Do not apply a single-plane-wave decay formula blindly to a finite beam.
- [ ] Measure PML return artifacts and repeat with increased thickness/clearance. Account for the material interface intersecting the absorbing region.

Deliverables: source-calibration plots/data; Fresnel comparison; evanescent-decay fit; PML convergence record.

**Done when:** flux Fresnel/TIR errors are ≤1 percentage point, decay-length error is ≤5% on a converged reference, and source/boundary uncertainties change representative coupling by <0.02 dB. These are proposed numerical criteria, not paper targets.

## T06 — Implement trustworthy modal transmission and reflection

Dependencies: T01, T03, T05.

- [ ] Run a straight receiving-guide test, then the same physical guide at the target inclination or in a rotated whole-system frame.
- [ ] Validate forward/backward mode coefficients, total signed flux and monitor-position invariance; identify selected mode order/polarization explicitly.
- [ ] Validate material treatment in mode solving at sampled wavelengths. Use the frozen-permittivity narrowband route if the broadband dispersive projection cannot be trusted.
- [ ] Implement incident reference runs and complex-field subtraction where needed for reflection.
- [ ] Build non-overlapping power-budget channels and reject passivity violations above numerical tolerance.
- [ ] Test extension/termination length, mode aperture, eigenmode resolution and PML sensitivity independently.

Deliverables: uniform-guide mode/flux benchmark; incident/reflected separation demonstration; power-budget report; monitor schematic.

**Done when:** mode/flux agreement and energy balance meet SIMULATION_SPEC.md; no hidden normalization makes η exceed one; tilting/rotating a reference guide does not create artificial coupling loss. Only then use modal power to score the coupler.

## T07 — Obtain and diagnose the nominal 2D coupler

Dependencies: T04–T06.

- [ ] Run 1550 nm at published twg, θwg and y for the declared reconstruction.
- [ ] Save real-field snapshots, complex field/intensity maps and time-averaged Poynting vectors showing the incident beam, evanescent region, receiving-guide power and reflected beam.
- [ ] Run a no-receiving-guide control to confirm TIR/reflection and a much-larger-gap control to show coupling decays.
- [ ] Compare the wedged gap against one clearly defined constant-gap control; this is a mechanism diagnostic rather than a figure target.
- [ ] If nominal coupling is poor, diagnose polarization, n_eff, gap, focus, clipping and modal extraction in that order before launching a broad optimizer.
- [ ] If missing geometry requires calibration, do a coarse bounded search in gap/intercept and interaction length; log candidates and preserve a pre-fit baseline.

Deliverables: nominal run record; power budget; diagnostic control comparison; field review packet; reconstruction/calibration history.

**Done when:** validated fields demonstrate transfer into the intended receiving guide, the controls behave physically, and the numerical results can be explained without source/monitor contamination. Reaching 88% is a later comparison criterion, not a substitute for these checks.

## T08 — Certify numerical convergence

Dependencies: T07; apply before production tolerance sweeps.

- [ ] Evaluate the proposed resolution ladder and refine further if needed.
- [ ] Independently vary physical domain, PML thickness, source aperture, output extraction plane, termination length and run duration.
- [ ] Repeat at least one thickness perturbation and one tilt perturbation to ensure convergence is not specific to the nominal case.
- [ ] Test a half-cell geometry translation and inspect sensitivity to subpixel treatment.
- [ ] Freeze the accepted numerical settings and their measured uncertainty; record resource estimates for remaining sweeps.

Deliverables: convergence CSV and plots; explicit error budget; selected production configuration; benchmark-derived runtime/RAM estimate.

**Done when:** every applicable numerical threshold in SIMULATION_SPEC.md passes, or the run is explicitly marked nonconverged with the failing observable. No headline reproduction claim from a nonconverged run.

## T09 — Reproduce spectrum, peak and bandwidth

Dependencies: T02, T08.

- [ ] Obtain 1500–1600 nm nominal transmission with a calibrated source and adequate spectral signal; widen the range if a needed crossing is outside it.
- [ ] Validate broadband results with independent narrowband points at peak and both band edges, and check the material-dispersion alternative.
- [ ] Extract actual maximum, η at exactly 1550 nm, relative 1-dB edges/bandwidth and absolute −1-dB edges/bandwidth.
- [ ] Overlay simulation and digitized Fig. 3 2D trace without rescaling amplitude or shifting wavelength to improve appearance.
- [ ] Compare Table 1 and Table 2 separately from the curve; report deviations with numerical/digitization uncertainty.

Deliverables: spectrum CSV; Fig. 3 2D overlay; peak/bandwidth table; source/material sensitivity comparison.

**Done when:** numerical gates pass and the comparison has a truthful pass/conditional/mismatch status using the rubric below. If disagreement persists, provide a short ranked diagnosis and missing inputs rather than arbitrary fitting.

## T10 — Reproduce fabrication spectra and 1550 nm tolerances

Dependencies: T09; nominal physical and numerical settings frozen.

- [ ] Run exact Fig. 4(a) angle spectra: 53.25°, 53.50°, 53.75°.
- [ ] Run exact Fig. 4(b) thickness spectra: 257, 262, 267 nm.
- [ ] At fixed 1550 nm sweep θwg initially from 52.8° to 54.2° in 0.1° steps; bracket each 1-dB crossing and refine to ≤0.025°.
- [ ] At fixed 1550 nm sweep twg initially from 247 to 277 nm in 2 nm steps, explicitly including 262 nm; refine crossings to ≤0.5 nm.
- [ ] Extend ranges if crossings are not bracketed. Keep gap pivot/thickness convention fixed and all other parameters unchanged.
- [ ] Measure local optimum only as a separate diagnostic. Do not replace the paper's nominal geometry in tolerance plots.

Deliverables: six spectra with overlays; fixed-wavelength sensitivity curves; asymmetric left/right crossing estimates and uncertainty.

**Done when:** spectral-shift directions agree or the mismatch is explained, both crossings are resolved, and Table 3 comparisons use fixed-wavelength relative loss. Do not compare only the peak of each perturbed spectrum.

## T11 — Reproduce in-plane alignment and reflection

Dependencies: T09; source convention frozen.

- [ ] Run exact Fig. 5(a) spectra at y = 10, 11, 12 µm.
- [ ] Run exact Fig. 5(c) spectra at tilt −1°, 0°, +1°.
- [ ] Sweep y initially from 7 to 15 µm in 0.5 µm steps; refine 1-dB crossings to ≤0.1 µm.
- [ ] Sweep tilt initially from −2° to +2° in 0.2° steps; refine crossings to ≤0.05° and include all integer-degree reflection points.
- [ ] At −2, −1, 0, +1, +2° report absolute reflected power and complete energy budget at 1550 nm. Also retain spectra if available because the paper's reflection wavelength is unspecified.
- [ ] Compare reflection-minimum location with transmission maximum. Plot a separately labeled peak-normalized reflection shape only as a tested interpretation of the published normalization.
- [ ] Run a small fixed-impact-point tilt diagnostic to quantify sensitivity to unknown pivot/source distance.

Deliverables: Fig. 5(a,c) overlays; tolerance curves; Fig. 5(d) raw and optional shape-normalized comparison; energy accounting; pivot-sensitivity note.

**Done when:** vertical and tilt crossings are resolved without refocusing/re-optimizing other parameters; reflected power is physically normalized; any incompatibility with the paper's reflection ordinate remains visible. Exact Fig. 5(d) amplitude reproduction stays unresolved unless its denominator and wavelength are established.

## T12 — 2D release and full 3D feasibility decision

Dependencies: T09–T11.

- [ ] Publish a 2D results report and a per-figure coverage table before spending effort on 3D.
- [ ] Estimate actual 3D memory/time from accepted geometry/resolution, DFT storage and measured 2D/3D initialization overhead.
- [ ] Benchmark a small finite-width guide/source problem with the proposed 3D backend; verify vector polarization, modal extraction, and precision.
- [ ] Select a feasible machine/backend/domain strategy; specify symmetry applicability and how centered/offset runs differ.

Deliverables: reproducible 2D report; resource feasibility note; declared next-stage status.

**Done when:** 2D work is independently usable and the 3D launch fits measured resources. If infrastructure is unavailable, mark T13–T14 pending; do not call the entire paper reproduced.

## T13 — Reproduce the finite-width 3D comparison

Dependencies: T12 and adequate compute.

- [ ] Extrude the identical in-plane geometry, using W = 10 and 15 µm and a properly normalized 3D Gaussian source.
- [ ] Use open/PML transverse boundaries and convergence-tested cladding/substrate extent. Document any valid symmetry reduction.
- [ ] Solve and track lateral guided modes; report fundamental, sum of guided modes, and total flux separately.
- [ ] Converge mode count until adding modes changes the summed guided power by <0.5 percentage points; distinguish bound, substrate and radiation channels.
- [ ] Compare nominal and edge wavelengths first, then full spectra after mesh/domain/mode convergence.
- [ ] Overlay Fig. 3 and assess how the 15 µm result approaches the 2D observable. A wider optional control can test the trend if feasible.

Deliverables: 3D geometry views and field slices; mode-content table; Fig. 3 three-curve overlay; 3D convergence/resource report.

**Done when:** numerical checks pass and width trends are compared using explicitly matched power definitions. A match of summed power with a mismatch of fundamental-mode coupling must be reported as such.

## T14 — Reproduce transverse alignment

Dependencies: T13.

- [ ] At the declared width, run z offsets 0, ±5 µm; provisional primary width is 15 µm because Fig. 5(b)'s width is unspecified.
- [ ] Include intermediate offsets 1–4 µm at 1550 nm to expose aperture/modal changes; repeat endpoint checks at 10 µm width as an ambiguity study.
- [ ] Rebuild/recenter the incident reference as required so beam clipping cannot masquerade as tolerance.
- [ ] Check ±offset symmetry with the correct polarization/vector reflection transformation and an unsymmetrized numerical spot check.
- [ ] Plot mode-resolved powers, total guided power, reflection and remaining radiation versus offset.

Deliverables: Fig. 5(b) spectra/overlay; offset scan; symmetry and width-sensitivity tests; transverse field slices.

**Done when:** the small reported penalty through ±5 µm is tested quantitatively under declared assumptions. Report the measured penalty, not a claimed ±5 µm 1-dB limit; the paper does not give that limit.

## T15 — Final reproducibility and visual-validation package

Dependencies: T12 for a 2D delivery; T13–T14 for full simulation coverage.

- [ ] Provide one documented workflow each for reference tests, nominal 2D, plotted sweeps, convergence checks, digitization overlays and supported 3D cases.
- [ ] Generate figures/tables from saved raw results with no rerunning FDTD required for report rendering.
- [ ] Provide a manifest connecting every plot to configuration hash, raw data, solver version and reference target.
- [ ] Include geometry, field, flux and modal review artifacts listed below; show fixed color scales and units where comparisons depend on amplitude.
- [ ] Separate reproduced, conditional, mismatched, not attempted and out-of-scope claims; list remaining author questions and highest-impact unknowns.
- [ ] Summarize any model calibration and evaluate the held-out fabrication/alignment curves without further tuning.

**Done when:** another engineer can regenerate the report from stored outputs, rerun a minimal reference and nominal case, identify all assumptions, and tell which paper claims have actually been tested. A table of honest mismatches is an acceptable scientific outcome; claiming a numerical target without evidence is not.

## Proposed quantitative paper-agreement rubric

These are reproduction goals chosen for this project, not uncertainties supplied by the authors. Apply only after numerical acceptance. Where digitization uncertainty is greater, show it and explain any adjusted comparison threshold explicitly.

| Quantity | Paper target | Proposed agreement threshold |
|---|---|---|
| Nominal η at/near 1550 nm | 0.88 / −0.56 dB | ±0.02 absolute efficiency (two percentage points) |
| Peak wavelength | 1550 nm | ±5 nm |
| Relative 1-dB spectral edges | 1507.23 / 1593.68 nm | each ±5 nm |
| Relative 1-dB bandwidth | 86.45 nm | ±10 nm; also pass edge comparison |
| Guide-angle left/right tolerances | approximately ±0.4° | each within 0.1° of reported magnitude |
| Thickness left/right tolerances | approximately ±7 nm | each within 2 nm |
| Source-y left/right tolerances | approximately ±2 µm | each within 0.5 µm |
| Source-tilt left/right tolerances | approximately ±0.8° | each within 0.2° |
| Fig. 3/4/5 spectral shapes | digitized visible curves | target RMSE ≤0.15 dB over visible, reliable samples; no fitted dB/λ offsets |
| Transverse ±5 µm degradation | visually ~0.1–0.15 dB near peak; no tabulated number | compare digitized difference with uncertainty; provisional ≤0.25 dB penalty as a trend check only |
| Reflection minimum | near 0° | minimum within ±0.2° of nominal and near maximum transmission; amplitude conditional on normalization |

Record tolerance crossings separately on either side of the nominal point. If a crossing is beyond the scan, report a bound and extend the scan where feasible. Compute relative loss against the same fixed nominal geometry at 1550 nm, rather than against each perturbed case's spectral peak or a refitted optimum.

Numerical success and paper agreement are independent columns in the results report. Full paper reproduction cannot be declared solely from matching the nominal peak: the curves, sensitivity directions, 3D observables, and measurement definitions also matter. If a reported table value conflicts with digitized curves, preserve both and flag the comparison as unresolved.

## Visual review gates

| Gate | Reviewer inspects | Reject / investigate if |
|---|---|---|
| V1 after T04 | material map, local thickness/gap, axes, beam and monitor overlays | source in wrong medium; gap/normal-thickness error; guide on wrong side; hidden clipping |
| V2 after T05–T07 | source/TIR controls, complex field, intensity and Poynting maps | wrong reflection direction; fields terminate at mesh artifacts; apparent waveguide power is direct beam flux |
| V3 after T08 | convergence and half-cell-shift plots | optimum tracks staircase jumps; PML/runtime changes alter efficiency |
| V4 after T09–T11 | source-curve overlays and raw powers | curves independently normalized to one; spectra shifted; tolerance re-optimized; reflection violates shared normalization |
| V5 after T13–T14 | 3D slices, mode profiles, centered/offset beams, mode-power distribution | clipped Gaussian; source symmetry forced at nonzero offset; total aperture power called fundamental-mode coupling |

Use actual simulation artifacts. Do not substitute illustrative/generated images for numerical field evidence. The review packet should include source filenames and numeric color bars so screenshots are auditable.

## Suggested implementation layout and interfaces

This is a proposed layout, not a request to build an abstraction framework. Keep modules small and specific.

| Component | Responsibility / interface |
|---|---|
| `configs/` | explicit nominal, hypothesis, convergence and sweep definitions with units/provenance |
| `src/geometry` | validated parameters → material geometry, coordinate transform and manifest |
| `src/materials` | dataset/model → n(λ), ε(λ), solver medium and fit diagnostics |
| `src/sources` | calibrated beam specification → source plus matching reference configuration |
| `src/ports` | monitor geometry, mode identities, flux and directional modal extraction |
| `src/run` | validated run → raw outputs plus complete status/provenance |
| `src/sweeps` | parameter list → bounded/resumable run records; no silent auto-optimization |
| `src/analysis` | raw powers → efficiencies, bandwidth/tolerance crossings, uncertainties |
| `src/report` | saved outputs + paper targets → overlays and result tables |
| `tests/` | meaningful analytic/source/port/energy and geometry-regression tests |
| `data/paper_targets/` | tables, digitized points and extraction metadata |
| `results/` | immutable configs, data, logs and artifact manifests |
| `reports/` | 2D/full reports, review packets, limitations and claim coverage |

Minimum spectral data fields: run ID, wavelength, incident power, mode ID, forward/backward modal power, total port flux, reflected power, other outgoing channels, energy residual, numerical status. Preserve linear quantities; derive dB in analysis. Output explicit NA/status values for unresolved modes or failed frequencies instead of zeros that look physical.

## Optional extensions after the paper reproduction

- [ ] Independent homebuilt 2D cross-check, following SIMULATION_SPEC.md's prerequisite tests.
- [ ] Reciprocal waveguide-to-Gaussian coupling check with identical normalized ports; compare overlap into the launched Gaussian channel, not all radiated power.
- [ ] Fiber/silicon entrance transmission, AR treatment and external alignment model.
- [ ] Bend/transition into the planar layer and wide-guide to single-mode routing conversion.
- [ ] Manufacturable spacer approximations and fabrication-error correlation.
- [ ] Multicore geometry, crosstalk and channel-density study.

These are additional studies, not missing curves from the present paper. Keep them out of the initial critical path.

## Pasteable coding-agent brief

Implement the reproduction specified in README.md, PAPER_REVIEW.md, SIMULATION_SPEC.md and TODO.md. Use Meep by default. Start with T01–T08 and produce a validated nominal 2D case before launching broad sweeps. Keep the paper's explicit parameters separate from missing geometry/material/source assumptions. Do not normalize or tune curves merely to obtain 88%. Validate power, polarization, TIR, gap geometry, source and output-mode extraction before comparing results. Deliver the 2D spectra and fixed-1550-nm tolerances, then assess resources before 3D. Every completed TODO needs linked artifacts and a pass/conditional/mismatch status. Submit real geometry/field/convergence plots for orchestration and visual review. Preserve the cached references and planning documents; revise assumptions transparently when new evidence appears.
