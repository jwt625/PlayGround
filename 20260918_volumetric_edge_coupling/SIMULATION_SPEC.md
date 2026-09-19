# Solver decision and numerical specification

This is an implementation specification, not code. All numerical settings below are proposed unless identified as paper values. See [PAPER_REVIEW.md](PAPER_REVIEW.md) for evidence and unresolved inputs.

## 1. Solver decision

| Option | Relevant capability | Work/risk for this task | Decision |
|---|---|---|---|
| **Meep + MPB through Python** | 2D/3D FDTD, geometry primitives, PML, Gaussian illumination, modal decomposition, CPU/MPI | Must validate tilted geometry, source convention, polarization, dispersive mode handling | **Primary implementation** |
| **FDTDX** | JAX FDTD, accelerator support, mode-source/overlap tooling | Benchmark thin oblique layers, source/dispersion support and numerical convergence; GPU access is not established here | Candidate later 3D acceleration backend |
| **flaport/fdtd** | Compact Python FDTD with NumPy/PyTorch backends and PML | Additional photonic source, subpixel geometry, modal and spectral metrology work | Not preferred for quantitative reproduction |
| **Homebuilt 2D Yee solver** | Complete control and transparent diagnostics | Must implement and validate CPML, injection, oblique interfaces, mode projection, dispersion if used | Optional independent verification track |

These are engineering judgments based on official project documentation, not measured solver performance. Meep's [repository](https://github.com/NanoComp/meep) documents dimensionality and MPI support. Its [installation guide](https://meep.readthedocs.io/en/latest/Installation/) lists conda-forge `pymeep` packages for Linux/macOS including ARM; use an isolated environment and pin the resolved build. No solver has been installed as part of planning.

Meep's [oblique-waveguide tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/) is directly relevant to output-port validation and warns about oblique guides entering PML. Its [subpixel documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/) supports the choice to use geometric objects with dielectric averaging; thin tilted interfaces still require convergence checks. [Mode decomposition](https://meep.readthedocs.io/en/latest/Mode_Decomposition/) provides forward/backward modal powers and notes MPB's material limitations.

[FDTDX's repository](https://github.com/ymahlau/fdtdx) documents JAX acceleration and an MIT license; its [mode tutorial](https://fdtdx.readthedocs.io/en/latest/notebooks/components/02_mode_source_detector.html) demonstrates mode overlap. [flaport/fdtd](https://github.com/flaport/fdtd) documents NumPy/PyTorch backends and PML. Neither is rejected as a Maxwell solver; Meep minimizes new numerical infrastructure for this particular reproduction. Meep uses GPL-2.0-or-later; the project's [license clarification](https://github.com/NanoComp/meep/blob/master/doc/docs/License_and_Copyright.md) distinguishes user control scripts from code linked into the solver.

## 2. Geometry contract

Use µm as the internal length unit; vacuum wavelength and f = 1/λ in Meep units. Display thickness/gap in nm and angles in degrees. Require units in external configuration names or schema. Convert once at the boundary.

Define the **paper coordinate frame** independently of any computational rotation:

- x: nominal incident-beam direction; y: height above cavity base; z: finite width/out-of-plane direction.
- Nominal source beam axis passes through y = 11 µm; z = 0 in the centered 3D model.
- α: positive sidewall inclination magnitude, approximately 54.74°; choose the downward-right sidewall shown in Fig. 2.
- For a sidewall-base point r_b, let u = (cos α, −sin α) run down the sidewall and n = (sin α, cos α) point from substrate toward oxide.
- Sidewall coordinates: r = r_b + s u + q n. The substrate boundary is q = 0 on the finite sidewall segment.
- Parameterize the substrate-facing guide boundary by q = g(s). For a straight guide at θwg, use g(s) = g_ref + (s − s_ref) tan(α − θwg). This convention makes the gap grow along downward-right guided propagation at the nominal angles, and shrink in the reverse-propagation picture.
- Build the other guide face by an offset **normal to the guide** of twg. Do not offset its global y coordinate by twg. Equivalently, at fixed s its q separation is twg/cos(α − θwg).

The named `g_ref` and reference position `s_ref` are mandatory. An angle pair without an intercept does not define a device. Physical source coordinates must be anchored to the same cavity-base origin even if the solver origin is at the cell center.

Required geometry configuration includes cavity depth/base width, finite sidewall endpoints, waveguide endpoints, oxide continuation, upper-cladding material, source plane/focus, and output extension. Do not extract absolute dimensions from schematic pixels. Set unknown values as documented assumptions, not hidden constants.

A provisional search family may use an interaction span of 10–40 µm and a minimum active-region gap of 0.02–0.50 µm, with actual slope fixed by the specified angles. These are broad hypothesis bounds suggested by the beam size and evanescent decay scale, not author dimensions. Parameterize by minimum gap if a centered `g_ref` would otherwise make the strip intersect the substrate. Reject negative gaps or overlapping solids before simulation. Test interaction/domain length independently so a too-small search box does not select the optimum.

Changing θwg must name the pivot/constraint. Start with a fixed reference gap at a declared location, then test a cavity-base-anchored interpretation if needed. Thickness sweeps hold the substrate-facing guide boundary fixed unless the author geometry says otherwise. For tolerances, freeze these choices across the entire sweep.

Use a single geometry description for 2D and 3D. The 3D version introduces finite guide width W along z and a two-transverse-coordinate Gaussian beam. Specify whether the oxide ridge has width W or extends beyond it. Extend substrate and background adequately in z; do not truncate them at the guide edges unless deliberately studying that hypothesis.

## 3. Source and polarization contract

The source medium must support the silicon-to-oxide TIR described in the paper. Baseline: calibrated Gaussian illumination in homogeneous silicon upstream of the sidewall. Include the fiber/silicon entrance only in a separately labeled extension.

Record the field components, not just a TE/TM string:

| 2D branch | Nonzero components | Incidence-plane interpretation |
|---|---|---|
| Ez branch | Ez, Hx, Hy | E perpendicular to the x–y incidence plane; s polarization |
| Hz branch | Hz, Ex, Ey | E within the x–y incidence plane; p polarization |

The paper states TE but does not map that label to these components. Screen both through slab-mode phase matching and a coarse coupler run; select a documented hypothesis and retain the alternative's diagnostics. Do not relabel a branch merely because its transmission is higher. In 3D explicitly specify the vector polarization transverse to the launch k-vector.

Use w0 = 5 µm with the declared conventional waist definition E ∝ exp(−r²/w0²), hence I ∝ exp(−2r²/w0²). This convention is an assumption to verify. Distinguish the spatial beam envelope from the temporal Gaussian pulse. The [Meep source API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussianbeam3dsource) distinguishes true 2D and 3D beam formulations and notes center-frequency accuracy. Check class availability in the pinned version. Validate measured waist and propagation against the selected formulation.

Source tests must measure center, waist, integrated power, propagation direction, and wavefront at two planes in homogeneous silicon. A broadband source needs narrowband spot checks at 1500, 1550, and 1600 nm, including tilted illumination: a fixed spatial phase profile can produce frequency-dependent launch angles. Do not assume normalized spectra cancel that error.

Primary tilt protocol: rotate the propagation direction while holding source center/focus convention fixed, and record the resulting impact point. A second diagnostic rotates about the nominal sidewall impact point. Their difference quantifies the missing pivot information. Recompute incident reference power for each source configuration, or establish reuse equivalence explicitly.

## 4. Materials and mode checks

Debug first with lossless, nondispersive nSi = 3.48 and nOx = 1.444; these are provisional values, not extracted paper inputs. Use n = 1 for air only in the explicitly air-clad hypothesis. Log n² and material IDs on plots.

Before interpreting spectral agreement, select documented Si/SiO2 optical data over 1500–1600 nm, freeze dataset/version, and record any FDTD fit error. Keep the constant-index run as a sensitivity comparison. Never fit material index solely to move a peak to 1550 nm without labeling the calibration.

Compute isolated receiving-guide modes for both polarization branches and each cladding hypothesis at nominal thickness; then inspect substrate-loaded cross sections where relevant. Report effective index, field profiles, power normalization, confinement, and mode count. Compare tangential momentum to the diagnostic in PAPER_REVIEW.md. A waveguide beside a high-index substrate may be leaky; do not classify every eigenvector of a periodic supercell as a bound output mode.

MPB has restrictions on dispersion/loss. Check the pinned Meep/MPB implementation's actual treatment at each monitor frequency; do not assume the material library fit is automatically represented correctly in modal projection. A robust fallback is narrowband simulations with the real material permittivity frozen at each sampled wavelength for both FDTD and mode solving. If material absorption is included, quantify it and use a consistent modal method. Cross-check power projection against a uniform-waveguide reference.

## 5. Monitors and power definitions

Let Pinc(λ) be the forward incident power from a separate homogeneous-source reference with the same dimensionality, source settings, mesh, and aperture. All powers in a 2D calculation are per invariant length; dimensionless efficiencies can be compared to 3D, raw watts cannot.

Required outputs:

- ηfund = Pforward,fundamental/Pinc.
- ηguided = sum of forward powers in identified guided modes/Pinc.
- Tflux = signed net total flux through the output aperture/Pinc.
- Rabs = independently measured reflected power/Pinc.
- Radiation into other channels, backward guided power, and material absorption if present.
- TdB = 10 log10(η), and positive insertion loss IL = −10 log10(η); label signs consistently.

Use the inclined output before the physical bend. Place monitors where an appropriate output-mode basis exists and where direct incident/reflected beams do not contaminate the aperture. Move the monitor along the guide to test invariance. If substrate leakage continues or the cross section varies appreciably, report position dependence and resolve the extraction-plane/model ambiguity rather than projecting onto an arbitrary isolated guide.

A port aligned with the computational axes is often simpler: one may rigidly rotate the entire geometry, source, and coordinates so that the inclined guide becomes horizontal. This is a coordinate change, not a physical bend. Preserve paper-coordinate parameters and prove rotation invariance on a reference problem. Alternatively use the documented oblique-port method, validated first on an isolated tilted guide. Never introduce an unreported bend just to make mode decomposition easier.

Validate the output extension and absorbing termination separately. Extending a uniform isolated guide is acceptable only after demonstrating it does not change the coupling region's fields or extraction. Moving/removing the physical bend is a model simplification requiring a sensitivity check if the bend could back-reflect.

For reflection, monitor the specular beam inside silicon over enough aperture as well as total other outgoing power. Use incident-field subtraction or directional separation if incident and reflected fields share a monitor. Do not subtract scalar powers where complex-field interference is present. Store the reflection aperture and reference run. A separate normalized shape Rabs(φ)/maxφ Rabs(φ) may be plotted as a **hypothesized** interpretation of Fig. 5(d), without replacing raw reflectance.

Check an enclosing flux budget around the passive scattering region, excluding the impressed source. Partition boundaries/ports without counting power twice. In a lossless calculation, total outgoing power must equal incident power within numerical error. The modal total must not exceed the appropriate forward flux; net flux differs if backward power is appreciable.

## 6. Mesh, domain and time convergence

Suggested 2D ladder: 25, 50, 75, 100 pixels/µm (40, 20, 13.3, 10 nm cells). Start screening at the first two levels; certify only from demonstrated convergence. If necessary extend to 150 pixels/µm. A 7 nm thickness tolerance cannot be resolved credibly by snapping a 262 nm layer to a coarse staircase.

Use analytic polygon/block geometry and verified subpixel treatment. Compare half-cell translated geometries at fixed resolution. Export both intended boundaries and actual sampled material maps; inspect the entire wedge and the thinnest layer. Dispersive smoothing behavior may differ from nondispersive behavior, so repeat the relevant check for the final material model.

Start with ~1.5–2 µm PML and test greater thickness; these are initial settings only. Tilted material interfaces entering PML deserve explicit reflection tests. If they fail, use a validated coordinate rotation or longer absorbing termination. Increase source/monitor clearance and physical domain independently of PML thickness. Keep source tails away from PML and unintended interfaces; begin with at least ~3 beam radii of aperture where the actual geometry permits it, then converge aperture.

Use a stable Courant factor appropriate to dimensionality and materials, initially 0.5 in Meep, and test a reduction at a representative point if errors remain. Run until field/energy decay and monitor values stabilize; start with decay around 1e−7 relative to peak, then extend runtime to check. A single probe at a field node is not a stopping test. Save time history and reject runs terminated solely by a wall-clock cap.

DFT sample spacing of 0.5–1 nm is suitable for plotting; refine around crossings to ≤0.25 nm. Dense DFT samples alone do not establish accuracy. Compare a longer time window and independent narrowband values at the peak, band edges, and one tilted case.

Proposed numerical acceptance at nominal and representative perturbed points:

| Check | Pass criterion |
|---|---|
| Consecutive finest accepted grids | peak transmission changes <0.5 percentage points and <0.03 dB; peak λ changes <1 nm |
| 1-dB spectral crossings | each edge changes <1 nm |
| Physical domain / PML / port location / longer runtime | each isolated change shifts relevant transmission <0.02 dB |
| Half-cell geometry translation | transmission change <0.03 dB and no systematic staircase optimum |
| Total energy balance | residual <1% of incident power over evaluated band |
| Uniform-guide mode vs flux calibration | agreement within 1%; negligible spurious backward component |
| Uniform source beam calibration | waist/center/direction errors documented and reduced enough to move η <0.02 dB |

These are proposed engineering thresholds, not guarantees of correctness and not paper-reported uncertainties. Tighten if numerical error masks a small plotted difference. Never loosen silently to make a run pass.

## 7. Sweep strategy

1. Resolve source, polarization, material and gap hypotheses on analytic/reference cases first.
2. Reconstruct nominal geometry with the three published optimized values fixed: twg = 262 nm, θwg = 53.5°, y = 11 µm.
3. If required, scan missing gap/intercept and length assumptions at 1550 nm, then compare a few spectral points. Record all candidates and scan bounds. Do not perform a five-dimensional dense high-resolution grid.
4. Freeze one candidate model using nominal data only. Hold out Fig. 4 and Fig. 5 for validation.
5. Reproduce the exact plotted variants, then finer fixed-1550-nm sweeps to determine tolerance crossings. Do not re-optimize other parameters inside a tolerance sweep.
6. Only after 2D convergence, run 3D widths and transverse offsets using identical in-plane geometry/material/source conventions.

A 2D model cannot produce finite-width truncation, lateral modal content, or out-of-plane displacement sensitivity. An analytic Gaussian-aperture calculation can sanity-check these effects but must not be labeled a 3D reproduction. Likewise, a finite 3D Gaussian illuminating an infinitely wide slab can excite many lateral components, so compare the same guided-power definition when discussing approach to the 2D limit.

## 8. Compute sizing and execution policy

Observed planning host: macOS arm64, 24 GiB RAM. These are illustrative cell-count estimates, not measured runtimes and not fixed domain specifications:

| Domain and mesh | Cells | Minimum raw double-precision field storage |
|---|---:|---:|
| 2D 40 × 35 µm, 20 nm | 3.5 million | ~84 MB for three real field components |
| 2D 40 × 35 µm, 10 nm | 14 million | ~336 MB for three real field components |
| 3D 40 × 35 × 30 µm, 40 nm | ~656 million | ~31.5 GB for six real field components |
| 3D 40 × 35 × 30 µm, 20 nm | 5.25 billion | ~252 GB for six real field components |

Actual allocations include material arrays, PML, DFTs, mode solving, possible complex fields and overhead. PML may enlarge the quoted domain. Do not launch the illustrative 3D domains on this laptop. Benchmark one 2D initialization and run; measure peak RSS and cell-updates/second. At 20 nm and Courant 0.5, 1 ps is roughly 30,000 steps; this does not establish the necessary runtime.

Bound local job concurrency by measured memory, initially one worker. Store only selected field planes/frequencies, not full 3D volumes at every step. Cache reference runs by a hash including source, materials, mesh, domain and monitor definition. Resume parameter sweeps at the run level with immutable completed outputs. A solver-state checkpoint is optional and must be validated before depending on it.

For 3D choose between a reduced/converged domain, validated symmetry at zero offset, MPI on a larger-memory machine, or a benchmarked accelerator backend. Positive/negative offset cases generally break source symmetry. Do not silently use periodic transverse boundaries, artificial extrusion of a 2D Gaussian, or coarse meshes to fit RAM. Infrastructure provisioning and spending are outside this planning deliverable.

## 9. Optional homebuilt 2D scope

Only activate this track if there is a specific benefit from an independent implementation. Minimum scope is one explicit polarization branch of the Maxwell Yee scheme; a scalar Helmholtz animation is insufficient.

It must include validated staggered E/H updates and units, a CFL check, graded CPML, a one-way equivalent-current or TFSF Gaussian source, complex DFT accumulation, correctly collocated flux integration, physical oblique-layer geometry treatment, incident-reference subtraction, and power-normalized modal projection. Add a dispersive ADE formulation only if the chosen material model requires it.

Acceptance prerequisites are vacuum/material plane-wave dispersion, analytic Fresnel reflection below/above critical angle, oxide evanescent decay, slab-mode effective index and power conservation, rotated uniform-guide transmission, PML reflection convergence, and agreement with a trusted solver on a simplified wedge. This is a separate implementation project; avoid building it in parallel merely to draw the same plots.
