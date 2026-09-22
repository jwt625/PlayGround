# R6-P1: geometry repair before transfer diagnosis

Review date: 2026-09-19. Inspected the committed port geometry/configuration, status report, geometry image, run/monitor paths and T02 implementation. This review contains analytic calculations and documentation changes only; no implementation code or FDTD runs.

**Decision:** repair the constructed geometry first. Then run one coarse Hz simulation with simultaneous probes at s=31, 34, 38, 41 and 44 µm, plus a genuinely uniform downstream reference if needed. Defer the full-domain 50/75 ladder and forward/reverse production pair. Do not lengthen the transition or report a physical splice loss from the current result.

## 1. Blocking construction defects

### G1 — The oxide polygon changes the nominal interaction upstream

`src/geometry.py` samples the curved substrate boundary but constructs `oxide_poly` from only four vertices: the two bottom endpoints and the two guide-face endpoints. Its lower boundary is therefore a straight chord from s=9 to s=58, not the declared q_sub(s).

The geometry list places oxide after substrate. Meep gives later overlapping objects precedence, so this chord replaces silicon with oxide upstream of the intended splice. Where the chord instead lies above the substrate boundary, it leaves an unintended air gap. [Meep geometry precedence](https://meep.readthedocs.io/en/latest/Python_User_Interface/#simulation).

For the committed nominal port parameters, the endpoint is q_sub(58)=-2.288048 µm and the oxide chord is `q_chord(s)=-2.288048*(s-9)/49`:

| Location | Consequence of current oxide polygon |
|---|---|
| Old nominal impact coordinate s=11.02211 | Replaces approximately 94.42 nm of substrate along its normal; local oxide thickness becomes about 158.19 nm instead of 63.77 nm at that coordinate. This is not a newly calculated physical beam-impact location. |
| s=31 | Replaces approximately 1.02729 µm of substrate; local oxide thickness becomes 1.52349 µm instead of 0.49620 µm. |

Thus the assertion that the interaction is preserved through s=33 is false for the **composed material geometry**, despite q_sub itself being zero there. Comparing the historical nominal power of ~60 with the new ~0.006 cannot measure splice transmission: the excitation/coupling region already differs.

**Required repair:** use the same ordered, sufficiently resolved boundary vertices for both adjacent materials. Include the exact splice endpoints, eliminate unintended overlap/air voids, and export the composed dielectric rather than separate polygon outlines. Merely reordering objects would not repair both overlap and void regions.

### G2 — The purported uniform section is not parallel to the guide

For s≥41 the current q_sub returns a constant. That boundary is parallel to the 54.74° sidewall, while the strip remains at 53.5°. It also gives the wrong derivative at the end of the transition.

For delta=alpha-theta_wg and desired normal oxide clearance D, the downstream boundary must satisfy:

`q_sub(s) = g(s) - D/cos(delta)`.

This has the same slope as g(s), giving constant guide-normal separation. A C1 blend from q_sub=0 to this moving target is sufficient for the initial 8 µm diagnostic; match the endpoint slopes, not only positions. State whether transition length is measured along the sidewall or guide. If measured along the guide, convert using `delta_s = L_guide*cos(delta)`.

**Required repair:** verify constant normal clearance at multiple downstream coordinates, continuous boundary position/slope at both joins, and identical material cross sections translated along the guide in the uniform region. Keep the original length/clearance until this corrected baseline is tested.

### G3 — The guide terminates inside the cell instead of in PML

The isolated-port branch now includes the entire guide endpoint s=58 in domain sizing and then adds 3 µm clearance plus 1.5 µm PML. The guide and oxide consequently end before PML, creating a physical terminal facet. This contradicts the absorbing-continuation contract and can return reflected power.

**Required repair:** size the domain from the physical measurement/source region and clearance requirements, then extend all uniform port materials through the appropriate PML to the cell boundary/beyond. Treat the numerical extension endpoint separately from a physical guide endpoint. Plot the actual PML inner boundary and material continuation. Check termination sensitivity after repair.

### G4 — The reported port aperture straddles the transition

A vertical monitor centered at s=44 with half-span 5 µm samples sidewall coordinates approximately **39.917–48.083 µm**. The transition ends at s=41. Its center is downstream, but the complete aperture is not within a uniform section. This prevents treating it as a validated translationally invariant port cross section.

**Required repair:** use the previously specified tail-converged aperture (initial guide-normal half-width 1.5 µm, vertical half-span about 2.52 µm), or move the monitor farther downstream. Confirm the *entire* source, monitor and eigensolver volume lie in uniform material. Local probes inside the splice remain diagnostic cross sections, not automatically well-defined modal ports.

The displayed geometry image does not show the actual substrate polygon, composed dielectric, monitors or PML bands. Its dashed sidewall and straight blue oxide edge therefore cannot validate this continuation.

## 2. Interpret the index discrepancy with a matched reference

The 2.3241 value at 25/µm differs from the continuum isolated value by 0.0383. This is not enough to diagnose inadequate 3 µm isolation. The guide spans only 6.55 cells; orientation, material sampling, eigensolver aperture and smoothing can change the discrete index. The continuum oxide decay length of the isolated Hz fundamental is roughly 0.14 µm, making direct loading across a correctly constructed 3 µm oxide barrier a weak explanation by itself.

Before a costly propagation run, perform a compact local mode calculation of the corrected uniform port and a truly isolated strip with **matched angle, mesh, subcell registration, smoothing and eigensolver aperture**. Export fields and compare their indices and identity metrics. Separate finite-resolution error relative to the analytic value from loading error relative to the matched numerical reference. The previous |delta neff|≤0.005 continuum criterion is a converged target, not a requirement that a coarse mesh already attain it.

The standard `src/run.py` and `scripts/validate_port.py` still compute s_mon=31 from this config; `r6_reciprocal.py` retains the previous source/monitor placement. The s=44 measurement may have come from an ad hoc call, but it is not reproduced by the committed configuration alone. Save the exact command, effective monitor definitions, raw fields/coefficient arrays and immutable run record. Do not infer a reproducible s=44 port from the YAML filename.

## 3. Cheapest useful diagnostic sequence

| Step | Work | Completion criterion |
|---|---|---|
| D0 | Repair G1–G4; no time stepping. Export analytic boundaries, composed dielectric and a nominal-versus-port difference map on a common physical grid. | Materials agree throughout the preserved interaction through s=33, apart from quantified numerical sampling error; no air sliver; correct uniform clearance; port reaches PML. Use common grid registration so a shifted cell does not masquerade as a material change. |
| D1 | Compact matched port/isolated eigenmode checks; Hz first. | Identified strip fundamental, stable against local eigensolver aperture; index difference attributed to loading versus discretization. Prior identity metrics remain applicable at the isolated port. |
| D2 | One forward Hz run at 25/µm, **1550 nm only**, probes s=31/34/38/41/44, and a farther uniform port if aperture clearance requires it. | Save native signed flux, forward/backward candidate coefficients, dielectric/mode fields and local strip participation for every plane. Do not call local band-number changes loss. |
| D3 | Locate any power decrease using signed flux through a control volume around the transition, including side radiation and reflection. | A loss is accounted for by radiation/reflection/channel conversion, not just a disappearing selected coefficient. Compare upstream/downstream in the same run and input normalization. |
| D4 | Check duration/port termination, then one diagnostic adjustment only if warranted. | Input pulse has reached the remote port, transients have decayed, and extending the run stabilizes accumulated complex amplitudes. If a real splice loss remains, compare the corrected 8 µm transition with 16 µm; do not simultaneously change clearance and length. |

Multiple monitors in one simulation are approved; this is not five separate FDTD jobs. At nonuniform planes, modal coefficients describe a chosen local basis, while total flux plus connecting side-boundary flux provides the transfer accounting. Reduce the aperture only after checking it captures the relevant guide tails; use a separate larger boundary for radiated power.

If loss remains large after these checks, it is reportable as loss of the **named numerical continuation**. It does not establish low coupling of the original interaction or the paper device. A longer substrate-loaded path can itself leak, so gentler does not guarantee better transmission. Use the measured radiation/reflection distribution to justify the next change.

## 4. Compute policy

For the quoted 75.6×68.7 µm 2D cell, the approximate spatial-cell counts are:

| Resolution /µm | Cells | Runtime extrapolation from 575 s at 25/µm |
|---|---:|---:|
| 25 | 3.25 million | 9.6 min measured |
| 50 | **12.98 million** | ~77 min |
| 75 | **29.21 million** | ~4.3 h |

The report's ~30M estimate belongs to resolution 75, not 50. Runtime estimates use R³ scaling for fixed 2D area and physical duration; initialization, geometry processing, DFT storage, eigensolves and memory pressure may change them. Counts are spatial cells, not a RAM estimate.

**Current allocation:** local geometry/eigenmode checks and one corrected coarse multiprobe run, with a duration extension or one targeted control as needed. No blind full-domain 50/75 sweep or Cartesian product of transition/clearance/polarization choices.

Reduce cost before convergence:

- Use one frequency, only the active polarization's components, line monitors and localized field windows. Do not accumulate full-cell six-component DFT maps or 51-frequency spectra for this single-wavelength diagnosis.
- Profile geometry initialization, time stepping, DFT extraction and eigensolves separately; record peak RSS. The large substrate polygon can make initialization materially different from the original block geometry.
- Use compact local domains for uniform-port and splice-only controls. For the full coupler, reduce excess margins only after retaining the source, reflected beam, collection aperture and absorbing continuation; validate truncation with a coarse comparison. Do not crop away the Gaussian/radiation channels merely to hit a memory target.
- Run one simulation at a time on the 24 GiB host. Use **12 GiB peak RSS** as the initial operating budget, leaving headroom for the OS and other applications; lower it if measured available memory requires. Do not accept swap-heavy runs as a convergence strategy.

After corrected coarse extraction and reciprocity pass, run 50/µm first on the accepted domain if the measured memory projection fits. Keep the full-domain 75/µm run deferred until timing/memory are profiled and a smaller validated domain or adequate compute is selected. Numerical convergence is postponed, not waived; no physical ceiling or tolerance claim becomes accepted meanwhile. No cloud allocation or unattended multi-hour batch is required by this handoff.

## 5. T02 continues independently, with small correctness repairs

The encoded numerical table values and relative/absolute thresholds agree with the cached paper text. However:

- Table 3 is on printed **page 5**, not page 4. The 0.88 linear peak is stated in the surrounding text, while Table 1 prints -0.56 dB; label that provenance separately. The transverse-offset note is narrative evidence, not a Table 3 row.
- `peak_and_edges` assumes ascending wavelength. Meep's ascending frequency array produces descending wavelength, which can exchange left/right edges and yield a negative bandwidth. Sort wavelength/value pairs together or explicitly validate/reorder inputs; handle duplicates/nonfinite values deliberately.
- `_crossing` misses a crossing exactly on the final endpoint because it checks only the first member of each pair for equality. Add an endpoint case along with reversed wavelength order and peak-below-absolute-threshold cases to the meaningful analysis checks.

Continue curve digitization and panel overlays. The three reported passing tests exercise useful ordinary cases but do not establish those boundary/order cases. No tests were rerun by the planning reviewer.

**Next return:** corrected material maps, matched discrete port modes, one multiprobe power/radiation record and measured compute profile. No additional decision is needed to fix construction defects or execute this bounded diagnostic.
