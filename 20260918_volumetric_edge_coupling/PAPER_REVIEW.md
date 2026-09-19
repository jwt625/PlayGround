# Paper extraction and visual review

Source: [Elshehaby et al., arXiv:2609.20686v1](https://arxiv.org/abs/2609.20686v1), submitted 17 September 2026; six PDF pages. Page numbers below are the printed PDF numbers. The PDF text and the original embedded figures were inspected, including the figures' legends and insets. This is a simulation paper and a single-channel proof of principle, not an experimental demonstration of a multicore package.

Evidence labels used in this package:

- **Reported:** explicitly stated in paper text or tables.
- **Figure:** read from a figure label or visually estimated from a curve; estimates are not raw data.
- **Derived:** calculation from stated inputs, with the assumptions exposed.
- **Proposed:** implementation or acceptance choice for this reproduction.
- **Unknown:** insufficiently specified by the available paper.

## 1. What is being reproduced

A Gaussian beam propagates through silicon toward an inclined silicon/oxide boundary. TIR produces an evanescent field in the oxide. A nearby thin silicon waveguide intercepts this field. A small difference between the waveguide angle and the etched sidewall angle makes the oxide separation vary along the interaction, shaping the coupling strength.

The intended primary observable is forward guided-mode power divided by incident Gaussian-beam power. It is sampled on the inclined waveguide after the interaction and before the subsequent bend. The paper also plots spectra under geometry/alignment changes, a width-dependent 3D comparison, and a reflection signal under source tilt.

Multicore coupling, interchannel crosstalk, mode conversion into a narrow single-mode planar routing waveguide, fabrication yield, and a working alignment feedback loop are future concepts rather than simulated results to reproduce here.

## 2. Extracted parameter ledger

| Parameter | Value / interpretation | Evidence |
|---|---|---|
| Original solver | Ansys Lumerical FDTD; 2D optimization followed by selected 3D simulations | Reported, p. 3 |
| Boundary conditions | PML on every domain boundary | Reported, p. 3 |
| Substrate / TIR incident medium | Silicon; beam reflects back into silicon at a silicon–oxide interface | Reported, operation principle, p. 2; Fig. 2 |
| Waveguide material | Silicon | Reported, p. 2 |
| Spacer | Buried oxide, wedged ridge | Reported, p. 2 |
| Cavity surroundings above receiving guide | Appears open in schematic; precise upper-cladding material is not specified | Figure / Unknown |
| Wafer / exposed facet | (100) silicon, {111} KOH sidewall | Reported, p. 2 |
| Sidewall angle to wafer plane | 54.74° in operation text; 54.7° in abstract/captions | Reported, pp. 1–2; retain both roundings |
| Waveguide inclination | 53.50° | Reported, Table 1, p. 3; reference/pivot needs explicit reconstruction |
| Waveguide thickness | 262 nm | Reported, Table 1; thickness direction is not rigorously defined |
| Gaussian waist radius | 5 µm | Reported, p. 3; waist position and exact source convention absent |
| Polarization | TE | Reported, p. 2; field-component mapping not specified |
| 2D propagation / beam-profile axes | +x propagation, y transverse profile | Reported, p. 3 |
| Nominal source height | y = 11 µm relative to cavity bottom | Reported, p. 3, Table 1 |
| Nominal source tilt | 0° relative to x | Reported, Table 3 and Fig. 5 caption, p. 5 |
| 3D width variants | 10 µm and 15 µm | Figure 3 legend, p. 3 |
| Plotted spectral interval | 1500–1600 nm | Figures 3–5 |
| Peak transmission | 0.88, approximately −0.56 dB | Reported, p. 3, Table 1 |
| Peak wavelength | 1550 nm | Reported, Table 1 |
| Reported 1-dB band edges | 1507.23 nm and 1593.68 nm | Reported, Table 2, p. 4 |
| Reported 1-dB bandwidth | 86.45 nm | Reported, Table 2 |
| Waveguide angle tolerance | approximately ±0.4° around 53.5° | Reported, p. 4, Table 3 |
| Thickness tolerance | approximately ±7 nm around 262 nm | Reported, p. 4, Table 3 |
| Vertical displacement tolerance | approximately ±2 µm around y = 11 µm | Reported, p. 4, Table 3 |
| Source tilt tolerance | approximately ±0.8° around 0° | Reported, p. 5, Table 3 |
| Transverse displacement | weak degradation through 5 µm; symmetry asserted | Reported, pp. 4–5; no numerical 1-dB limit provided |
| Tolerance definition | maximum 1-dB drop relative to nominal coupling at fixed 1550 nm | Explicitly reported, p. 4 |
| Output reference plane | inclined guide immediately after interaction, before bend | Explicitly reported, pp. 3 and 6 |

No mesh spacing, time step, PML thickness, decay threshold, domain size, numerical material index, material database fit, monitor aperture, mode order, or raw simulation files are given in the six-page paper.

## 3. Simulation inventory and exact plotted cases

| Paper result | Dimensionality | Cases to reproduce | Required observable |
|---|---|---|---|
| Geometry optimization described on p. 3 | 2D | Sweep thickness, source y, guide angle; original ranges/algorithm absent | Best coupling at 1550 nm plus response surface near nominal |
| Fig. 3, Tables 1–2 | 2D baseline; 3D width comparison | Infinite-width 2D, 10 µm 3D, 15 µm 3D | Guided transmission versus wavelength; peak and 1-dB edges |
| Fig. 4(a) | 2D | 53.25°, 53.50°, 53.75° | Three spectra with all other settings fixed |
| Fig. 4(b) | 2D | 257, 262, 267 nm | Three spectra with all other settings fixed |
| Fig. 5(a) | 2D | y = 10, 11, 12 µm | Three spectra |
| Fig. 5(b) | 3D | transverse offset 0 and ±5 µm | Spectra; paper displays a combined ±5 µm label |
| Fig. 5(c) | 2D | source tilt −1°, 0°, +1° | Three spectra |
| Fig. 5(d) | Presumed 2D based on tilt analysis; not independently stated | source tilt −2°, −1°, 0°, +1°, +2° | Normalized reflected power versus tilt |
| Table 3 | 2D for listed variables | Finer sweeps than the three illustrative cases | Left and right 1-dB crossings at 1550 nm |

Figures 1–2 are concept illustrations, not field maps from FDTD. A useful reproduction should add actual field and Poynting-flow diagnostics, but label them as new validation artifacts.

## 4. Visual findings to preserve

Original embedded images are available locally rather than relying on OCR:

- [Fig. 2](references/2609.20686/embedded/img-51.png): the Gaussian lobe is drawn within silicon; oxide separates substrate and receiving guide; the reverse-propagating guided field in panel (b) moves from a large gap toward a small gap. No dimensional scale or absolute gap is present.
- [Fig. 3](references/2609.20686/embedded/img-58.png): the 15 µm trace nearly coincides with the 2D trace; the 10 µm trace peaks about 0.2 dB lower. These are visual estimates, pending digitization. Increasing width changes modal content as well as aperture interception; this is not by itself proof of single-mode transfer.
- [Fig. 4](references/2609.20686/embedded/img-68.png): larger guide angle and larger thickness move the spectral peak toward longer wavelengths in the plotted cases. A correct peak value with the wrong sensitivity direction is not a successful reproduction.
- [Fig. 5(a)](references/2609.20686/embedded/img-75.png): y = 10 µm is visibly worse; y = 11 and 12 µm almost overlap. Do not impose symmetric response around y = 11 or equate the plotted ±1 µm samples with the reported ±2 µm tolerance.
- Fig. 5(c): increasing tilt moves the peak toward longer wavelengths. The ±1° curves peak near/outside the plotted window; their loss at fixed 1550 nm is what matters for tolerance.
- Fig. 5(d): visually estimated ordinate values at −2, −1, 0, +1, +2° are about 1.00, 0.65, 0.32, 0.42, 0.77. These are approximate readings only, with at least ~0.03 ordinate uncertainty. The normalization denominator and wavelength/aggregation are unspecified.

The reflection minimum is roughly 0.32 on the figure's scale, while nominal transmission is about 0.88. If both were fractions of the same incident power at the same wavelength, their sum would exceed one. This flags an undefined normalization or differing measurement condition; it is not evidence of gain. Do not force raw reflectance to 0.32 or define it as 1 − transmission. Preserve the published curve and compute physically normalized reflection independently.

The red reflected arrow in Fig. 2 is schematic. At a 54.74° sidewall it need not point exactly vertically. Use the vector reflection law for field validation and monitor placement.

## 5. Missing inputs and severity

| Unknown | Why it matters | Next action / permissible provisional treatment |
|---|---|---|
| Absolute oxide gap, endpoints, slope intercept, interaction length | Exponential sensitivity; angle alone does not define coupling | Highest priority: recover dimensions from author files; otherwise explicitly scan a bounded family |
| Cavity depth, base width, sidewall endpoints, source-to-sidewall distance | Defines the meaning of y = 11 µm, clipping, and available interaction length | Preserve an explicit cavity-base origin; record provisional dimensions |
| Rotation pivot when changing guide angle | Changes both gap and apodization | Declare base-anchored and/or fixed-gap-at-beam hypotheses; never silently switch |
| Thickness measured normally versus along a global axis | 262 nm can describe substantially different strips | Primary hypothesis: physical normal thickness; retain axis-thickness interpretation as an unresolved alternative |
| Upper cladding and oxide termination | Changes effective index and leakage | Check air-clad and oxide-clad interpretations only as named hypotheses |
| Refractive-index data / dispersion / absorption | Affects phase matching, spectrum, material loss | Record chosen data and separate dispersionless debugging from final material model |
| TE field components | Opposite conventions occur between incidence-plane and solver labels | Run mode and TIR diagnostics for explicit Ez and Hz branches before selecting a branch |
| Source focus, aperture, 2D beam convention, wavelength dependence | Alters beam footprint and frequency response | Calibrate waist and incident flux in homogeneous silicon; use narrowband checks |
| Entrance facet and sourcepower normalization | Determines whether reported efficiency includes fiber-to-silicon Fresnel loss | Primary internal-coupling model begins in silicon; treat package entrance as separate unresolved scope |
| Mode index and number of expanded modes in 3D | A 10–15 µm wide guide can be laterally multimode | Report fundamental, sum of guided modes, and total flux separately |
| Width used in Fig. 5(b) | Directly controls transverse-offset response | Provisional 15 µm, explicitly marked; test 10 µm as sensitivity |
| Tilt pivot / source position held fixed | Tilt can move spot as well as change k-vector | Primary fixed source-center case plus one fixed-impact-point diagnostic |
| Reflection detector geometry, denominator, evaluation wavelength | Needed to compare Fig. 5(d) quantitatively | Absolute energy accounting first; published-shape comparison only until clarified |
| Numerical convergence and raw curve data | Sets reproducibility error floor | Independent convergence and digitization with uncertainty |

## 6. First-principles cross-checks

These are derived diagnostics, not additional parameters reported by the authors.

1. With a horizontal incident ray and sidewall inclination α = 54.74°, incidence relative to the normal is 35.26°. With provisional nSi = 3.48 and nOx = 1.444, the critical angle is approximately 24.5°, so TIR is plausible **from silicon into oxide**. A beam in air incident directly on silicon does not implement this mechanism.
2. For those assumed indices at 1550 nm, oxide amplitude decay length is approximately λ/[2π sqrt(nSi² sin²θi − nOx²)] ≈ 0.18 µm. Tens of nanometers of gap uncertainty can matter. This is the amplitude decay length, not the intensity decay length.
3. The sidewall/guide angle difference is about 1.24°. In sidewall coordinates the nominal separation slope is about tan(1.24°) ≈ 0.0216, or ~22 nm gap change per µm of sidewall distance. A 10 µm interaction changes gap by roughly 0.22 µm. The absolute intercept remains unknown.
4. Projecting incident momentum onto the sidewall gives an effective tangential index near nSi cos(54.74°) ≈ 2.01; projecting onto the guide gives approximately nSi cos(53.50°) ≈ 2.07. Calculate actual isolated and substrate-loaded modes for each polarization/cladding hypothesis. A large mismatch is a reason to inspect the interpretation, not to change the reported angle or thickness silently. The structure is finite and wedged, so exact translational phase matching is only a diagnostic.
5. 0.88 corresponds to −0.555 dB. A 1-dB drop relative to that peak means transmission about 0.699, or −1.555 dB relative to incident power. Absolute −1-dB transmission instead means 0.794. Export both bandwidth definitions because Table 2's heading alone is ambiguous; use relative-to-peak as the main reproduction convention.

## 7. Questions for the authors, if exact reproduction is pursued

This is a prepared request list, not an email that has been sent. Corresponding address in the paper: melkabbash@arizona.edu.

1. Can you share the Lumerical project, construction script, raw monitor exports, and software version?
2. What are cavity dimensions and coordinates; oxide gap at both interaction endpoints; guide-angle pivot; and physical thickness direction?
3. What are the material datasets, top cladding, TE field components, beam focus, source plane, and source-medium settings?
4. Is incident normalization before or after the fiber/silicon entrance? Is a facet/AR layer included?
5. Which guide modes enter the 2D/3D transmission, where are the monitors, and what width was used for Fig. 5(b)?
6. What is Fig. 5(d)'s wavelength, reflection collection aperture and normalization? Are the Table 2 edges relative to peak or an absolute insertion-loss level?
7. What mesh, subpixel treatment, PML, stopping condition and convergence checks were used?

## 8. Claim boundaries

An independently converged solution for a declared reconstruction can be scientifically useful even if it misses 88%. It must be labeled an assumed-geometry reproduction attempt. A fitted gap/material/source family that matches the nominal spectrum is not uniquely identified by that fit. Hold out fabrication/alignment traces to test it. Do not call missing 3D results reproduced by a 2D model, or call pre-bend transmission a packaged I/O efficiency.
