# T03 orchestration decision: preserve geometry, validate the diagnosis

Decision: **(a) remains the primary paper-parameter reconstruction; (b) is a bounded physical diagnostic within it.** Do not adopt alternative angles/thicknesses as the main model. Continue T05–T06 and the targeted checks below; hold production T09–T11 sweeps and 3D until they pass. Author files would improve confidence but are not required to perform these checks. No author contact has been made.

Current status is **low provisional net output flux; cause not yet isolated**. It is premature to call this a faithful disagreement with the paper or a proven phase-mismatch limit. The infinite substrate interface, finite strip endpoints, and assumed gap are a reconstruction, not a fully specified literal author geometry.

This review inspected the implementation, saved run metadata, geometry preview, and `nominal_fields_cw.png`; checked the version-pinned Meep example/source; and evaluated the analytic asymmetric-slab dispersion relation using calculator arithmetic. No implementation code was changed and no additional FDTD simulation was run.

## 1. Evidence that changes the immediate plan

### 1.1 The isolated TE index needs correction

For air / Si / semi-infinite oxide, n = 1 / 3.48 / 1.444, full physical thickness d = 0.262 µm, and λ = 1.55 µm, the fundamental lossless slab equations are:

Let h = k0 sqrt(nSi² − N²), qa = k0 sqrt(N² − nair²), and qo = k0 sqrt(N² − nOx²), with N = neff and k0 = 2π/λ. For mode order zero:

- TE/Ez: hd = atan(qa/h) + atan(qo/h).
- TM/Hz: hd = atan[(nSi²/nair²) qa/h] + atan[(nSi²/nOx²) qo/h].

These follow from continuity of the tangential fields (and the dielectric-weighted derivative for Hz). They use **full thickness**, not half thickness. All transverse decay constants refer to semi-infinite claddings. They do not describe the substrate-loaded four-layer structure.

Independent substitution brackets give:

| Check | Result |
|---|---|
| TE root at d = 262 nm | N between 2.959 and 2.960, approximately 2.960 |
| TM root at d = 262 nm | approximately 2.286 |
| Thickness implied by reported TE N = 2.615 | approximately 168.7 nm, not 262 nm |
| Thickness implied by TM N = 2.285 | approximately 261.9 nm |
| Thickness for N = 2.009 | TE ≈87.4 nm; TM ≈231.9 nm |
| Thickness for N = 2.070 | TE ≈93.7 nm; TM ≈238.2 nm |

Thus the TE discrepancy is larger than reported, while the blanket claim that matching requires an 80–85 nm core is not applicable to TM. The TE tilt estimate also needs revision. These corrections do not establish efficient coupling for either branch; they establish that the analytical diagnostic must first be made reproducible. Have the coding agent independently reproduce the equations/roots and compare MPB at identical cross section. Do not substitute the thinner cores into the headline reproduction.

### 1.2 Meep supports oblique-guide modal decomposition

An arbitrarily rotated monitor plane is not required. The official **Meep v1.34.0** [oblique-source example](https://github.com/NanoComp/meep/blob/v1.34.0/python/examples/oblique-source.py) demonstrates a rotated guide crossing an axis-aligned flux plane, followed by `get_eigenmode_coefficients` with `direction=mp.NO_DIRECTION` and `kpoint_func` aligned with the guide. The [mode tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/) describes this explicitly.

Validate that route on an isolated uniform guide first. It does not automatically supply a physically meaningful local guided basis in a longitudinally varying, substrate-leaky region. If necessary rotate the entire device computationally or use a validated isolated output extension, preserving the physical reference plane. Keep the custom perpendicular-line flux as an independent cross-check, not the sole coupling observable.

### 1.3 Output and input flux conventions differ

`src/ports.py:141–148` multiplies the sampled DFT Poynting products by 0.5, while its incident reference uses `mp.get_fluxes`. In the version-pinned [Meep dft.cpp](https://github.com/NanoComp/meep/blob/v1.34.0/src/dft.cpp), `dft_flux::flux()` integrates the real E–H product without this extra 0.5; field and flux DFT accumulation share the Fourier convention.

This identifies an expected factor-of-two convention discrepancy before interpolation/aperture errors. Validate it by comparing custom and built-in flux through the same plane in a homogeneous reference. Do not simply double the saved coupling and declare it corrected: modal identity, backward power, sampling, and termination remain unresolved. A factor of two alone would not explain the distance from 88%.

### 1.4 The output port is close to a hard termination

The saved nominal geometry has stack start s = 9 µm, stack length 16 µm, and guide extension 8 µm. The guide ends at s = 33 µm; the extraction line is at s = 31 µm. `src/geometry.py` creates a finite prism ending there, and the domain places PML farther away. The waveguide does not continue through an absorbing termination.

Consequently a reflected guided wave can return across the monitor, and **net flux is not forward guided power**. Test a validated absorbing continuation, guide-end displacement, and forward/backward decomposition before attributing small net flux solely to poor launch coupling. Any continuation must preserve the intended coupling region and have demonstrated negligible back-action.

The upstream endpoint also matters: in the saved geometry it is at y ≈12.65 µm, only ≈1.65 µm above the y = 11 µm beam center, within a 5 µm waist. Endpoint diffraction and a partly uncovered beam footprint cannot be ignored. Display the beam footprint against finite wedge endpoints; do not treat the interaction as an infinite uniform prism coupler. Extending a wedge upstream can make its gap negative, so this is a physical geometry question, not an automatic length adjustment.

### 1.5 Saved results do not yet establish convergence or visual modal identity

- `results/nominal_bca5b6f62687e410/config.effective.yaml` records **25 pixels/µm and 60 time units after the pulse**, despite the current nominal YAML requesting 50 and 200. Use the effective saved configuration in claims; preserve both records.
- The field-map script displays the absolute value of a real instantaneous CW field. It is not a complex steady-state amplitude map or time-averaged Poynting map. The image shows interference and substantial radiation, but cannot identify how much power is in a forward guide mode.
- The displayed geometry preview and field map are not sufficient to certify T04 without a common configuration/run ID, actual sampled permittivity, and port/end/PML overlays.
- `scripts/search.py` reuses one reference despite geometry-dependent focus/domain changes. Demonstrate that normalization reuse is valid, or recompute it per case.

## 2. Why substrate leakage is worth checking

Prism loading and leakage are intrinsic to prism–film coupling. The reverse process is guide power radiating into the prism; a forward incident beam can couple efficiently into the same structure if its phase/front and envelope match. A gap that grows along the receiving direction can reduce subsequent leakage. Therefore “the mode is leaky” does not by itself contradict high capture efficiency.

Primary literature explicitly treats the competition between injection and leakage: [Ulrich, JOSA 60, 1337 (1970)](https://doi.org/10.1364/JOSA.60.001337). [Mode analysis and prism coupling for multilayered optical waveguides, Applied Optics 20, 3158 (1981)](https://doi.org/10.1364/AO.20.003158) addresses loaded propagation constants, leakage rates, and tapered gaps. These sources justify a diagnostic, not the assumption that this paper's 262 nm structure has a mode with real index 2.0 or reaches 88%.

Treat substrate loading as physics already present in the FDTD geometry. Do not insert an artificial neff ≈2.0 mode or count substrate radiation as receiving-guide transmission. The isolated-slab phase mismatch is a useful warning; it is not a rigorous impossibility bound for a finite, wedged, loaded structure.

## 3. Authorized next work packages and exit criteria

The coding agent should execute these in order. R1 analytical work and T02 digitization can proceed while reference simulations run; no parallel agents are required.

| ID | Work | Required evidence / completion criterion |
|---|---|---|
| R1 | Recompute air/Si/oxide slab modes, both polarizations, at 262 nm; compare analytic relation and MPB | Equations, physical thickness, index data, solver root/residual and mode profile saved; analytic versus converged MPB neff agree to ≤0.005; explain/correct TE = 2.615 |
| R2 | Complete T05 homogeneous source and Fresnel/TIR controls, and same-plane custom/native flux calibration | Custom/native flux within 1% after a documented convention correction; source direction/waist and energy accounting verified; match actual runtime/mesh |
| R3 | Complete T06 on an isolated inclined strip: built-in oblique modal extraction, backward/forward powers, monitor displacement and absorbing termination | Mode/flux agreement within 1%; spurious backward fraction <0.1% in the reference; increasing termination length moves transmission <0.02 dB; custom line sampling converged |
| R4 | Rerun nominal reconstruction at 1550 nm for Ez and Hz with corrected measurement and termination; no gap optimization | Raw forward, backward and net power; ηfund, other channels and <1% energy residual; at least two meshes and doubled-runtime spot check, followed by T08 convergence for any claim |
| R5 | Bounded substrate-loading diagnostic at the same 262 nm thickness/materials; sample constant local gaps around 0.064, 0.15, 0.244 and 0.50 µm | Track complex β and leakage versus gap using outgoing substrate conditions, or an equivalently validated scattering calculation; compare detuning with linewidth/coupling length; do not force a real-index target |
| R6 | Reverse-launch the validated receiving mode through the actual wedge at 1550 nm, both candidate polarizations if R4 leaves ambiguity | Complex emitted beam at a homogeneous-Si reference plane, angular spectrum, emitted phase/envelope and power-normalized overlap with the original Gaussian; reciprocal efficiency difference ≤max(0.0001, 5% of the larger efficiency), with numerical uncertainty established below that threshold |

R5 freezes the wedge locally into a parallel-layer model; document this approximation and its axis choice. A real-eigenvalue closed/periodic supercell mode is not automatically a leaky resonance. Use outgoing-wave conditions and verify stability against boundaries, or inspect resonant field enhancement/phase from a transfer/scattering matrix.

**Do not diagnose a lossless infinite plane-wave stack solely by a dip in reflected intensity.** When the prism is the only propagating far-field channel, reflection can have unit magnitude while its phase and internal stored field reveal the resonance. Finite-beam lateral guide extraction is a different power-balance problem. Report the observable used.

For R6, compare coupling into the specific reciprocal Gaussian channel, not all power emitted into silicon. Use a port far enough from the loaded section to define the receiving mode, and prove that the chosen termination/extension does not change the interaction. If β, field topology or port definition is still ambiguous, resolve that rather than adding another optimizer.

## 4. Decision after the diagnostics

- If corrected metrology yields strong coupling: continue nominal-spectrum convergence and T09, then freeze assumptions before T10–T11.
- If the isolated and loaded analyses plus reciprocal angular spectrum demonstrate a mismatch and corrected coupling stays low: adopt a **validated mismatch for the declared reconstruction**. Run a compact nominal spectrum and selected paper perturbations; do not waste compute performing every fine tolerance sweep around a failed operating point.
- If a loaded mode/beam overlap explains the discrepancy: retain the same geometric baseline, document the physical explanation, and validate it through direct/reverse agreement. This is not permission to tune to a prescribed neff.
- If competing geometric interpretations remain indistinguishable: prepare the author-file request using the existing question list. The request is not a reason to suspend analytic/reference work. Sending requires the user's instruction.
- Alternative thickness, angle or large-tilt cases may be used only as explicitly labeled **positive controls** after the reference tests pass. They cannot replace the paper values or count as a reproduction.

Highest-value author question: exact construction/source script and the definition of output guided-mode transmission, including polarization, all oxide/guide vertices, termination, mode monitor, and source normalization.

## 5. Additional implementation acceptance issues to repair before production

These are code-review findings for the coding agent, not edits performed by the planner:

1. `save_run` currently overwrites files in an existing configuration-hash directory, and run identity does not encode a solver/source-code version. Preserve earlier results when correcting normalization/geometry; do not call current records immutable until overwrite prevention and version provenance work.
2. `NumericsConfig.cour` and `decay` are not applied by `run_coupler`; stopping is a fixed time. Honor or reject configured fields, and report the effective settings.
3. `10 log10(abs(eta))` hides negative signed flux. Preserve the sign and reject/flag invalid efficiency rather than making backward flow look like positive transmission.
4. Setting `wedge_sign=-1` changes polygon direction but leaves `plan.d`/`plan.m` based on the original θwg. Any such scan needs self-consistent physical guide angle, normal thickness and port orientation. Historical sign-flip results are not validated controls.
5. Zero-slope gaps are currently rejected by config validation, preventing the planned constant-gap diagnostic. Permit valid reference/control geometries without weakening checks on the production model.
6. The source/material comments should not claim author-supplied SMF-28 parameters or a prior decision making constant-index materials the final headline model. Those were not established by the paper review. Keep the constant-index model labeled a provisional baseline.

T01 and T04 have working implementation artifacts, but full acceptance remains conditional on the applicable items above. No package reinstall or wholesale solver rewrite is warranted.
