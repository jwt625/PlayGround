# 002 — Physics definitions and validation knowledge base

Status: analytic contract implemented; vector definition requires restricted-domain validation. Date: 2026-09-13.

## Scalar Gaussian contract

At a common waist plane, `u ∝ exp(-x²/wx²-y²/wy²)`, with ∫|u|² dA=1 and intensity 1/e² diameter MFD=2w. For aligned ellipse axes, displacement d, relative transverse phase slope q and linear polarization angle φ:

`η = cos²φ × ∏[2 wAj wBj / Sj × exp(-2 dj²/Sj - qj² wAj² wBj²/(2 Sj))]`, where `Sj=wAj²+wBj²` and j=x,y.

For the implemented paraxial air-gap model, `qj=(2π n_gap/λ) sin θj`. The two component angles are small, each ≤0.1 rad; n_gap is explicit, never silently neff. This is a scalar paraxial phase-ramp model, not a rigidly rotated Maxwell solution. No curvature, longitudinal gap propagation or Fresnel factors in current results. For equal circular modes, shift-only η=exp(-d²/w²); circular 2× waist mismatch η=0.64. An x-only 2× mismatch gives 0.8. A π/2 polarization rotation is exactly zero within numerical tolerance.

## Vector contract — do not assume every bilinear overlap is a bounded efficiency

For forward lossless modes use `P=0.5 Re ∫(E×H*)·ẑ dA`. A common reciprocal flux overlap is `C_AB=1/4 ∫[E_A×H_B* + E_B*×H_A]·ẑ dA`, normalized by sqrt(P_A P_B). This gives self-overlap one. Tidy3D documents the corresponding [mode overlap workflow](https://www.flexcompute.com/tidy3d/examples/notebooks/ModeOverlap/) and [monitor integral](https://feature-examples.d3nzcgsw5oo0x1.amplifyapp.com/tidy3d/examples/notebooks/XarrayTutorial/).

Crucial gate: the flux bilinear form is not generally a positive-definite inner product over arbitrary modes from different high-index structures. Squaring a symmetrized flux overlap across incompatible impedances can exceed one. Do not clamp that failure into an apparently valid coupling efficiency. Validate the convention for each allowed interface; exclude lossy/leaky/backward modes and unsupported interfaces. A rigorous discontinuity question needs mode matching/scattering, including reflection. Record raw coefficients and reject outside the supported efficiency domain. Same-guide orthogonal forward bases provide the cleanest multimode projection exercises. `1−Ση_n` is only “not captured by the selected forward guided modes” under the stated complete/orthonormal projection assumptions; it is not automatically radiation loss.

## Knowledge tasks

- [ ] K2.1 Select and independently verify the production vector projection convention for cross-platform endpoint questions, including different-impedance counterexample.
- [ ] K2.2 Material functions: primary dispersion data for Si, silica, process-specific SiN, LN ordinary/extraordinary; validity wavelengths/temperature and tensor axes. Never infer vendor index profiles from MFD alone.
- [ ] K2.3 Benchmark fixtures with meshes, complex fields, normalization and solver/license versions. Reference [Fallahkhair et al., JLT 2008](https://photonics.umd.edu/pubs/ja-20/) for vector FDE; inspect license before source reuse.
- [ ] K2.4 Step-index fiber: correlate core radius/NA/wavelength; V=2πa NA/λ<2.405 for LP01-only ideal step-index questions. Named commercial fiber is not defined fully by its Gaussian MFD and nominal NA.
- [ ] K2.5 LN crystal cut, propagation axis and sidewall geometry, then anisotropic benchmarks. An isotropic proxy cannot teach LN polarization.

Proposed release tolerances (engineering choices, not literature constants): Gaussian analytic-vs-quadrature |Δη|<1e−8; slab |Δneff|<1e−4; external vector |Δneff|<1e−3 and |Δη|<0.005; ordinary mesh/domain changes |Δη|<0.005 and precision mode <0.001. Also check solver residual <1e−8 under a documented relative norm, positive forward power, symmetry/parity, stable mode identification and boundary energy fraction <1e−6 plus expanded-domain confirmation. Mesh convergence alone does not validate a wrong operator.
