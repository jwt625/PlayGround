# 005 — Nominal modes and constrained randomization

Status: sourced nominal anchors and executable candidate sampler delivered; numerical envelopes await solver validation. Date: 2026-09-13. Machine-readable authority: `data/presets.json` (18 presets); `source_ids` link every sourced anchor to its evidence. Ranges below are educational design choices unless explicitly identified as vendor MFD tolerances. They are not PDK rules, process distributions or guarantees of single-mode operation.

## Fiber and analytic categories

| Category at 1550 nm unless stated | Nominal MFD, µm | Sampling envelope, µm | Evidence / interpretation |
|---|---:|---:|---|
| SMF-28-like C band | 10.4 | 9.9–10.9 | [Corning](https://www.corning.com/content/dam/corning/media/worldwide/coc/documents/Fiber/product-information-sheets/PI-1424-AEN.pdf) specifies 10.4±0.5; Gaussian replacement is approximate |
| SMF-28-like at 1310 nm | 9.2 | 8.8–9.6 | Same datasheet specifies 9.2±0.4 |
| UHNA1-like | 4.8 | 4.5–5.1 | [Coherent datasheet](https://www.coherent.com/resources/datasheet/components-and-accessories/specialty-optical-fibers/uhna4_spec_202011122126.pdf) |
| UHNA4-like | 4.0 | 3.7–4.3 | Same source |
| UHNA7-like | 3.2 | 2.9–3.5 | Same source; 1500–2000 nm specified operating band, so do not reuse its C-band preset at 1310 nm |
| Generic HNA teaching family | 4.0 | 3–6 | Deliberate broad family, not one product |
| Lensed fiber at focus | 2.5 | 2–5 | Generic assumed air-side Gaussian waist; [Thorlabs measurement note](https://media.thorlabs.com/globalassets/family-pages/SharedAssets/s/si/single_mode_lensed_fibers_app_note.pdf) motivates explicit plane/working-distance handling |
| Elliptical Gaussian | 4×3 | each axis 1–12 | Pure analytic educational range |

Use anchor wavelengths first. Do not independently randomize commercial-fiber MFD and wavelength. A later generic C-band family can use a documented MFD(λ) model; the named presets stay at their cited measurement wavelengths. Gaussian waist convention is in K002. For lensed fibers derive divergence `NA≈λ/(πw0)` under the paraxial air model; working distance, waist, phase curvature and NA cannot all be arbitrary independent knobs. Current lensed preset is at focus only.

## Integrated guides (all dimensions µm)

| Preset | Nominal | Routine candidate envelope | Interpretation / eligibility |
|---|---|---|---|
| SOI strip | width .45, height .22 | w .35–.65; h .21–.23 | Oxide clad; quasi-TE0 requested, guidance verified numerically |
| SOI wide | .9×.22 | w .7–1.5; h .21–.23 | Multimode expected; explicit TE0/TE1 identity |
| SOI rib | w .65, film .22, residual slab .09 | w .5–.9; slab .06–.15; film .21–.23 | slab < film |
| Thick SiN | 1.5×.8 | w 1–2; h .7–.9 | Possible higher modes; fundamental selected |
| Medium SiN | 1.2×.35 | w .7–2; h .3–.4 | No blanket single-mode claim |
| Thin/wide SiN | 3×.15 | w 2–5; h .1–.25 | Ellipticity and weak vertical confinement; domain check |
| Si SSC endpoint | Si tip .10×.22 inside 4×3 outer guide, n=1.46 | tip .06–.18; outer w 2–7, h 2–6, n 1.45–1.50 | Hypothetical centered compound guide in silica; MFD is an output |
| SiN taper endpoint | .2×.35 | w .1–.4; h .15–.4 | Potentially very weakly confined; reject boundary/slab leakage |
| TFLN ridge | top w 1.2, film .6, etch .3, sidewall 70° from horizontal | w .8–1.8; film .4–.7; etch .2–.4; angle 60–80° | x-cut; propagation along crystal Y, vertical X; air clad; tensor solver required |
| SiN/LN hybrid | SiN 1.2×.35, LN film .6, oxide gap .2 | SiN w .8–1.8, h .3–.4; LN .4–.7; gap .1–.5 | Assumed stacked design; test lateral localization and hybrid-mode identity |

Evidence anchors: historical [imec platform sheet](https://www.imec-int.com/drupal/sites/default/files/2019-03/IC-link%20integrated%20si-photonics%20platform_web.pdf) lists 450 nm C-band strip and 650 nm rib widths; [SOI coupler research](https://arxiv.org/abs/2312.13329) discusses 220 nm SOI. [LIGENTEC](https://www.ligentec.com/technology/) documents .8/.35/.15 µm thicknesses, not the game width ranges. The thin .15 µm C-band game guide is an educational extrapolation, not a claim of AN150 C-band qualification. [LN research](https://www.nature.com/articles/s41467-024-46512-4) documents .6 µm x-cut film and .3 µm target etch; the game's width/angle/envelope are independent assumptions.

Initial substrate model is semi-infinite silica. It excludes silicon-handle leakage; packaging-realistic SSC validation must add finite BOX, handle, undercut and actual outer material. Top width and residual slab are explicit. No nominal neff, MFD, overlap or insertion loss is assigned to an unsolved guide. “Oversized SSC” is a candidate until solved MFD/field moments confirm it.

## RNG policy

Implemented `samplePreset`: versioned Mulberry32 uint32 seed, triangular samples centered on each nominal, bounded cross-constraint rejection, fixed supported anchor wavelength, explicit unsolved status. This is candidate geometry generation, not a physical mode certification. Routine height envelopes show design exploration, not wafer process tolerances. Advanced SOI envelope .3–.9 wide and .18–.30 thick is recorded separately and not selected by default.

For future numerical pairs, choose a preset first; share process thickness/material/cladding across A/B unless that variable is the lesson. Change one variable at L1–2; two at L3; broader coupled parameters later. For fabrication-tolerance questions use separate correlated process-error models, initially user-chosen ±10/25/50 nm width perturbations, not the broad design envelopes. Wavelength additions must have validated dispersion and mode selection.

Alignment sampling uses solved field size, not core width. For Gaussian effective radii define `s_j=sqrt((wAj²+wBj²)/2)`. Then pure shift loss multiplies η by `exp(−Σ(dj/sj)²)`. Ordinary |d|/s: 0–1.5; harder cases up to 2.5. At identical circular waists, 1 dB occurs at d=0.480w=0.240 MFD; 3 dB at d=0.831w=0.416 MFD. Thus SMF 10.4 µm gives about 2.50/4.32 µm, versus HNA 4 µm giving .96/1.66 µm. These are analytic inferences, not measured assembly tolerances.

Phase-ramp ordinary air tilt 0–3°, hard cap .1 rad (5.73°); use dimensionless q*w to balance losses. For equal circular modes, ηtilt=exp[−(π n_gap w sinθ/λ)²]. Large modes tolerate more translation but less angular error. No literal high-index kink uses this transform.

Implemented analytic pool samples a target bucket, an actual η inside it, a random waist, and derives the required offset. Rotate offset direction randomly, vary circular/elliptical mode sizes, and accept only normalized displacement ≤2.5. This inverse construction guarantees coverage without extreme geometries; it currently trains alignment only. Mandatory analytic anchors separately exercise mismatch, tilt and polarization. Broad balanced multi-concept sessions are C4 work.

- [x] K5.1 Record vendor fiber anchors and distinguish named/generic models.
- [x] K5.2 Serialize nominal guide recipes and proposed bounds with provenance.
- [x] K5.3 Implement deterministic bounded candidate sampling and an analytic coverage bank.
- [ ] K5.4 Solve per-preset nominal + envelope corners; measure neff, MFD/moments, guidance and mesh errors, then tighten bounds.
- [ ] K5.5 Add finite BOX/handle/real SSC stacks and process-specific indices.
- [ ] K5.6 Publish geometry-to-mode maps and measured acceptance rates before enabling numerical RNG in gameplay.
