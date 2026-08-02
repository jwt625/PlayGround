# Nonuniform-FBG reconstruction notes

## What is measured in the supplied artifacts

- MDF03 heater length and FBG length: 10 mm.
- Ambient temperature: 23 °C.
- MDF03 attenuation at the 1480 nm pump: 32.43 dB/cm, or 746.7 m^-1 for power.
- Reported MDF03 hot-zone FWHM: approximately 2.6–2.8 mm and nearly power-independent.
- Cold FBG labeled peak: 1547.493 nm.
- Cold-trace peak: approximately -31.05 dBm; plot floor: approximately -44 dBm.
- Cold-trace 3 dB bandwidth: approximately 0.30 +/- 0.02 nm. The right branch is
  partially hidden by the 17 mW trace, so this was interpolated between the digitized
  2 dB width (0.264 nm) and 4 dB width (0.337 nm).
- Heater powers: 0, 17, 73, 131, 188, 244, 296, 344, and 367 mW.
- Labeled long-wavelength peaks: 1547.493, 1547.792, 1548.831, 1549.790,
  1550.589, 1551.329, 1552.068, 1552.647, and 1552.947 nm.
- Approximate FBG peak temperatures digitized from Fig. 5(b): 23, 66, 171, 251,
  315, 371, 415, 459, and 479 °C. Error bars and line thickness limit accuracy.

The paper says the *spectral edge*, not the labeled final local maximum, is used for the
temperature conversion. That distinction matters. It also contains a textual value of about
460 °C at 367 mW that is not fully consistent with the plotted point near 479 °C. The code uses
the plotted calibration because the user explicitly supplied that plot.

## Thermal parameterization

The paper's local-equilibrium model is

    T(z)-Tamb = alpha P exp(-alpha z)/(h_eff pi d).

The full constant-property cylindrical-fin equation has the analytic solution

    theta(z) = A cosh(m z) + B sinh(m z) + C exp(-alpha z),
    m^2 = h_eff pi d/(k A),
    C = alpha P/[k A (m^2-alpha^2)],

with A and B set by the end boundary conditions. `fbg_model.fin_temperature_rise_k`
implements insulated ends. With the paper's h_eff=1451 W m^-2 K^-1, the fin length is
1/m=0.172 mm, so axial conduction barely broadens the nominal absorption profile.

There is therefore a real inconsistency worth retaining in the model:

- Beer–Lambert alpha=0.7467 mm^-1 gives a one-sided half-maximum distance of 0.928 mm.
- The reported thermal FWHM is 2.8 mm. If it were interpreted as a one-sided
  exponential half-width, it would imply beta_eff=ln(2)/2.8=0.2476 mm^-1.

That one-sided interpretation leaves the far end of a 10 mm FBG warm and translates the whole
cold spectral edge, contrary to Fig. 5(a). A centered Gaussian with 2.8 mm FWHM improved the
envelope but generated unrealistically regular, deep short-wavelength fringes. It was therefore
replaced by an inverse fit of an asymmetric generalized-Gaussian profile:

    f(z) = exp[-(|z-z0|/w_left)^p_left],   z <= z0
           exp[-(|z-z0|/w_right)^p_right], z > z0.

Every forward evaluation is the coherent transfer-matrix model. The inverse objective uses
60 pm-smoothed spectral envelopes from 73, 131, 188, 244, 296, and 367 mW, plus a soft
2.8 +/- 0.35 mm camera-FWHM constraint. The recovered, physically oriented profile is:

| Recovered parameter | Value |
|---|---:|
| Peak position z0 | 2.066 mm |
| Total FWHM | 2.743 mm |
| Left half-width | 0.411 mm |
| Right half-width | 2.331 mm |
| Left shape order | 4.425 |
| Right shape order | 2.060 |

Power-only reflection cannot distinguish f(z) from f(L-z), so the optimizer's literal solution
centered at 7.934 mm was mirrored to put the sharp edge first and broad thermal tail along the
heater direction. The recovered profile reduces the smoothed-envelope RMSE from 2.12 dB for
the centered Gaussian to 0.96 dB (per-trace absolute-height offsets eliminated in both cases).
It also removes the unrealistically symmetric short-wavelength cavities.
An alternate differential-evolution seed returned z0=7.941 mm, FWHM=2.744 mm,
left/right half-widths 2.332/0.412 mm, and orders 2.069/4.50, confirming the same
sharp-edge/broad-tail solution within the screenshot-limited precision.

This FBG-contact temperature is not asserted to equal the bare-MDF Beer-Lambert temperature.
BN paste, the second fiber, longitudinal offset, and strain transfer can change the effective
profile sampled by the grating. `thermal_profile_models.png` retains the nominal Beer-Lambert
and exact-fin profiles for comparison.

## Optical model and assumptions to fit

The implementation divides the grating into locally uniform slices and multiplies the
coupled-mode transfer matrices. Local temperature maps to local Bragg wavelength by integrating
the paper's segmented sensitivities (10, 11.8, 13.3, 14.4, and 15.1 pm/K). The initial grating is
uniform-period and Gaussian-apodized, constrained from the black trace.

Commercial priors support this interpretation. ITF specifies >80% reflectivity, 0.3 +/- 0.1 nm
FWHM, and apodization with >15 dB sidelobe suppression for SMF-28-compatible sensing FBGs.
AtGrating specifies 10 mm sensing gratings with >=90% reflectivity, <=0.3 nm FWHM, and >=15 dB
sidelobe suppression. These are unusually close to the digitized black trace.

Parameters not reported by the paper, and therefore inferred/common-assumption values, are:

| Parameter | Default | Role / identifiability |
|---|---:|---|
| Effective index | 1.447 | Standard silica-fiber value; weakly affects detuning scale |
| Peak coupling kappa | 560 m^-1 | Fits ~0.30 nm FWHM; Delta-n_eff~2.76e-4 |
| Gaussian sigma/L | 0.30 | Gives modeled cold SLSR ~14.7 dB and endpoint amplitude 0.249 |
| Modeled cold reflectivity | 99.8% | Plausible high-reflector prior; absolute R is not identifiable from dBm screenshot |
| Hotspot full FWHM | 2.743 mm | Inverse fit with 2.8 mm camera prior |
| Hotspot shape | Strongly asymmetric | Sharp 0.411 mm half-width; broad 2.331 mm tail |
| Hotspot position | 2.066 mm or mirrored 7.934 mm | Power spectrum has axial-reversal ambiguity |
| OSA resolution | 0.020 nm | Common assumed value; linear-power Gaussian convolution |
| Absolute dBm scale | launched-equivalent -31.05 dBm plus -44 dBm floor | Plot-display nuisance parameter |

The heated curves show an additional saturating ~4.5 dB reduction of their cold-envelope peaks
that a lossless 1-D CMT model does not reproduce. For visual comparison the plotting script uses

    A_meas(P) = -4.50 [1-exp(-P/66.87 mW)] dB.

This fits the digitized peak-height reduction to about 0.2 dB RMS. It is explicitly a
measurement/coherence nuisance term—not optical absorption or claimed physical FBG loss.
Possible contributors are unresolved phase/strain nonuniformity, polarization/interrogator drift,
and screenshot digitization. `spectrum_at_power()` returns unscaled physical reflectivity.

Only power-reflection screenshots are available. Phase, raw OSA resolution/filtering, source
power at the interrogator, grating reflectivity, and exact alignment are missing, so the inverse
problem is non-unique. Raw spectra plus grating specs would permit a proper joint fit.

The inverse workflow and its outputs are in `fit_temperature_profile.py`,
`outputs/recovered_temperature_parameters.json`, `outputs/recovered_temperature_shape.csv`,
and `outputs/recovered_temperature_fit.png`. The colored screenshot traces used by the inverse
are exported by `digitize_spectra.py`.

## Literature basis

- The supplied paper: Ko, Kim, and Ahn, *Optics Express* 34, 28164–28176 (2026),
  DOI 10.1364/OE.605266.
- Won et al., "Distributed temperature measurement using a Fabry–Perot effect based chirped
  fiber Bragg grating," *Optics Communications* 265, 132–138 (2006): transfer-matrix fitting
  of nonuniform-temperature FBG spectra (the supplied paper's Ref. 32).
- Erdogan, "Fiber grating spectra," *Journal of Lightwave Technology* 15, 1277–1294 (1997):
  standard coupled-mode treatment of fiber gratings.
- ITF Technologies, `ITF_DataSheet_FBG-Sensors_v2`: >80% R, 0.3 +/- 0.1 nm FWHM,
  apodized, SLSR >15 dB.
- AtGrating, `apodized-fbg.pdf`: 10 mm, >=90% R, <=0.3 nm FWHM, SLSR >=15 dB.
