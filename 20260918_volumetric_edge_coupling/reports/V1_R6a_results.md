# V0 / V1 / R6a results

All numbers at 1550 nm, constant indices, nominal 262 nm / 53.50 deg, res 25/um
unless stated. No thickness or angle was changed.

## V0 — figure and provenance repairs

- `slab_profile` (`src/modes.py`) now returns a monotonically decreasing
  coordinate so the top cladding, core and bottom cladding plot as one
  contiguous line (the spurious connecting diagonals are gone).
- `fig_mode_profiles.png` shades the Si core as a horizontal band in y (was a
  vertical span in |field|) and labels the axes as normalised |Ez| and |Hz|.
- Figures carry the config hash and branch; `reports/figs/provenance.md`
  records timestamp / config hash / branch / mesh / runtime / solver build for
  each generation. The Poynting figure is now saved per branch
  (`fig_nominal_poynting_hz.png`), fixing the earlier Ez-vs-Hz ambiguity.
- Still open: exported **loaded**-mode fields (analogue of the isolated
  profiles) and a sampled-dielectric zoom with source, both port apertures and
  the inner PML edge.

## V1 — port validation and same-aperture accounting

Method check on an effectively isolated straight guide (air/Si 262 nm/oxide,
single bound mode, EigenModeSource):

| quantity | value |
|---|---|
| native signed flux | 400.3634 |
| modal forward power | 400.3606 |
| native / modal | **1.0000** |
| backward fraction | 5.6e-6 |

So native flux and modal power agree to <0.01 % on the same aperture for a
clean single-mode port. The method is trustworthy.

Applied to the nominal coupler output, Hz branch, on the identical aperture:

| quantity | value |
|---|---|
| native signed flux | 103.88 |
| sum forward (bands 1-6) | 100.23 |
| sum backward (bands 1-6) | 5.28 |
| modal net | 94.95 |
| residual (native - modal net)/native | 8.6 % |
| band 1 (n_eff 2.298) forward | 60.32 |
| band 2 (n_eff 1.206) forward | 33.84 (cladding/radiation) |
| Pinc (homogeneous reference) | ~8573 |

Selected-band forward guide content is 60.32/8573 = **0.70 %**; the earlier
2.3 % custom-line flux was a different aperture and is not a controlled
comparison. Band 1 is the guide-like loaded mode; band 2 carries substantial
radiation. Mode identity remains open (band selection vs field overlap).

## R6a — reciprocal Hz guide-mode launch

Launching the loaded band-1 Hz mode from the output toward the interaction
(`scripts/r6_reciprocal.py`, res 25):

| quantity | value |
|---|---|
| guide reflection to +d | 0.50 % |
| emitted -x power at the Si reference plane (x=-12 um) | 18.59 |
| fraction of native launched flux | 18.6 % |
| overlap with the time-reversed 5 um Gaussian (profile) | **0.275** |

The reverse field map (`fig_r6_reverse_hz.png`) shows the launched guide mode
turning into a **broad diverging fan** in silicon, not a collimated Gaussian
retracing the incident path. Only ~28 % overlap amplitude with the target
mode means most emitted power leaves at angles the fibre mode cannot accept.

## Interim reading (not a validated ceiling)

Forward (0.70 % guide channel) and reverse (broad emission, ~28 % mode
overlap) point the same way: in this declared reconstruction the guide mode
and the 5 um Gaussian are substantially mismatched. The reverse measurement is
preliminary: the native launched flux includes radiation bands, the reference
plane can capture near-field, and direction separation is not yet complete, so
the reciprocity residual is not yet established at the R6 acceptance threshold.
The forward/reverse pair has not been shown to agree within
max(1e-4, 5 %). Mode identity, leakage and accumulated phase remain open. This
does not yet establish a validated sub-1 % optical limit, and it does not
disprove the paper.

## Next

1. Complete R6a: launch a single identified guide channel, separate outgoing
   directions, construct the reciprocal Gaussian from the calibrated forward
   illumination (phase curvature included), and close the energy budget.
2. R6b: two finer meshes (50, 75/um) and termination/aperture convergence.
3. Then select compact mismatch study vs a narrowly targeted calibration.
