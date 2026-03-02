# DevLog-000: Starlink Swarm Beam Estimation

## Date
- February 16, 2026 (US)

## Project Goal / Context
Build a meme-style but technically grounded visualization that estimates the best-case focused beam spot on Earth from a hypothetical "ultimate Starlink swarm" where satellites inside field-of-view (FOV) transmit coherently toward one ground target.

This is a thought experiment for visualization and estimation, not a claim about real deployed capability.

## What This Model Should Output
- Estimated number of simultaneously visible satellites (`N_visible`) under elevation mask.
- Effective synthetic aperture scale from satellite geometry (`D_eff`).
- Estimated ground spot scale (`spot_diameter`) from wavelength + aperture.
- Coherence penalty (phase/timing/atmosphere) reducing ideal gain.
- Sidelobe risk indicator for sparse/non-uniform arrays.

## Plan
1. Lock modeling assumptions and operating scenarios (conservative / aggressive / meme-max).
2. Build geometry module: visible count vs elevation mask and shell altitude mix.
3. Build beam module: spot estimate from `lambda`, slant range, and `D_eff`.
4. Add coherence-loss module from RMS phase error.
5. Add visualization sliders in `escalator.html`/`escalator.js` with clear disclaimers.
6. Validate with sanity checks and publish assumption table in UI.

## Assumptions Needed (to start)
1. **Frequency band (`f`)**
- Candidate bands used by Starlink filings: Ku/Ka ranges.
- Wavelength drives theoretical spot width directly.

2. **Constellation snapshot**
- Use either current on-orbit TLE snapshot or hypothetical full Gen2 shell deployment.
- Need epoch date attached to every estimate.

3. **Elevation mask (`el_min`)**
- Practical values: 10 to 30 degrees.
- Lower mask increases `N_visible` and aperture span, but worsens atmosphere/slant path.

4. **Coherence model**
- Ideal: perfect phase-lock and delay calibration across satellites.
- Realistic: add RMS phase error (`sigma_phi`) and compute gain loss `exp(-sigma_phi^2)`.

5. **Satellite weighting/power**
- Equal amplitude per satellite for first pass.
- Optional tapering later for sidelobe suppression.

6. **Propagation/environment**
- Start with free-space model.
- Add atmospheric/rain penalty toggle for Ku/Ka.

7. **Target metric definitions**
- Spot metric: first-null diameter or -3 dB width (choose one and stay consistent).
- Gain metric: ideal coherent (`~N^2`) vs incoherent (`~N`) reference.

8. **Array geometry simplification**
- First pass: use effective aperture diameter from visible footprint.
- Later: sparse array simulation for sidelobes/grating effects.

## Research Results (current pass)

### A) Live constellation data (current geometry anchor)
- Source: CelesTrak Starlink TLE feed
  - https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle
- Snapshot pulled Feb 16, 2026 (TLE epoch day around 26046 = Feb 15, 2026 UTC).
- Parsed objects in file: ~9,558 Starlink entries.
- Approx altitude clusters derived from mean motion (km):
  - ~480 km (largest cluster)
  - ~540 to 560 km (large)
  - ~570 km, ~360 km (smaller but significant)

### B) Estimated visible satellites from one ground point
Computed from Earth line-of-sight geometry over the TLE-derived altitude mix (global isotropic approximation):
- `el >= 30 deg`: ~31 satellites visible on average
- `el >= 25 deg`: ~44
- `el >= 20 deg`: ~63
- `el >= 10 deg`: ~141
- `el >= 0 deg`: ~344

Interpretation: for coherent operation, usable `N` in realistic elevation masks is likely in the tens, not thousands, at any instant.

### C) Geometric footprint scale (example shell)
For altitude ~550 km, max FOV footprint diameter around a user location:
- `el >= 30 deg`: ~1,587 km
- `el >= 25 deg`: ~1,881 km
- `el >= 20 deg`: ~2,250 km
- `el >= 10 deg`: ~3,330 km

This footprint scale is a first proxy for synthetic baseline / effective aperture extent.

### D) Frequency/band references for parameter sweeps
- Summary of SpaceX Gen2 filing values (secondary source summary):
  - https://www.satcom.guru/2021/08/spacex-files-for-second-generation.html
- Bands listed there align with common Starlink Ku/Ka usage ranges and can seed sweep presets.

### E) Beamforming / coherence equations to use
- Interferometric baseline-limited resolution concept (`theta ~ lambda / B`) reference:
  - https://public.nrao.edu/telescopes/alma/interferometry/
- Phased-array coherence loss vs RMS phase error reference:
  - https://www.analog.com/en/resources/analog-dialogue/articles/phased-array-antenna-patterns-part3.html

### F) Atmospheric delay/attenuation references
- Tropospheric delay background and typical magnitudes:
  - https://gssc.esa.int/navipedia/index.php/Tropospheric_Delay
- ITU-R propagation methodology reference for satcom availability/rain:
  - https://www.itu.int/rec/R-REC-P.618

## Modeling Baseline for Implementation (v0)
Use three scenarios:
1. **Conservative**
- `f = 12 GHz`, `el_min = 30 deg`, `N = 30`, moderate phase error.

2. **Aggressive**
- `f = 20 GHz`, `el_min = 20 deg`, `N = 60`, tighter phase control.

3. **Meme-Max**
- `f = 30 GHz`, `el_min = 10 deg`, `N = 140`, near-ideal coherence.

## Known Gaps / Next Research Tasks
1. Pull a directly accessible FCC/SpaceX primary document mirror for exact Gen2 shell and power table citations in this repo.
2. Add explicit timing/path-error thresholds (picoseconds / millimeters) per frequency preset.
3. Replace isotropic visible-count approximation with latitude-aware Monte Carlo from real TLE propagation.
4. Add sparse array sidelobe simulation (2D pattern heatmap) instead of only spot-size scalar.

## Notes
- This DevLog intentionally separates "physics upper bound" from "real operational Starlink capability".
- Final UI should label outputs as **hypothetical coherent swarm estimate**.
