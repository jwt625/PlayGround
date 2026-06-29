# Devlog

## 2026-06-27

Goal: research and reproduce the normalized `C_m,n` coefficient figure for 3600 km standard SMF with Python, keeping references cached and producing an interactive HTML plot.

Progress:

- Created project structure: `src/`, `references/`, `outputs/`, `scripts/`.
- Cached Cartledge et al. 2017 PDF locally.
- Cached Optica landing/abstract page for Gao et al. 2014 locally.
- Implemented `src/reproduce_cmn.py`:
  - RRC frequency-domain pulse generator.
  - Chromatic dispersion propagation.
  - Direct four-pulse overlap integral over `m,n,z`.
  - Normalization to `C_0,0`.
  - Plotly contour output with an OSA-like stepped color scale.
  - Cached-data HTML regeneration via `--from-npz`.
- Added `scripts/evaluate_visual.py` for source/generated screenshot extraction, plot-panel alignment, RGB metrics, and diff artifacts.

Notes:

- Cartledge et al. Eq. (5)-(7) gives the coefficient integral used here.
- Figure 13.10 / the screenshot states the plotted coefficients are normalized dB values for 3600 km standard SMF.
- The plot is insensitive to absolute nonlinear coefficient after normalization, but sensitive to numerical pulse and filtering conventions.
- The default symbol rate is set to 16 Gbaud based on the 128 Gb/s DP-16QAM example described near the cited figure lineage.

Open TODO:

- Verify the exact Gao et al. simulation parameters from the full paper PDF if available.
- Tune sampling/z quadrature for visual match versus runtime.
- Add a convergence table for `sps`, `fft_symbols`, and `z_steps`.

Visual eval result:

- Ran `scripts/evaluate_visual.py` against Cartledge Fig. 1 on PDF page 8.
- The initial eval only resized/flipped/transposed the crop. A second-stage registration was added to optimize interpolated `tx`, `ty`, `sx`, and `sy`.
- Full black-panel metrics are misleading because the black background dominates:
  - Resize-only RGB MAE full panel: `16.32`
  - Resize-only RGB RMSE full panel: `55.46`
  - Resize-only SSIM full RGB: `0.854`
- Signal-region metrics show the actual mismatch:
  - Resize-only signal pixels: `15.50%` of plot panel
  - Resize-only RGB MAE signal: `104.53`
  - Resize-only RGB RMSE signal: `140.79`
- Best registered transform:
  - Axis transform trial: `rot180`
  - `tx = 8.86 px`, `ty = -9.54 px`, `sx = 0.839`, `sy = 1.038`
  - Registered RGB MAE full panel: `9.62`
  - Registered RGB RMSE full panel: `39.63`
  - Registered SSIM full RGB: `0.883`
  - Registered signal pixels: `13.06%` of plot panel
  - Registered RGB MAE signal: `73.34`
  - Registered RGB RMSE signal: `109.64`
- Visual diagnosis: generated plot has smooth monotonic lobes, while the source figure has strong oscillatory/null structure. The current formula/implementation is therefore not a faithful reproduction.
