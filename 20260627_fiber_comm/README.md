# Reproducing normalized `C_m,n` coefficients

This workspace reproduces the structure of the normalized nonlinear perturbation coefficients shown in Figure 13.10 of the provided screenshot:

> Normalized `C_m,n` coefficients in dB for 3600 km of standard SMF.

The figure is reprinted from Cartledge et al., "Digital signal processing for fiber nonlinearities," *Optics Express* 25(3), 1916-1936 (2017), which cites Gao et al., "Reducing the complexity of perturbation based nonlinearity pre-compensation using symmetric EDC and pulse shaping," *Optics Express* 22(2), 1209-1219 (2014).

## Cached references

- `references/cartledge_2017_dsp_fiber_nonlinearities.pdf`
- `references/gao_2014_reducing_complexity_sedc_rrc.html`

## Model

The reproduced plot follows the coefficient definition summarized in Cartledge et al. Eq. (5)-(7):

```text
C_m,n proportional to integral_z integral_t
conj(u_z(t)) u_z(t-nT) u_z(t-mT) conj(u_z(t-(m+n)T)) dt dz
```

For the normalized figure, scalar prefactors such as nonlinear coefficient, launch power, and any z-independent normalization cancel in `20 log10(|C_m,n / C_0,0|)`.

Implemented assumptions:

- Standard SMF: `D = 17 ps/(nm km)` at `lambda = 1550 nm`.
- Total link length: `L = 3600 km`.
- Symmetric electronic dispersion compensation: integrate over `0 <= z <= L/2`.
- Pulse: frequency-domain root-raised-cosine response with rolloff `rho = 0.1`.
- Symbol rate default: `16 Gbaud`, consistent with a 128 Gb/s DP-16QAM example.
- Numerical result is clipped to `[-40, 0] dB` for plotting.

The source paper figure is a numerical calculation rather than a closed-form plot. The exact appearance depends on pulse normalization, time/frequency window size, z quadrature density, whether attenuation/span power evolution is included, and the precise transmitter/matched-filter convention. The current script is intended as a reproducible physics-based reconstruction, not a pixel-identical extraction.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/reproduce_cmn.py
```

Outputs:

- `outputs/cmn_3600km_interactive.html`
- `outputs/cmn_3600km_data.npz`

Useful faster test run:

```bash
.venv/bin/python src/reproduce_cmn.py --max-index 25 --fft-symbols 512 --z-steps 31
```

Higher-fidelity run:

```bash
.venv/bin/python src/reproduce_cmn.py --sps 12 --fft-symbols 2048 --z-steps 121
```

Regenerate only the HTML from cached data:

```bash
.venv/bin/python src/reproduce_cmn.py --from-npz outputs/cmn_3600km_data.npz
```

Visual comparison/evaluation against the cached Cartledge figure:

```bash
.venv/bin/python scripts/evaluate_visual.py
```

Outputs are written under `outputs/eval/`, including the source screenshot, generated screenshot, aligned plot crops, RGB absolute-difference heatmap, contact sheet, and `visual_eval_metrics.json`.

## TODO

- Compare against the original Gao et al. figure if a downloadable full PDF is obtained.
- Add optional fiber attenuation/span weighting if matching absolute coefficient conventions becomes important.
- Add image export for a static publication-style PNG.
- Benchmark vectorized or numba-accelerated overlap integration for larger grids.
