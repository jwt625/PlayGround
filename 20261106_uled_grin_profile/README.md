# GRIN profile optimization for blue uLED (and uVCSEL) multimode fiber

Optimize an axially symmetric, monotonic GeO2-SiO2 graded-index profile to minimize overfilled-launch (OFL) modal
dispersion for a 430 nm uLED link (Avicena-style slow-and-wide), and the same for an 850 nm uVCSEL, with smooth and
discrete-layer (staircase) profiles. Motivation and source notes: `references/md/avicena-deep-dive-notes.md`.
Plan, decisions, progress: `DevLog/DevLog-000-grin-profile-optimization-plan.md`.

## Model
- Scalar LP modes from a radial finite-difference eigen-solve per azimuthal order (no WKB), all guided modes.
- Group delay per mode by Hellmann-Feynman with local material group index (Sellmeier: SiO2 Malitson 1965,
  GeO2 Fleming 1984, linear mole-fraction mixing).
- Objective: OFL RMS intermodal delay (equal power per guided mode), averaged over a Gaussian source spectrum.
  Strict = nominal design; robust = sqrt(mean sigma^2) over peak-index (+-2%) and VCSEL wavelength (+-10 nm) ensemble.
- Constraints (config/study.yaml): A 50 um / NA 0.20, B 80 um / NA 0.28 (shape free, coupling floor = etendue of the
  alpha optimum), C Pareto over core diameter and NA (alpha profiles).
- Parameterizations: alpha; Bernstein monotone (continuous, or with an edge step); N-layer staircase (free radii and
  levels), N = 2..16.

## Layout
- `grinopt/` material, profiles, solver, evaluate
- `scripts/run_study.py` batch (stages 1, 2, 1r, 2r, 3; resumable, skip-existing, JSONL logs in `logs/`)
- `scripts/make_figures.py` figures (`outputs/figures/`) and `outputs/summary.md`
- `scripts/make_report.py` interactive report `outputs/report/grin_report.html` (template `scripts/report_template.html`)
- `scripts/validate_alpha_opt.py` nondispersive check vs Olshansky-Keck alpha_opt = 2 - 12 Delta / 5
- `tests/` analytic checks (step-index Bessel modes, parabolic oscillator, Hellmann-Feynman vs finite difference)

## Run
```
uv run pytest -q
uv run python scripts/run_study.py --stage all      # long; run in tmux
uv run python scripts/make_figures.py
uv run python scripts/make_report.py
```

## Current state
2026-11-07: study running (tmux session `grin-opt`). Results and conclusions to be added when stage 3 completes.
