# DevLog-000: GRIN profile optimization for a blue uLED multimode link

| Field | Value |
|---|---|
| Date | 2026-11-06 |
| Status | Plan written, implementation in progress |
| Scope | Optimize an axially symmetric, monotonic GRIN index profile to minimize modal dispersion at 430 nm |
| Authors | Claude Code, at Wentao Jiang's request |

## Motivation
Irrational Analysis, "Avicena Deep-Dive" (substack, read 2026-11-06; notes in `references/md/avicena-deep-dive-notes.md`).
Avicena's blue uLED slow-and-wide links use multimode fiber bundles. The article's stated key problem:
"maximize coupling efficiency while minimizing modal dispersion". OM3 is a smooth, many-layer, expensive
GRIN profile optimized for 850 nm; Avicena runs at 430 nm (their slides p.70, p.77) and builds cheaper custom profiles.

## Objective (agreed 2026-11-06)
Minimize the overfilled-launch (OFL) RMS intermodal group-delay spread

  sigma_modal = sqrt( sum_lambda S(lambda) * sigma_m^2(lambda) ),
  sigma_m^2(lambda) = sum_modes w_m (tau_m - <tau>)^2 / sum w_m

- tau_m: group delay per unit length of scalar LP_lm mode m at wavelength lambda, from a numerical radial mode solve
  including material dispersion of the local GeO2-SiO2 composition (no WKB).
- w_m: equal power per guided mode (OFL, LED-like); degeneracy 1 for l = 0, 2 for l > 0.
- S(lambda): Gaussian LED spectrum, center 430 nm, FWHM config parameter (default 20 nm, assumption).
- Design variable: GeO2 mole fraction profile x(r), monotonic non-increasing from x0 at r = 0 to 0 (pure silica
  cladding) at r = a; axially symmetric. No trench (would violate monotonicity).
- Chromatic dispersion is reported (total RMS incl. intramodal), not optimized.

## Decisions (his answers, 2026-11-06)
1. Wavelength: 430 nm (Avicena slide), not 480 nm. LED FWHM is a config parameter.
2. Material: GeO2-doped silica core, pure silica cladding.
3. Constraints: all three configurations, toggleable in config:
   - A `fixed_om3`: core 50 um, NA 0.20 (OM3 class, Table 1 of Photonics 13, 408 shown on p.85). Shape free.
   - B `fixed_large`: core 80 um, NA 0.28 (OM1-class NA per same table). Shape free.
   - C `pareto`: core radius and NA free within bounds; trade sigma_modal against fiber etendue (coupling proxy).
4. Metric: RMS delay, OFL, spectrally averaged; also report max DMD, OFL -3 dB bandwidth at L, mode count.
5. (added 2026-11-06, mid-session) Also run the same optimization for uVCSEL sources: default 850 nm
   (the article references AMS OSRAM 850 nm uVCSEL; 1060 nm noted as alternative), narrow spectrum
   (FWHM config, default 1 nm, assumption). Same OFL metric for comparability (conservative for a VCSEL, whose
   real launch is restricted); restricted launch left as an open item.
6. (added 2026-11-06, mid-session) Discrete-layer MMF: monotonic staircase with N layers (level and radius of each
   layer free, peak level fixed at x0), optimized for N in a sweep, to show sigma_modal vs layer count
   (cost proxy: preform layer count). Cell-averaged n^2 on the FD grid so the objective is continuous in radii.

## Physics model
- Scalar weakly guiding wave equation, per azimuthal order l:
  psi'' + psi'/r - l^2 psi / r^2 + k^2 n^2(r, lambda) psi = beta^2 psi.
- Radial finite differences on a cell-centered grid, Dirichlet at r = R (cladding margin), symmetric tridiagonal
  eigenproblem (scipy `eigh_tridiagonal`), guided modes beta > k n_clad.
- Group delay exact for the discrete problem by Hellmann-Feynman:
  d(beta^2)/dk = <psi| 2 k n(r) N(r) |psi>, N = n - lambda dn/dlambda (local material group index),
  tau / L = (1 / c) d beta / d k = (1 / c) (1 / (2 beta)) d(beta^2)/dk.
- Material: Sellmeier with linear mole-fraction interpolation of oscillator strengths and resonance wavelengths
  between SiO2 (Malitson 1965) and GeO2 (Fleming 1984), as implemented in `ofiber.refraction.doped_glass`.
  Fleming GeO2 data range 0.36-4.3 um covers 430 nm. Mixing rule validity at 430 nm is an assumption.
- Coupling proxy for C: fiber etendue G_f = integral pi NA_loc(r)^2 2 pi r dr, NA_loc^2 = n(r)^2 - n_clad^2
  (proportional to guided mode count). Coupled fraction from an etendue-limited source ~ min(1, G_f / G_s).

7. (2026-11-06, agent decision, found in pilot) Coupling floor. Fixed core + peak NA does not pin coupling: a
   staircase optimizer collapsed the guiding region to ~1 mode and reached sigma = 0. Free-form and staircase designs
   must keep etendue >= (1 - 1%) of the stage-1 alpha optimum of the same case; penalty 100 ps/m per unit fractional
   deficit, >5% deficit rejected. Alpha scans are unconstrained (they define the reference).
9. (his answer, 2026-11-07) Add a robust objective alongside strict: sqrt(mean sigma^2) over an ensemble of
   peak index scale x0 x {0.98, 1.00, 1.02} and, for the VCSEL, center wavelength {-10, 0, +10} nm (uLED: spectrum
   already averaged). Stages 1r/2r; alpha bounds widened to [1.80, 2.60] for the robust stage (strict stage 1 used
   [1.80, 2.30]; several strict optima sit on that bound). Strict and robust reported side by side; final metrics
   include the robust sigma for every design.
8. (2026-11-06, agent decision) Metric kept strict: all guided modes, lossless, OFL. Near-cutoff mode groups dominate
   sigma at these V numbers (see validation). Diagnostic sigma over inner 80% mode groups is reported alongside.

## Parameterization
- Baseline: alpha profile x(r) = x0 (1 - (r/a)^alpha).
- Free-form: x(r) = x0 (1 - F(r/a)), F = Bernstein polynomial of degree N with monotone coefficients
  0 = c_0 <= c_1 <= ... <= c_N = 1 (cumulative softmax of N unconstrained parameters). Monotone by construction.
  Variant `bernsteinstep`: c_N < 1 free, i.e. a monotone step down to cladding at r = a.
- Staircase: N layers, free radii and levels (cumulative softmax), first level = x0.
- Optimizer: per design 3 independent starts (seed 1 at the alpha optimum, others perturbed); each start runs
  CMA-ES (40% of budget) -> Powell (30%) -> Nelder-Mead (rest). Budget 3000 evals (Bernstein), 120 x n_params
  (staircase). Best seed is kept; seed spread is reported as optimizer uncertainty.

## Validation plan
- Sellmeier: fused silica n(589.3 nm) = 1.4584 (Malitson).
- Solver vs analytic step-index LP modes (Bessel characteristic equation).
- Solver vs analytic infinite parabolic profile eigenvalues.
- Without material dispersion, numerical optimum alpha vs alpha_opt = 2 - 12 Delta / 5 (Olshansky & Keck 1976, as
  quoted in US4260221, eq. 3). Full formula with profile dispersion (y parameter) not yet verified from primary source.
- Grid convergence (dr, cladding margin).

## TODO
- [x] references/md notes from the article
- [x] material.py (Sellmeier, dn/dlambda, group index, x0 from NA)
- [x] solver.py (radial FD, all guided LP modes, HF group delay; stemr driver)
- [x] evaluate.py (OFL RMS, max DMD, -3 dB bandwidth, etendue)
- [x] profiles.py (alpha, monotone Bernstein, staircase)
- [x] tests (analytic checks) - 5 pass
- [x] validation: nondispersive alpha_opt vs Olshansky-Keck
- [ ] scripts: alpha sweep, free-form optimization per config, OM3@850 design evaluated at 430, step-index baseline
- [ ] config C Pareto sweep
- [ ] uVCSEL source case (850 nm) for configs A/B, and cross-evaluation (uLED-optimized fiber at 850 nm and vice versa)
- [ ] staircase parameterization + sigma vs N layers sweep, for uLED and uVCSEL
- [ ] robust-objective stages 1r/2r + final stage 3 (chained in tmux window `robust`, logs/chain_robust.sh)
- [x] plots: static (scripts/make_figures.py) + interactive report (scripts/make_report.py ->
  outputs/report/grin_report.html), first version on partial data; auto-regenerated by logs/chain_report.sh
- [ ] README, fresh-context audit

## Progress log
- 2026-11-06: Read article, rendered fiber section slides (pp. 67-87). Clarifications answered. Folder + plan created.
- 2026-11-06 23:30: Solver + tests pass: silica n_D = 1.45840; step-index LP b within 2e-3 of Bessel roots; parabolic
  eigenvalues within 1% of oscillator spacing; HF group index vs finite-difference d beta/dk within 1e-6.
- 2026-11-06 23:35: Grid convergence (alpha 2.05, 850 nm, 50 um/NA 0.2): sigma 0.7701 / 0.7726 / 0.7721 for
  (dr 0.05, margin 15), (dr 0.025, margin 15), (dr 0.05, margin 50) um. dr = 0.08 not converged for 80 um/NA 0.28
  (27% off) -> dr = 0.05 kept.
- 2026-11-06 23:42: Pilot (uvcsel A) exposed the mode-collapse loophole -> coupling floor (decision 7). Pilot results
  discarded. Second pilot: CMA alone stalls / varies between runs -> optimizer chain + 3 restarts.
  Continuous Bernstein stalls near alpha (0.42 vs 0.586 ps/m); edge-step variant 0.135; staircase N=12 0.048-0.159
  depending on run. Edge step confines outer mode groups.
- 2026-11-06 23:55: Speed: stemr driver + center-node reuse -> 3x (uLED 80 um: 3.1 -> 1.05 s per 3-node eval).
- 2026-11-06 23:56: Full study launched in tmux `grin-opt`, 8 workers; logs/run_study_20261106_235626.jsonl.
- 2026-11-07 00:05: Validation (outputs/validation/alpha_opt_*.json), no material dispersion, inner 80% mode groups:
  alpha_num 1.9827 (V 38), 1.9803 (V 75), 1.9797 (V 121) vs alpha_OK 1.976 (Delta 1%); 1.9904 vs 1.988 (Delta 0.5%).
  With all guided modes the optimum moves to 2.0-2.2 (near-cutoff groups), i.e. WKB theory does not describe the
  strict OFL optimum at these V. The minimum-rms WKB coefficient N1 Delta^2/(20 sqrt 3 c) is from memory and was not
  found in a primary source; stored as `sigma_wkb_unverified_ps_per_m`, not used.
- 2026-11-07 00:10: Stage 1 done (64 jobs). Finding: strict-OFL sigma(alpha) is a sawtooth. sigma falls smoothly
  with alpha until the next mode group crosses cutoff (alpha up -> etendue and mode count up), then jumps. uLED A:
  2.30: 0.88 (711 modes), 2.36: 0.66 (719), 2.38: 1.24 (725), 2.52: 0.92 (749). Several stage-1 optima sit on the
  scan bound 2.30. The strict-OFL optimum is therefore a just-before-cutoff design, not robust to ~1% Delta or a few nm
  of wavelength. The classical alpha_opt only describes the inner mode groups. Question raised to Wentao: keep strict
  metric only, or add a robust (perturbation-ensemble) objective. Stage 2 (strict) keeps running meanwhile.
- 2026-11-07 00:15: Robust objective implemented (decision 9). Smoke test uvcsel A: robust alpha_opt 2.020,
  sigma_robust 1.19 ps/m (strict optimum 2.1535 sits just before a cutoff jump). Fixed stage-3 grouping bug
  (`id.split("__s")` also split `__stair`; now regex `__s\d+$`). The running strict process still holds the old stage-3
  code, so logs/chain_robust.sh waits for it, deletes its final__*.json, then runs stages 1r, 2r, 3.
- 2026-11-07 00:30: Interactive report (his request): scripts/make_report.py + scripts/report_template.html ->
  outputs/report/grin_report.html. Plotly 3.0.1 + MathJax 3.2.2 (cdnjs), data embedded, dark default with sun/moon
  toggle, sharp corners, palette from the dataviz skill (validated dark + light; light contrast WARN relieved by the
  table), pinned tooltips (click point, click blank to clear), log/lin toggle, small multiples for all cases, sortable
  table, method + symbol table, validation. sigma vs wavelength sampled at 1 nm (cache
  outputs/report/cache_sigma_vs_lambda.json) with optional Gaussian spectral averaging. Verified with headless system
  Chrome (Playwright, temp profile): 15 plots, pin create/clear, lin/log, theme, source switch, sort; 0 console errors.
  tmux window `report` regenerates figures + report after the robust chain finishes.
