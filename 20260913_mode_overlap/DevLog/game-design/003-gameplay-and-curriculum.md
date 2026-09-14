# 003 — Gameplay and scoring decisions

Status: actionable design, not implemented. Date: 2026-09-13. Depends on K002 eligibility.

First route offers a 20-question speedrun immediately. Setup selects answer mode and difficulty; each question shows geometry A/B and mode A/B with µm axes, wavelength, polarization and approximation label. Input → submit → reveal → next. Timer measures visible question-to-submit time only; feedback/background time excluded. No automatic timed advance; keyboard Enter submits/continues, arrows adjust slider.

Continuous answer: ordinary η slider plus numeric input and fine controls (0.001; optional 0.0001 near one). Always show guessed η and its dB conversion. Display exact η=0 as infinite mismatch loss. For finite scoring only use ε=1e−6 (60 dB floor); do not substitute that floor for the physical answer.

Proposed transparent numerical score: `100 exp(−|Lguess−Ltrue|/3)`, no speed multiplier in first release. Store raw signed η error, absolute η error and uncapped floor-defined dB error. Accuracy is primary; report time separately. Tune the 3 dB scale using pilot play, not physics assertions. Interval mode scores distance to the selected interval (zero if contained) under the same dB metric; report interval calibration separately from point-prediction metrics. Ranking uses normalized pairwise correctness; exclude true differences <0.005 or accept ties. Scores from different answer modes are separate leaderboards/profiles.

Intervals: [0,.1), [.1,.3), [.3,.6), [.6,.8), [.8,.95), [.95,1]; only last upper edge inclusive. Outcome balancing has two additional subdivisions above .95 (see G008). Do not reveal the target bucket before answering.

Reveal: guess, η, mode-mismatch loss, signed error, one dominant-effect sentence, optional expanded explanation/integrand. Exactly zero overlap is allowed for analytic polarization/parity. Never call mismatch loss exact interface insertion loss.

Results: score, MAE, median absolute η error, RMSE, mean dB error, response-time distribution, category breakdown, signed bias by η bucket. Avoid confident weakness claims with fewer than five samples in a category. Persist locally with reset/export; schema versions separate scores across rule changes.

Difficulty 1: one Gaussian perturbation; 2: two effects/fiber-size comparison and validated simple guides; 3: asymmetry/SSC/thin-vs-thick SiN; 4: heterogeneous/vector/cutoff; 5: phase/parity/anisotropy/multimode. Campaign chapters follow original §34. Raw Si↔SMF is a useful rare low-overlap anchor, not the entire fiber-chip curriculum.

- [ ] G3.1 Pilot slider precision and score scale with 20 fixed analytic cases.
- [ ] G3.2 Verify identical-intensity phase questions offer signed/phase views before submit.
- [ ] G3.3 Calibrate difficulty from pilot errors; keep performance claims tentative.
- [x] G3.4 Every mode A/B question (Gaussian, Gaussian variant, waveguide) is presented as a single toggling/flashing panel by default, not side-by-side; adjustable flash rate, manual toggle and Space, shared extent and color scale. New question types must ship toggling (UI4.8).
