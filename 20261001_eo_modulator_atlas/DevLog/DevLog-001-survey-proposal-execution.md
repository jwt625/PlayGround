---
title: EO modulator atlas - pre-build survey, proposal, and agent execution
date: 2026-10-01
status: active
scope: record the initial survey, the full proposal, the frozen user decisions, and the parallel-agent execution log
authors: Claude (at Wentao Jiang's request)
---

# DevLog-001: Survey, proposal, execution

Historical execution record. For current categories, tranches, dependencies and
agent claims, start with [DevLog-002](DevLog-002-work-plan-and-ownership.md) and
[WORKBOARD](../WORKBOARD.md).

Companion to DevLog-000 (plan/decisions/TODO). This log captures the material that
led to DevLog-000: what existed before, the full structured proposal, the user's
answers that froze the scope, and how the parallel build actually ran. Reconstructed
2026-10-01 from the session transcript.

## 1. Starting point (survey, read-only)

Assets that existed before this project:

| Asset | What it provides | Gaps |
|---|---|---|
| `PlayGround/20260614_2d_eo_fem_solver` (GPL, JS core in `web/core`) | Browser 2D electrostatic + optical-mode solver; YAML config, tensor epsilon, scalar/vector FD modes, TFLN and BTO example YAMLs | No EO overlap / Vpi*L, no RLGC / Z0 / S21, no loaded-line model, non-conforming electrodes, vector modes unbenchmarked |
| Private analog InP MZM repo (no license) | Conceptual CL-TWE reference only | No license, not reusable code |
| `agentic-multiphysics-harness` | EO literature plan, DevLog-007, solver decks for Chen 2022, Lee 2026, Shen 2024 | All three decks report `fail` against paper gates |
| `agentic-multiphysics-harness` rules | Evidence ledger classes `paper_exact / figure_digitized / project_inference / unknown`; recorded pitfalls: differential-voltage factor of 2, fiber-to-fiber IL vs waveguide loss | Keep these rules |
| Private benchmark CSVs | Schema precedent, ~35 devices across BTO, InP, Si, Si-LN, TFLN, TFLT | Private source; facts must be re-sourced |
| `PlayGround/20260320_OFC/OFC_2026_high_speed_materials_papers.md` | ~16 high-speed modulator papers | OFC list only |
| Private browser-explorer project (Python ref + HTML + CSV + tests) | Pattern for a self-contained browser-run explorer | Private source; pattern only |
| `jwt625.github.io` blog | ~70 EO-related items across weekly posts, tweet-sourced; no carrier-depletion Si MZM, GaAs, graphene or mid-IR papers; Lee 2026 absent | Heading/citation mismatches (Rahman2025, SiGe-driver item, Valicourt DOI); numbers come from tweets/OCR and are never used as DB values, only as `discovered_via` pointers |
| Blog scraping infra | `_posts/scraping/` JSON, image OCR, CLIP index, `lookup-tweet-source` skill | `_posts/config.json` holds an API key + browser profile path; must never be copied into this project |

Toolchain at start: `uv 0.7.16`, `node 23.7.0`, `pnpm 10.12.4`; Crossref and arXiv APIs reachable.

## 2. Initial proposal (as presented)

Target deliverable, per the user's brief: (1) a tracked metrics+metadata table; (2) a
skill that distills a reference into DB rows and a sim config; (3) browser-runnable
simulation configs (no stored solver output); (4) data visualization.

Proposed tree (design; the shipped tree in DevLog-000 is the simplified form):

```
README.md, DevLog/DevLog-000-plan.md
data/
  schema/devices.schema.yaml    # columns, units, enums, conventions
  papers.csv                    # one row per publication
  devices.csv                   # one row per device / operating point
  organizations.csv             # university, company, national lab, group/PI, country, region
  foundries.csv                 # fab, wafer supplier, process (as disclosed)
  materials.csv                 # EO tensors, eps, n, sourced
  evidence/<paper_id>.yaml      # per value: locator (page/fig/table), basis, confidence
  sources.jsonl                 # url, doi, sha256, access date, license
sims/<paper_id>/config.yaml     # geometry, materials, targets, tolerances, provenance per parameter
app/                            # static SvelteKit: /explore, /table, /sim?config=
.claude/skills/eo-modulator-distill/
scripts/validate_db.py          # fails if a non-N/A value has no evidence entry
```

Core `devices.csv` column groups as proposed:

- Identity: `paper_id` (FirstAuthorYear), `device_id`, `device_class` (MZM, ring, EAM, phase shifter), `platform`, `electrode_type` (lumped, TWE, CL-TWE, segmented, plasmonic), `drive` (push-pull, differential, single-ended).
- Operating point: wavelength, length, temperature.
- Voltage: Vpi, Vpi*L (DC and RF with frequency), plus `vpi_convention` and a derived flag.
- Bandwidth: EO 3 dB and 6 dB, reference frequency (DC / 1 GHz / 10 GHz), measured-to frequency.
- Optical loss: on-chip IL, fiber-to-fiber IL, propagation loss, with an exclusion column.
- RF: loss in dB/cm at a stated frequency, Z0, n_RF, optical n_g.
- Basis: per-metric enum (measured, simulated, extracted, predicted).
- Geometry: film thickness, etch depth, rib width, electrode gap/width/thickness, buffer, substrate, cladding, cut.
- System: baud, format, line rate, energy per bit.
- Bias and power: power handling, bias stability.
- Metadata: DOI/arXiv, venue, year, source type, authors, group, company, university, country, region.
- Reproducibility grade: A = geometry and materials fully disclosed; B = needs figure digitization; C = metrics only. Sim configs only for A and B.

Skill `eo-modulator-distill` workflow: resolve DOI/arXiv and cache with sha256 (rate-limited, open access); extract text and figures; fill `devices.csv` + evidence sidecar with N/A where silent; grade reproducibility; emit `sims/<id>/config.yaml` with a provenance class per parameter; run headless against paper targets and record tolerances; hand to a fresh-context audit.

Sim runner: reuse the GPL `eo_fem` core client-side as a quasi-2D chain: 2D electrostatics -> optical mode -> EO overlap (Vpi*L) -> quasi-TEM RLGC (Z0, n_mu, loss) -> loaded-line ABCD -> EO S21 (velocity mismatch + loss). v1 targets dielectric TWE devices only; SOH/plasmonic/InP get circuit-level templates.

Visualization shortlist (Plotly, dark, sharp corners, log toggles, hover provenance): Vpi*L vs BW frontier by platform; IL vs Vpi*L; Vpi*L*alpha FOM; BW and baud vs year; length vs BW; platform range bands; country/organization views; disclosure-completeness heatmap; sim reproduction scorecard.

Phases: P0 schema + seed; P1 skill piloted on 3 diverse papers; P2 sim engine with analytic gates; P3 visualizations; P4 scale-up with one subagent per batch and audits.

## 3. Frozen user decisions (2026-10-01)

Presented as 9 questions; the user answered ("defaults accepted", with notes on 2, 3, 5):

1. Location/visibility: PlayGround dated folder now, public spin-off later. Public means no private-startup / private-repo / private-note references; facts from a private literature cache are re-sourced from primary papers.
2. Scope: Pockels/plasmonic/SOH/InP are core; carrier-depletion Si MZMs, EAMs and rings get rows. **User note:** focus less on InP and Si for sim (complex semiconductor physics), but still track them as rows. EO comb / photonic-computing papers excluded unless they report modulator metrics; DSP/SerDes out.
3. Row granularity: one row per device variant/operating point, with a paper row above it; the table is expandable. **User note:** show the most complete record or the best device from a multi-device paper, using a metric such as Vpi*L / IL_opt / IL_rf.
4. Seed strictness: verified-only; tweets as pointers only; paywalled papers become an exact-filename download list. **User note:** give me a list to download paywalled papers.
5. Provenance/rights: facts plus locators; PDFs and extracted text stay local. **User note:** extract text and figure images. (Later, 2026-10-01, the user revised the policy to commit the full cache including PDFs - see DevLog-000 progress and `.gitignore`.)
6. Sim v1: dielectric TWE chain only.
7. Engine path: copy and attribute the GPL `eo_fem` core; add browser WASM constrained triangulation for thin electrodes / sloped sidewalls.
8. Results policy: configs + paper targets + tolerances only; no solver outputs committed; headless pass/fail recorded in the DevLog.
9. Skill home: repo `.claude/skills/`, free of personal info.

Checked before P2: the harness DevLog recorded an earlier audit finding of a
differential-voltage factor-of-2 capacitance error in `eo_fem`; whether it was fixed
is still unverified.

## 4. Candidate compilation

Two subagents compiled candidates in parallel (Crossref + DataCite + OpenAlex; arXiv API
returned 429/503 to the seed agent, so arXiv metadata came from DataCite and arXiv
calls were spaced out; OFC 2026 codes resolve as `10.1364/OFC.2026.<code>`).

- Seed list: 134 unique rows (priority 40/80/14; platform split dominated by lithium_niobate
  then inp_mqw / silicon; `discovered_via` includes local corpus, OFC2026, Drive doc, tmp md, blog).
- Landmark list: 76 Crossref-verified rows (priority 33/40/3); 9 lithium_niobate rows are hybrid Si/SiN-LN.
- Merged and deduped: `data/candidates.csv` = 198 rows (priority 1: 65, 2: 116, 3: 17), after merging arXiv preprints with journal versions and resolving ID collisions.
- Private-corpus paths in the candidate CSV are written as `@corpus_A/...` / `@corpus_B/...` placeholders; the real absolute paths live only in a git-ignored private mapping file.
- Unresolved identities and mismatch flags are recorded in the staging reports (`candidates_seed_report.md`, `candidates_landmark_report.md`). 8 priority-1 paywalled papers are queued in `data/manual_downloads.md`.

## 5. Pilot distillation and schema v2

Three diverse pilots were distilled with the skill into `data/_staging/pilot_*`, then merged
into `data/` (3 papers, 7 devices, 8 organizations, validator 0 errors):

- chen2022 - TFLN capacitively loaded TWE MZM, APL Photonics, CC-BY-4.0. 3 device rows (5/7/10 mm), grade B, with the first `sims/chen2022/config.yaml` (loaded + unloaded cross-sections, provenance classes, 11 targets, missing list, limitations).
- kohli2025 - BTO plasmonic on SiN, Light: Science & Applications, CC-BY-4.0. 3 device rows (MZM, IQ, racetrack), grade C.
- ogiso2016 - InP n-i-p-n CL-TWE, Electronics Letters, publisher-copyright. 1 device row, grade C.

Pilot feedback drove schema v2: bound qualifiers (`gt`/`lt`/`approx`), a push-pull MZM Vpi
column, InP epitaxy / waveguide orientation / resonator / stack columns, `basis` values
`author_estimate` and `design_target`, `series_push_pull`, `er_type`, EO roll-off, and a
more permissive evidence check. Feedback files: `data/_staging/pilot_*/SKILL_FEEDBACK.md`
and `SPEC_PROPOSALS.md`.

## 6. Batch distillation: plan and interruption

- Batches `p1_01..p1_11`, 5 papers each (`data/_staging/batches/`), built by `scripts/make_batches.py`
  with `data/_staging/BATCH_INSTRUCTIONS.md` as the shared contract (per-paper Crossref identity,
  rate-limited local/arXiv fetch, skill + schema v2, sim config only for grade A/B dielectric TWE,
  dry-run merge to 0 conflicts / 0 validation errors, per-batch report).
- Six batch distillers (`p1_01..p1_06`) were launched in parallel alongside the engine and app builders.
- At 2026-10-01 17:44Z (~10:44 local) all agents were killed by an API session limit (HTTP 429),
  resetting 14:50 local. On-disk state confirms the interruption was early: only `p1_01/cr/` holds
  Crossref raw responses (5 JSON files); the other batch dirs and `evidence/` are empty. Batches
  p1_02..p1_11 did not run.
- The app and sim-engine agents also died at the same moment; their partial output is what is on disk.

## 7. Execution notes

- arXiv API throttled the seed agent (429/503); Crossref, DataCite and OpenAlex were used instead, all calls spaced >= 2 s one at a time.
- Every extraction used `--local-origin local_corpus`; no output file names the private source repository.
- One wrong local PDF was identified and excluded (a deep-learning paper mislabeled as a hybrid Si-LN MZM); the intended Valdez/Mookherjea paper was added as `valdez2022` from its real DOI.

## 8. Open items / next actions

- Resume batch distillation p1_01..p1_11 (rate-limited; paywalled items await `references/_inbox/` downloads).
- Finish the browser sim engine and headless gates (verify the factor-of-2 capacitance finding first).
- Build the SvelteKit app (`/table`, `/explore`, `/sim`).
- Add the fresh-context audit pass and record corrections in DevLog-000.
