---
title: EO modulator atlas - plan, decisions, TODO
date: 2026-10-01
status: active
scope: database + distillation skill + browser simulation configs + visualization
authors: Claude (at Wentao Jiang's request)
---

# DevLog-000: EO modulator atlas

## Goal

A tracked library of academic EO modulator demonstrations with (1) a core-metrics database, (2) a skill that distills a reference into DB rows and a simulation config, (3) browser-runnable simulation configs (configs only, no stored solver output), (4) data visualization.

## Decisions (user answers, 2026-10-01)

| # | Decision |
|---|---|
| 1 | Dated PlayGround folder now, public spin-off later. Public means no private-startup / private-repo / private-note references; facts sourced from a private literature cache are re-sourced from the primary papers. |
| 2 | Scope: Pockels / EO-effect devices core (TFLN, TFLT, BTO, SOH, polymer, plasmonic, hybrid Si/SiN-LN/LT, InP, AlGaAs). Carrier-depletion Si, EAM, rings tracked as rows. Simulation effort concentrates on dielectric TWE devices; InP/Si get rows but low sim priority (complex semiconductor physics). EO comb / photonic-computing papers only if they report modulator metrics. DSP/SerDes items in the source doc are out of scope. |
| 3 | Table rows are expandable: one paper row, expandable to its devices. The collapsed row shows the representative device = most complete record, ties broken by a figure of merit (see Representative device). |
| 4 | Paywalled papers: produce a download list with exact target filenames; user drops them into a folder; ingest afterwards. |
| 5 | Extract text AND figure images from each source. Applied rule: full extraction (PDF, text.md, figure PNGs) lives in the git-ignored `references/` cache for every source. Only sources with an open license that permits redistribution (CC-BY family, recorded per source) have text + figures promoted to a tracked folder; all others are tracked as facts + page/figure locators only. Changing this is a one-line policy switch in `scripts/promote_open_extracts.py`. |
| 6 | Sim chain v1: 2D electrostatics -> optical mode -> EO overlap (VpiL) -> quasi-TEM RLGC (Z0, n_mu, loss) -> loaded-line ABCD -> EO S21. Dielectric TWE only. |
| 7 | Reuse and attribute the 20260614 `eo_fem` JS core (his own GPL code); add conformal polygon meshing for thin electrodes / sloped sidewalls. |
| 8 | Sim results are not committed. Configs + paper-reported targets + tolerances only. Headless verification runs locally; pass/fail is recorded in DevLogs, not as data files. |
| 9 | Skill lives in repo `.claude/skills/eo-modulator-distill/`, free of personal info. |

## Architecture

```
data/
  schema/devices.schema.yaml    column definitions, units, enums, conventions
  candidates.csv                discovered papers before verification
  papers.csv                    one row per publication (metadata + provenance)
  devices.csv                   one row per device / operating point (core metrics)
  organizations.csv             normalized org names -> type, country, region
  evidence/<paper_id>.yaml      per value: locator (page/fig/table), basis, extraction note
  sources.jsonl                 url, doi, sha256, retrieved_on, license, rights decision
references/<paper_id>/          ignored local cache (pdf, text.md, figures/)
sims/<paper_id>/config.yaml     browser-runnable simulation config + paper targets + tolerances
app/                            static SvelteKit app: /table, /explore, /sim
scripts/                        validate_db.py, build_views.py, promote_open_extracts.py, ...
.claude/skills/eo-modulator-distill/
```

Integrity rule enforced by `scripts/validate_db.py`: every non-empty value in `devices.csv` and `papers.csv` metric columns has an entry in `evidence/<paper_id>.yaml` with a locator and a basis, or is flagged `derived` with its formula. Empty cell = not reported (`N/A`), never a guess.

## Representative device (multi-device papers)

1. Completeness score = count of core metric fields reported (Vpi or VpiL, 3 dB BW, on-chip IL, RF loss, Z0, n_RF, ng) / total core fields.
2. Tie-break: figure of merit `FOM = f3dB[GHz] / (Vpi_eff[V] * 10^(IL_onchip[dB]/10))`, where `Vpi_eff = Vpi * 10^(RF_loss_total[dB]/20)` when RF loss is reported, else `Vpi`. Higher is better. Defined only for TWE MZM / phase-shifter rows with BW, Vpi and IL all reported; otherwise the row ranks by completeness only.
3. The UI exposes alternatives (VpiL*alpha, BW/Vpi, lowest VpiL) as a selector; the default is rule 1 then rule 2.

## Phases and TODO

- [x] P0a Scaffold folder, decisions, plan
- [x] P0b Schema + validator + skill v1 (scripts/validate_db.py, merge_staging.py, extract_source.py, fetch_source.py; 4 pytest pass)
- [ ] P0c Candidate list from: Drive doc, tmp EO md, blog scan, private literature cache (re-sourced), OFC 2026 list, landmark expansion by platform
- [ ] P1 Pilot distillation of 3 diverse papers with the skill; audit; fix skill
- [ ] P1b Batch distillation (parallel subagents, rate-limited downloads); paywalled download list
- [ ] P2 Sim engine in browser (analytic gates first), configs for dielectric TWE papers
- [ ] P3 App: expandable table + plots + sim runner
- [ ] P4 Fresh-context audits (twice), audit corrections section below, README

## Progress log

- 2026-10-01 17:13Z scaffolded; toolchain checked (uv 0.7.16, node 23.7.0, pnpm 10.12.4; Crossref + arXiv API reachable).

- 2026-10-01 17:50Z spawned: seed candidate compiler, landmark expander, sim engine implementer. arXiv API returned 429/503 to the seed agent (two agents querying concurrently); reordered to Crossref + DataCite first, arXiv only after a >= 20 min pause with >= 10 s spacing.
- 2026-10-01 18:05Z pilot distillation launched on chen2022 (TFLN CL-TWE, sim config), kohli2025 (BTO plasmonic), ogiso2016 (InP n-i-p-n) from local PDFs; outputs staged under data/_staging/pilot_*; skill feedback requested.
- 2026-10-01 pilots merged into data/ (chen2022, kohli2025, ogiso2016; 3 papers, 7 devices, 8 orgs; validator 0 errors). Schema v2 applied from pilot feedback (bound qualifiers, push-pull MZM Vpi column, InP/resonator/stack columns, basis += author_estimate/design_target, series push-pull, eo_rolloff).
- 2026-10-01 candidates: seed 134 + landmark 76 -> data/candidates.csv 198 rows after dedupe (priority 1: 65, 2: 116, 3: 17). 8 priority-1 paywalled papers without local copy queued in data/manual_downloads.md. P1 batches p1_01..p1_11 (5 papers each); p1_01..06 launched, remaining after slots free.
- 2026-10-01 open items from pilots: ER/BW bounds now via qualifiers; unvalidated `context_values` block used by one pilot (not part of schema); organizations naming policy in schema convention (e).
- 2026-10-01 17:44Z batch p1_01..p1_06, the app builder and the sim-engine builder were all killed by an API session limit (HTTP 429, reset 14:50 local). Only `data/_staging/p1_01/cr/` (5 Crossref JSON) was written; p1_02..p1_11 did not run. Full narrative: DevLog-001.
- 2026-10-01 reference policy revised at the user's request: the full `references/` cache (PDFs, text.md, figures) and `data/_staging/` processed outputs are now tracked (`.gitignore` updated); only the private local-path mapping stays ignored. `references/README.md` updated to match.
- 2026-10-01 first commit pass: project committed in batches (scaffold/schema/skill, database+candidates, reference cache, staging, engine/sims, app); home-absolute local paths were rewritten to a tilde-prefixed form and private-corpus paths to `@corpus_A` / `@corpus_B` placeholders in committed data files.

## Audit corrections

(none yet)
