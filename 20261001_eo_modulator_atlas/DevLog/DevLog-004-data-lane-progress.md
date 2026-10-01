---
title: EO modulator atlas - data lane (D0/D1/D2/Q1) progress
date: 2026-10-01
status: active
claim: coordination/claims/claude-data-lane.md
authors: Claude (at Wentao Jiang's request)
---

# DevLog-004: Data lane progress

Owner: `claude-data-lane` (see claim file). Plan and acceptance gates: [DevLog-002](DevLog-002-work-plan-and-ownership.md). Engine/app (E1, U1) belong to `codex-main`; this log does not touch them.

## Rules in force (data lane)

- At most 3 concurrent subagents (an 8-way parallel launch hit the API session limit on 2026-10-01 10:44 local).
- One download owner: only `scripts/prefetch_batch.py` (run by the coordinator, one at a time) touches the network for sources and Crossref; distillers are offline.
- No schema change without an interface claim; batches write `SPEC_PROPOSALS.md` / reports.
- Nothing enters canonical `data/*.csv` before Q1 review of that batch (D2 integrates one batch at a time).
- No commits by agents.

## Checklist

- [x] Claim recorded (`coordination/claims/claude-data-lane.md`), WORKBOARD rows D0, D1, D2, Q1 updated
- [x] D0 fetch serialization: `scripts/fetch_source.py` exclusive file lock held across spacing + request, circuit breaker (403/429/503/HTML/network error, 30 min cooldown), atomic `.part` -> `source.pdf` with %PDF magic, minimum size and Content-Length checks. 5 tests (`tests/test_ingest_fetch.py`): atomic cache + skip-existing, 4 simultaneous callers spaced >= MIN_DELAY, 429 arms breaker and blocks others without a network call, HTML challenge not cached, truncated body never becomes a cache hit
- [x] D0 prefetch: `scripts/prefetch_batch.py` (single download owner): Crossref JSON, local-corpus alias resolution via the git-ignored private mapping (PDF copy, or text-only copy for pre-extracted sources), arXiv PDF, Crossref-listed PDF link for CC-licensed papers, extraction, `prefetch_status.jsonl`, `needs_download.md`
- [x] D0 private-path hygiene: `data/candidates.csv`, `candidates_seed.csv` and batch CSVs now carry only `@corpus_*` / `@local_corpus` aliases; resolved paths live only in `data/_staging/candidates_seed_local_paths_private.csv` (git-ignored). `make_batches.py` rewrites any absolute or home-relative path to an alias and adds `source_state` (cached | local_alias | open_fetch). `git grep` for home paths over `data coordination scripts .claude sims` is clean
- [x] D0 policy alignment: skill rule 8 and 9 and workflow steps 2-3, `BATCH_INSTRUCTIONS.md` rewritten (offline distillers, tracked cache, no stale machine paths)
- [x] D0 merge tests (`tests/test_ingest_merge.py`): new paper, duplicate conflict vs `--replace-paper-ids` replacement (old devices removed), org conflict on type/country/region only (notes ignored), evidence files never overwritten
- [x] D0 checks: `uv run pytest -q` 26 passed; `uv run ruff check scripts tests` clean; `uv run mypy` clean (pymupdf typing override in `pyproject.toml`); `scripts/validate_db.py` 0 errors
- [x] Prefetch of p1_01..p1_11 run twice (first pass hit a Wiley 403 that armed the then-global breaker and blocked arXiv; breaker made per-host, publisher PDF attempts restricted to OA-friendly hosts, retry pass fetched the arXiv sources). Result: 55 papers, 11 still without a retrievable source (OA publisher PDFs blocked to scripts: Optica, AIP, Wiley, Nature OA via DOI), added with the 8 earlier paywalled ones to `data/manual_downloads.md` (19 papers). Text-only local sources (no figures): qiu2026, su2026, zhang2026b, liu2026c.
- [x] Q1 pilot audit (fresh context): `data/_staging/audits/pilots-q1-fresh.md`; all three accepted with corrections (1 high, 7 medium, 24 low, 10 cross-paper issues). Corrections applied to canonical data 2026-10-01 (backup in the session scratchpad): chen2022 ER lower-bound qualifiers + er_type, approx qualifiers, bw3db_ghz lower bound 67, eo_rolloff 1.4 dB at 67 GHz, drive/vpi_convention evidence (derived), RF-line values kept only on the 10 mm row, HK country code; kohli2025 plasmonic_mzm -> mzm + tag, approx qualifiers, notes; ogiso2016 bound moved to bw3db_ghz, eo_rolloff 2.7 dB, il_basis author_estimate; crossref.json cached for the three pilots. Verification of the corrections is part of the next audit pass
- [ ] D1 batches (3 concurrent roles): p1_01 distilled, audited (0 high / 3 medium / 20 low, `data/_staging/audits/p1_01-q1-fresh.md`), corrected by its author and integrated (D2) 2026-10-01; p1_09 distilled, audit running; p1_02 and p1_03 distilling; p1_04..p1_08, p1_10, p1_11 queued
- [ ] Q1 review per batch (fresh-context auditor, read-only)
- [ ] D2 integration, one batch at a time, then regenerate the app data view

## Progress log

- 2026-10-01 14:51 local: user reported the usage limit recovered and that a separate agent (`codex-main`) had scoped the work and taken C0/E1/U1. State inspected: 11 commits by the user, engine and app committed or in progress by `codex-main`, no batch report from the interrupted workers (only `p1_01/cr/` Crossref responses), canonical data = 3 papers / 7 devices / 8 organizations.
- 2026-10-01 ~15:10: D0 completed (see checklist). Prefetch started.

## Known interface issues for later tranches (not applied)

- `devices.csv` has no column for `vpi_rf_mzm_pushpull`; kohli2025 notes it (a coordinated schema tranche is needed if several batches hit the same gap).
- The evidence file for ogiso2016 carries a non-validated `context_values:` block (numbers with no column); the validator ignores unknown top-level keys. Decide in D2 whether to formalize or drop.
- Row basis columns (`vpi_basis`, `bw_basis`, `il_basis`) are single-valued; per-field basis in the evidence wins (rule 11).
- Rights: the tracked reference cache includes publisher-copyright PDFs (user decision 2026-10-01 per `codex-main`'s record; not independently confirmed in this conversation). `redistribution` stays per source; R1 must confirm the public-spin-off policy before any public push.

## Convention decisions after the pilot audit (2026-10-01)

Recorded in the conventions block (a)-(k) of `data/schema/devices.schema.yaml` (comment text; no column changes) and SKILL.md rule 11: bounds via `qualifiers` for any over/below/about wording; bandwidth without crossing = `bw3db_ghz` lower bound + `bw_measured_to_ghz`; `drive` means arm drive topology; `eo_rolloff_db` positive drop; author-deduced values never `measured`; locators use the PDF page index; plasmonic nature in `tags`; HK/TW own country codes; crossref.json required evidence for dates/licenses.

## Items for the sim/SPEC owner (not applied by this lane)

From the audit of `sims/chen2022/config.yaml` (all low severity): S1 `sio2_cladding` provenance class (0.9 um is the text value, not digitized); S2 several LN and SiO2 constants labelled `standard_reference` although their notes say unverified (suggest an `unverified_standard` class); S3 sidewall note (about 68 deg) disagrees with the polygon (65.8 deg); S4 target signs (S21-style negative roll-off) vs the CSV convention (positive drop, convention g); S5 "T-rail" is not the paper's term (T-segment). The audit also confirms the paper-simulated vs measured target labelling.

## Integration record (D2)

| Batch | Date | Papers / devices / orgs added | Review | Notes |
|---|---|---|---|---|
| pilots (chen2022, kohli2025, ogiso2016) | 2026-10-01 | 3 / 7 / 8 | Q1 fresh audit, corrections applied after merge (verification pending in the pre-release pass) | backup of pre-correction data kept outside the repo |
| p1_01 | 2026-10-01 | 5 / 9 / 11 | Q1 fresh audit, author corrections applied before merge | deng2026 sim config not runnable (no sourced RF permittivities); qiu2026 text-only source |

After each integration: `scripts/validate_db.py` 0 errors, `uv run pytest -q` 28 passed, `scripts/build_views.py` regenerated `app/static/data/atlas.json` (currently 8 papers, 16 devices, 19 organizations, 0 warnings).

Clarification added to convention (c): when a paper states a bandwidth bound below the plotted range (exceeds 100 GHz on a trace measured to 110 GHz), `bw3db_ghz` carries the paper's bound with `gt` and `bw_measured_to_ghz` the actual measured range.

Open for the integrator: org-name spellings across batches (EPFL, imec) must be unified when later batches merge; the merge only compares type/country/region. Sim configs written by batches may contain unverified constants (flagged in provenance); none has been run.
