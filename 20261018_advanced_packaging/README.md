# Advanced packaging literature triage (2022-2026)

Screens 2022-2026 advanced-packaging literature (ECTC, T-CPMT, EPTC, 3DIC, IEDM, VLSI, EDAPS, EPEPS, IRPS, ITherm,
ICEP, ESTC, EMPC, IWLPC, IMAPS), scores each abstract for extractable stack-up / process detail, and selects ~300
papers for a second-stage extraction agent. Spec: `init-prompt.md`. Plan, decisions, audit corrections:
`DevLog/DevLog-000-literature-triage-plan.md`.

## Deliverables (`outputs/`)
- `candidates.csv`: every enumerated record incl. front matter, duplicate index entries (marked in rationale) and curator-added out-of-window disclosures (DOI, title, first author, affiliation, venue, year, session, buckets, score, rationale, access, cache Y/N).
- `cache_list.csv`: selected papers, sorted by primary bucket then year, with priority, lineage_id, outside_screen and selection_reason.
- `triage_report.md`: venue enumeration, spec-list coverage, per-bucket summary and gaps, orientation reviews, method.

## Pipeline (`scripts/`, run with `uv run python scripts/<name>.py`)
1. `xplore_fetch.py`, `xplore_toc_check.py`: IEEE Xplore enumeration (headless Chromium, rate-limited) and TOC completeness check.
2. `crossref_fetch.py`: IMAPS Symposium and IWLPC records via Crossref.
3. `openalex_enrich.py`, `xplore_abstracts.py`: full abstracts, affiliations, OA status.
4. `ectc_sessions.py`: ECTC session/committee mapping from ectc.net final programs.
5. `prefilter.py`: merge into `cache/pool.jsonl`, bucket regex prefilter.
6. `make_batches.py`, `make_rescore.py`: LLM scoring batches (scored by subagents per `config/scoring_prompt.md`).
7. `make_curation_inputs.py`: per-bucket curation inputs (curated by subagents per `config/curation_prompt.md`).
8. `select_cache.py`, `report.py`: deterministic selection and report.

Config in `config/` (venues, buckets, orgs, selection rules, report notes). `cache/` and `logs/` are git-ignored
intermediates. API keys and contact email live in `.env` (git-ignored).
