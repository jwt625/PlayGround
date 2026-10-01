# Claim: claude-data-lane

- Agent: Claude (original orchestrator conversation: survey, proposal, skill, schema, candidates, pilots, batch launch)
- Task IDs: D0, D1.01-D1.11 (via at most 3 concurrent subagents), D2 (serial integrator), Q1 (via fresh-context auditors)
- Status: claimed; D0 in progress
- Updated: 2026-10-01
- Progress/handoff: `DevLog/DevLog-004-data-lane-progress.md`

## Owned write paths

D0 tooling:
- `.claude/skills/eo-modulator-distill/`
- `data/_staging/BATCH_INSTRUCTIONS.md`, `data/_staging/batches/`
- `references/README.md`
- `scripts/fetch_source.py`, `scripts/extract_source.py`, `scripts/make_batches.py`, `scripts/merge_staging.py`, `scripts/merge_candidates.py`, `scripts/init_tables.py`, `scripts/validate_db.py` (validator logic only; schema changes need an explicit interface claim)
- `tests/test_ingest_*.py`, `tests/test_validate_db.py`

D1 batches (per-batch ownership delegated to subagents under this claim):
- `data/_staging/p1_NN/`, `references/<assigned-paper-id>/`, `sims/<assigned-paper-id>/` (never `sims/chen2022/` or `sims/SPEC.md`)

D2 integrator (serial writer, after Q1 review of a batch):
- `data/papers.csv`, `data/devices.csv`, `data/organizations.csv`, `data/evidence/`, `data/manual_downloads.md`, `data/candidates.csv`, `data/_staging/integration/`, `app/static/data/atlas.json` (regenerated only by `scripts/build_views.py`; coordinate with U2 owner)

Q1 auditors write only `data/_staging/audits/<scope>-<agent-id>.md`.

## Read-only inputs
`engine/`, `app/`, `sims/SPEC.md`, `data/schema/devices.schema.yaml` (changes only through a coordinated interface claim), `DevLog/DevLog-000..003`, `WORKBOARD.md`, `coordination/claims/codex-main.md`.

## Explicit exclusions
No edits to `engine/**`, `app/src/**`, `app/package.json`, `sims/SPEC.md`, `WORKBOARD.md` rows owned by other claims, or `DevLog-000..003`. No schema change without an interface-change claim. No commits (the user commits). No solver outputs stored. No agents beyond three concurrent roles (the earlier 8-way parallel launch triggered the API session limit).

## Dependencies / requested interface changes
- D0 fetch serialization must be fixed and tested before more than one agent downloads; until then at most one download owner.
- Schema changes proposed by batches are collected in the batch staging dirs and decided in one coordinated tranche, not applied opportunistically.

## Acceptance checks
Per DevLog-002 section 2 (D0, D1, D2, Q1) plus: `uv run pytest -q` and `uv run python scripts/validate_db.py` pass after each integration; no private source paths in tracked files.

## Handoff report path
`DevLog/DevLog-004-data-lane-progress.md`
