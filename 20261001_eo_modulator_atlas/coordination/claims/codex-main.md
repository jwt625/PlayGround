# Claim: codex-main

- Agent: Codex in the conversation that inspected the repository on 2026-10-01
- Task IDs: C0, E1, U1
- Status: active; E1/U1 implementation resumed at the user's request
- Updated: 2026-10-01
- Progress/handoff: `DevLog/DevLog-003-cross-section-progress.md`

## Owned write paths

Coordination/documentation:

- `WORKBOARD.md`
- `coordination/claims/codex-main.md`
- `DevLog/DevLog-000-plan.md` (status/link updates)
- `DevLog/DevLog-001-survey-proposal-execution.md` (successor link only)
- `DevLog/DevLog-002-work-plan-and-ownership.md`
- `DevLog/DevLog-003-cross-section-progress.md`
- `README.md`
- `.gitignore` (generated app assets only)

E1 engine boundary and baseline:

- `engine/src/config.mjs`, `engine/src/config.d.mts`
- `engine/src/run.mjs`, `engine/src/run.d.mts`
- `engine/src/optics.mjs` (off-window material fix and baseline only)
- `engine/tests/config.test.mjs`, `engine/tests/electrostatics.test.mjs`, `engine/tests/optics.test.mjs`
- `engine/tests/fixtures/parallel-plates.yaml` (analytic input shared with browser smoke)
- `engine/cli.mjs`, `engine/package.json`
- `engine/schema/sim.schema.json`, `sims/SPEC.md` (cross-section contract addendum only)
- `engine/README.md`, `engine/LICENSE`

U1 app integration:

- `app/src/routes/sim/+page.svelte`, `app/src/routes/about/+page.svelte`
- `app/src/routes/+layout.svelte` (U1 narrow-viewport header fix only; claimed after visual QA exposed overflow)
- `app/src/lib/sim.worker.ts`, `app/src/lib/CrossSection.svelte`
- `app/src/lib/ScoreCard.svelte`, `app/src/lib/logic.test.ts`
- `app/scripts/sync-sims.mjs`, `app/scripts/smoke.mjs` (if added)
- `app/package.json`

Generated local verification outputs: `app/build/`, `app/.svelte-kit/`,
`app/static/sims/`, `logs/`. No solver output is to be committed.

## Explicit exclusions

- No paper distillation, candidate changes or edits to `data/*.csv`,
  `data/evidence/`, `data/schema/`, `data/_staging/p1_*/` or reference caches.
- No edits to `sims/<paper_id>/config.yaml`; draft inputs retain their current status.
- No claim on downloader/merge/extraction scripts or distillation skill policy work (D0).
- No claim on EO overlap, RF loss, loaded-line/EO response implementation (E2–E4).
- No claim on full-vector/metal optical modelling after E1's baseline handoff (E2).
- No claim on table/explore feature work beyond the scorecard link (U2).
- No independent audit claim. The implementation author cannot satisfy Q1/Q2 by self-review.

## Current work

Finish the E1/U1 acceptance checks listed in DevLog-002, reconcile the runtime
contract with SPEC/schema, and report a reviewable handoff. No expansion into
other tranches without updating this claim. Release `engine/src/optics.mjs` to E2
and the scorecard/worker integration to U3 explicitly at handoff.
