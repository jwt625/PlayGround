---
title: EO modulator atlas - E1/U1 progress and handoff
date: 2026-10-01
status: ready_for_review
owner: codex-main
tasks: [E1, U1]
---

# DevLog-003: Cross-section runner progress

Scope and write boundaries: [claim](../coordination/claims/codex-main.md).
Coordination plan: [DevLog-002](DevLog-002-work-plan-and-ownership.md).
This log records implementation and checks; it does not upgrade paper validation.

## Acceptance checklist

- [x] Runtime, JSON schema and SPEC agree on the supported cross-section subset.
- [x] Invalid/unsupported inputs fail explicitly; target comparison uses only independently computed, comparable metrics.
- [x] Electrostatic and scalar optical analytic tests pass on the final tree.
- [x] CLI success/error/target-failure behavior and browser/Node agreement verified.
- [x] App unit tests, Svelte check and production build pass.
- [x] Chrome run/cancel/error/retry/stale-result/unknown-config/navigation checks pass.
- [x] Narrow viewport and subdirectory deployment checked.
- [x] Documentation and review handoff complete; Q2 remains independent.

## Progress journal

- 2026-10-01 — Resumed E1/U1 after the user accepted the claim and requested
  continued work. Inspected the workboard, claim files and working tree; no
  competing claims were present. The previous preview server is no longer running.
- Input-boundary review underway: align supported chain/loading options, target
  eligibility and mesh limits with the schema/SPEC before running final checks.
  No canonical database or paper input files will be changed in this tranche.
- 2026-10-01 — Input review completed for this tranche: reject malformed chain,
  loading references, numerical controls and partial optical indices. Unknown
  topology, frequency-specific targets and configured group index no longer
  participate in target agreement. Both meshes obey the configured vertex budget;
  a real index cannot bypass the unsupported-metal check. SPEC now documents
  implementation scope, defaults, units, return fields and exit semantics.
- Engine tests now pass 17/17, including analytic cases, runner integration,
  CLI 0/1/2 behavior and contract checks. Corrected app tests pass 5/5 and Svelte
  reports 0 errors/0 warnings. Browser interaction and final build checks are next.
- 2026-10-01 — Production build passed. Initial expanded Chrome smoke passed
  dashboard/table/explore/about and the actual Chen worker/Node comparison, then
  caught ambiguous select labelling in the section control. Added explicit
  accessible names for config, section and mesh controls; rerunning the suite.
  Worker teardown now detaches handlers and ignores messages from superseded runs.
- 2026-10-01 — Root-path interaction suite passed, including cancellation,
  invalid input, unsupported optical error, retry, stale/alternate results and
  Node/browser agreement. Visual inspection then exposed an existing mobile
  header that widened the entire app beyond the viewport; the initial local
  container assertion missed it. Extended the U1 claim narrowly to the global
  header layout and strengthened the smoke assertion to inspect the root width.
  Subdirectory deployment verification is underway.
- 2026-10-01 — Reviewed new data-lane progress from commit `8029881`: D0 tooling,
  the pilot corrections, p1_01 integration and the p1_02/p1_03/p1_09 reports and
  interface proposals. Canonical data and generated view now contain 8 papers,
  16 devices and 19 organizations. Updated the app's Kohli filter test to the
  audited `mzm` classification; production filter logic is unchanged.
- The root-path Chrome suite passed after the header fix; visual inspection of
  the 390 px screenshot confirms the global header fits. The next run includes
  the newly integrated data and incomplete-draft behavior.
- Added a separate `inspectConfig` path: malformed geometry still fails, but
  valid geometry and disclosure notes remain visible when physical inputs are
  missing. It returns no resolved materials, does not fill constants, and leaves
  Run disabled. Strict CLI/worker parsing is unchanged. This addresses the
  p1_01/p1_02 preview request without a schema/status/sentinel migration.
- Current gates: engine 20/20, app 5/5, Svelte 0 errors/0 warnings, database
  validator 0 errors. Python collection exposed the newly required extraction
  extra (`pymupdf`); installing the declared dependency and updating README setup
  before rerunning. Root/base-path browser checks and handoff remain in progress.
- Python checks passed after installing the declared extraction extra: 28 tests,
  with PyMuPDF/SWIG deprecation warnings only. YAML 1.2 parsing plus JSON Schema
  validation agrees with the strict runtime for all seven paper drafts: Deng/He
  reject only the missing RF-permittivity fields; the other five pass the input
  boundary. All seven support the geometry preview. This is not a numerical or
  provenance validation of any newly supplied constants.
- The new browser check caught a missing space between the blocked-preview label
  and error text caused by Svelte whitespace handling. Fixed the text expression
  and allowed long diagnostics to wrap; added a narrow-viewport check for the
  incomplete draft as well as the runnable fixture.
- Final root and `/eo-atlas` builds and Chrome suites passed on the updated
  implementation. Inspected desktop, incomplete-draft and 390 px screenshots;
  all fit their intended layouts. Restored the default root-path build for local
  preview. E1/U1 are now `ready_for_review`, not independently accepted/completed.

## Cross-agent review and follow-up

The implementation review used the data lane's reports and audits as inputs; it
is not a replacement for Q1 or an independent numerical audit. No canonical CSV,
evidence, batch, paper-config or data-lane claim files were edited.

| Item | Finding / disposition | Next owner |
|---|---|---|
| D0 and p1_01 | Tooling and first batch integrated in `8029881`; validator and all 28 Python tests rerun here | Data lane continues serial D2 and Q1 |
| Pilot corrections | Kohli class is now `mzm`; app integration test updated. Bounds and EO roll-off convention changed in canonical evidence, not the simulation files | U2 must preserve qualifiers in comparison/representative views; Q1 verifies pilot corrections |
| Missing constants (p1_01/p1_02) | Preview/disclosure request implemented without loosening solver requirements or introducing a schema sentinel | Data authors supply verified constants before numerical use |
| p1_09 audit | `p1_09-q1-fresh.md` reports 2 high, 6 medium, 29 low issues. High items BC1/LC1 concern recalled LT constants presented with a citation not verified for LT. Configs parsing successfully does not resolve these findings | Data lane applies/verifies corrections before accepting these configs as sourced inputs |
| Current generated config list | `atlas.json` lists Chen, Deng, Li2026ba and Lin2025; the latter two papers are not yet canonical. He, Kharel and Liu files also exist but are absent from this generated view | D2 should reconcile discoverability and integration state on its next view refresh; U2 should distinguish draft config availability from reviewed paper coverage |
| Chen config audit S1–S5 | Provenance class, unverified constants, sidewall description, roll-off sign and electrode terminology remain paper-input corrections; this tranche did not edit Chen | Claim the Chen input explicitly for Q1 correction/E5 before changing it |
| Verified constants and tensors | Shared cited RF/Pockels library, partial-tensor warnings, BTO crystal directions, arm/drive convention, variant support | E2 design proposal; coordinated interface migration before implementation |
| RF and periodic cells | Loss tangents, three-section cells, neutral loaded-line naming, target basis/bounds and positive-drop versus signed-S21 conventions | E3/E4 with E5 and data owner; these metrics remain unevaluated in E1 |
| Optical boundaries | Truncated windows, uncertain layers and metal-aware/vector modes require a model/convergence decision | E2 and Q2; no invented optical index or implicit PEC approximation |

Detailed proposals remain in the owning batches' `SPEC_PROPOSALS.md` files. The
existing provenance classes suffice to record an explicit project assumption;
an unverified citation must not be treated as verified just because its class
passes an enum check. No new provenance class was added in this tranche.

## Checks

Run from the repository root unless a directory is shown. Base inspected:
`8029881` (data lane), following `ef32d46` (E1/U1 implementation). This follow-up
is uncommitted; no agent commit was created.

| Check | Command / procedure | Outcome |
|---|---|---|
| Engine | `cd engine && npm test` | 20 passed; parallel plates, full/half-domain voltage normalization, anisotropic rotation, TE/TM slab modes, group index, resource/error guards, target eligibility and CLI 0/1/2 |
| App integration | `cd app && npm test` | 5 passed with the updated canonical data |
| Svelte/TypeScript | `cd app && npm run check` | 0 errors, 0 warnings |
| Database | `.venv/bin/python scripts/validate_db.py` | 0 errors |
| Python tooling | `.venv/bin/python -m pytest -q` | 28 passed; 5 PyMuPDF/SWIG deprecation warnings |
| Root build/browser | `cd app && npm run build && SMOKE_SCREENSHOTS=../logs/e1-u1 npm run smoke` | Pass; actual Chen and synthetic optical browser/Node agreement, all interaction/error cases, incomplete-draft and narrow-viewport checks |
| Subdirectory | `cd app && BASE_PATH=/eo-atlas npm run build && BASE_PATH=/eo-atlas npm run smoke` | Pass with the same coverage |
| Input contracts | All 7 paper YAML files and the analytic fixture parsed with engine YAML 1.2; `Draft202012Validator.check_schema` and `iter_errors` against `engine/schema/sim.schema.json` | 5 paper configs + fixture pass schema; Deng (3 missing `eps_r`) and He (4 missing `eps_r`) fail as intended; strict runtime agrees and all paper previews render |
| Patch hygiene | `git diff --check` | Pass |

The schema check used a local `jsonschema` installation and transient parsed
inputs at `logs/e1-u1-schema-inputs.json`. Do not use PyYAML's YAML 1.1 number
interpretation for this check: values such as `4.1e7` are valid YAML 1.2 numbers.
Build output retains the existing large Plotly chunk warning; bundle work belongs
to U2/R1. Screenshots, build logs and generated app files are ignored local QA
artifacts. No solver result file or output array is included in the patch.

## Handoff

**E1/U1 are ready for independent review.** The original implementation spans the
files listed in [the claim](../coordination/claims/codex-main.md). This follow-up
changes the config inspector/type declaration, simulator route, engine boundary
tests, app data test and browser smoke, plus SPEC/README/coordination documents.
It adds incomplete-draft inspection and reconciles the new data without changing
the numerical algorithms, paper inputs or database schema.

The shared interface is `runCrossSection(text, options)` and its typed result;
`inspectConfig(text)` is display-only. Future stages must preserve independently
computed target coverage, units, error semantics and the terminal-voltage
normalization documented in SPEC. A passing input check or analytic test does
not promote `validation_status` or resolve an unverified constant.

Files released for the next **explicit** claim:

- E2: `engine/src/optics.mjs`; dedicated new overlap/model tests and modules per
  DevLog-002. Shared config/material/runner/SPEC changes still need coordination.
- U3: simulator route, worker, `CrossSection.svelte` and `ScoreCard.svelte` when
  its E4/E5 dependencies are ready; preserve U1 error/cancel/coverage behavior.
- U2: `app/src/lib/logic.test.ts` and the narrow header portion of
  `app/src/routes/+layout.svelte`; further shared-layout changes need a claim.

Codex retains coordination and shared E1 contract maintenance through review.
No successor is assigned by this release. Q2 should independently check voltage
and energy factors, anisotropic frame mapping, optical normalization, finite
domain/window assumptions and target eligibility; author-written tests do not
satisfy that audit. Mesh/domain convergence, Chen metal handling, EO/RF/periodic
physics, literature regressions and Q1 correction verification remain open.
