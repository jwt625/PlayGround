---
title: EO modulator atlas - coordinated work plan and ownership
date: 2026-10-01
status: active
scope: categories, tranches, dependencies, claims, acceptance gates and handoffs
---

# DevLog-002: Coordinated work plan

This is the current execution plan, following the user's request to plan and
claim work before continuing implementation. DevLog-000 retains project
decisions; DevLog-001 is historical. [WORKBOARD.md](../WORKBOARD.md) summarizes
current assignments; individual files under `coordination/claims/` define write
ownership. No new agents have been launched by this continuation.

**Progress update, 2026-10-01:** E1/U1 are ready for review; final checks and file
releases are in [DevLog-003](DevLog-003-cross-section-progress.md). The data lane
has claimed D0/D1/D2/Q1 and integrated p1_01 (8 papers / 16 devices / 19 orgs);
see [DevLog-004](DevLog-004-data-lane-progress.md). The starting snapshot and
dispatch-cache map below are historical, not current source availability.

## 1. Verified starting state

| Area | Observed state on 2026-10-01 | Implication |
|---|---|---|
| Canonical database | 3 papers, 7 devices, 8 organizations; validator reports 0 errors | Expand through staging, preserving evidence requirements |
| Candidates | 198: priority 1 = 65, priority 2 = 116, priority 3 = 17 | Candidate discovery is complete for the initial scope; verification is separate |
| Priority-1 disposition | 2 already canonical, 55 in 11 batches, 8 on manual-download list | Kohli pilot is not one of the two priority-1 canonical papers |
| Reference cache | 12 source PDFs with extracted text/figures, including the three pilots | Nine queued papers can start from cached sources |
| Batch execution | No `p1_*/BATCH_REPORT.md`; only p1_01 Crossref responses exist | Do not assume the interrupted workers completed distillation |
| Original app | Table/explore implemented; sim/about were placeholders | App completion and data scale-up can proceed independently |
| Original engine | Geometry, mesher, FEM, electrostatics, material and optical modules; no runnable chain/CLI/tests | Validate foundations before adding coupled physics |
| Current working tree | E1/U1 additions by codex-main, uncommitted | Treat as an active claim, not a completed shared baseline |

Existing checks observed during this continuation: 17 Python tests passed;
database validation passed; 11 new engine tests passed. App build and desktop
Chrome smoke passed on an intermediate working snapshot. The first five-test
app run had one fixture mismatch (`mzm` versus `plasmonic_mzm`); the fixture was
corrected but has not yet been rerun. Final checks of the latest tree are pending.

## 2. Categories and detailed tranches

Each tranche should produce a reviewable handoff. Dependencies describe when
implementation can integrate; reading sources and drafting a proposal can start
earlier in an agent's own report file.

### C — Coordination

**C0: Work decomposition and claims — codex-main, complete.**
Deliver this plan, workboard and an explicit claim for the existing changes.
Record overlaps, shared-file owners, pending checks and recommended scheduling.
Acceptance: another agent can select a task and identify its inputs, outputs,
dependencies and write boundaries without reconstructing the chat.

### D — Evidence, sources and database

**D0: Ingestion tooling and instruction alignment — claude-data-lane; implementation reported complete.**

The list below records the original defects/acceptance scope. DevLog-004 reports
their fixes and tests; they are not outstanding fetch defects after `8029881`.

- Reconcile outdated instructions with the user's revised tracked-reference
  policy. Skill rule 8, early DevLog-000 text and batch-generation comments still
  describe an ignored cache; the later user decision and `references/README.md`
  describe the tracked cache. Preserve source-specific rights metadata.
- Remove stale machine-specific paths from batch instructions. Prefer an existing
  `references/<paper_id>/` source before trying obsolete local-path hints.
- Fix cross-process fetching coordination: `fetch_source.py` currently uses a
  timestamp without a lock. Its claim of one request at a time across agents is
  not enforced. Add locking or a single request broker and meaningful concurrency
  tests; cover failure/429 behavior and prevent partial PDF files becoming cache hits.
- Review `make_batches.py` so re-running it cannot publish private resolved paths
  into now-tracked batch artifacts. Test replacement/conflict behavior in the
  staging merge path before scaling integration.
- Write contract changes/proposals explicitly; do not expand the data schema
  opportunistically during a paper extraction.

Write ownership: `.claude/skills/eo-modulator-distill/`,
`data/_staging/BATCH_INSTRUCTIONS.md`, `references/README.md`,
`scripts/fetch_source.py`, `scripts/extract_source.py`, `scripts/make_batches.py`,
`scripts/merge_staging.py` and task-specific tests. Coordinate any requested
DevLog/schema edits with their owners. No canonical metric edits.

Acceptance: policy text agrees; two simultaneous fetch attempts cannot violate
the intended serialization; blocked downloads produce actionable records; no
private path leaks; ingestion/merge tests and existing Python tests pass.

**D1.01–D1.11: Priority-1 distillation — claimed by claude-data-lane.**

Batch allocation is now coordinated under that claim (at most three concurrent
roles); do not claim these paper batches independently of its owner.

One batch owner reads primary sources, inspects numerical figures/tables, writes
paper/device/organization rows and evidence, and drafts eligible dielectric-TWE
configs. Use the existing skill and batch contract, with the later user reference
policy taking precedence over stale cache wording. Do not infer device metrics
from title, candidate metadata, blog pointers or unrelated comparison tables.

Write ownership for D1.N: `data/_staging/p1_NN/`, plus
`references/<assigned-paper-id>/` and `sims/<assigned-paper-id>/`. Keep schema
proposals and needed-download entries in the batch's staging folder. Never edit
canonical tables, other batches, shared material constants, or engine code.
The existing Chen config is outside all these batch claims.

Acceptance per batch:

1. Every assigned paper has a disposition: distilled, no device rows, needs
   download, or failed with a concrete reason; partial access is not completion.
2. Every populated metric has matching evidence with locator, unit, basis and
   relevant drive/bandwidth/loss convention. Missing values remain empty.
3. New organizations match the naming policy; duplicate paper versions are resolved.
4. Configs are inputs only, with provenance, missing inputs and honest targets.
   CLI execution, when requested, is a technical check, not automatic paper validation.
5. Dry-run merge has 0 conflicts and 0 validation errors, and `BATCH_REPORT.md`
   states what was read, judgment calls and remaining blockers.
6. Q1 reviews the batch before D2 integration. A structurally valid row is not
   necessarily a correctly extracted value.

**D2: Canonical integration and data-view refresh — claude-data-lane; serial writer.**

Accept reviewed staging artifacts, resolve organization/ID conflicts, integrate
with `merge_staging.py --apply`, update the manual-download list, validate the
canonical database, and regenerate the app view. Maintain a short per-batch
integration record with row counts and corrections. Rejected rows remain staged.

Write ownership: `data/papers.csv`, `data/devices.csv`, `data/organizations.csv`,
`data/evidence/`, `data/manual_downloads.md`, `app/static/data/atlas.json` and
integration records under `data/_staging/integration/`. Schema/build-view changes
need an explicit interface-change claim, since U2 consumes their output.

Acceptance: validator + Python tests + app data-contract tests pass; no unreviewed
metric enters canonical data; regenerated view agrees with canonical counts;
no solver outputs or private source paths are included. Apply one batch at a time.

### E — Numerical engine

**E1: Runnable cross-section baseline — codex-main; ready_for_review.**

Deliver shared YAML parsing/validation, browser/Node runner, CLI, typed result
boundary, scalar optical material fix, input schema and documentation. Test
parallel-plate capacitance, charge versus energy, differential/half-domain
normalization, tensor rotation, TE/TM slab modes and waveguide group index.

Scope is electrostatics plus optional dielectric scalar optical modes. Explicitly
reject unsupported optical metal handling and missing required physical inputs.
Keep unsupported targets `not_evaluated`, including effective periodic-line
RF targets when only one section is solved. Do not update a paper's validation
status from analytic tests or from successful CLI execution.

Acceptance: engine tests pass, input errors are readable, Node/browser results
agree for the same input, runtime/SPEC/schema agree on the implemented subset,
CLI exit semantics distinguish execution from target agreement, numerical caps
and cancellation are documented. Q2 reviews the voltage normalization and limits.
E1 does not claim full mesh convergence or Chen device reproduction.

**E2: Optical limits and EO overlap — unassigned; E1 handoff available.**

- Review metal intersections, domain truncation, optical boundary conditions,
  scalar/vector validity and material dispersion ranges. Resolve Chen's optical
  window problem using a justified model, not an invented gold index.
- Freeze explicit single-arm/push-pull/differential voltage and arm definitions
  before implementing VπL. Separate terminal voltage normalization from the
  optical phase difference between arms.
- Implement rotated-tensor Pockels overlap and mode normalization; expose the
  convention and assumptions in outputs.
- Verify uniform-field/analytic overlap, symmetry/sign tests and the expected
  drive factors; compare representative anisotropic cases to an independent
  calculation or solver before generalizing.

Write ownership after release: `engine/src/eo-overlap.mjs` (new), relevant new
tests, and `engine/src/optics.mjs`. Changes to `materials.mjs`, SPEC, config or
runner require coordinated interface edits. No target tuning.

**E3: Uniform RF line and loss — unassigned; standalone design can start now.**

Use section C′/C₀′/L′ to compute RLGC, propagation and impedance with explicit
units, frequency conventions and sourced loss models. Implement/test conductor
and dielectric loss; a paper-specified attenuation table must be labelled as an
input rather than an independent loss prediction. Review missing substrate
conductivity and loss tangents instead of defaulting them to known values.

Write ownership: new `engine/src/rf-line.mjs`, any new complex-number helper,
and task-specific tests/docs. Dependencies: E1 result contract; request shared
material/config/runner edits through the owner. Acceptance: lossless TEM limits,
passivity, frequency/unit conversions and at least one analytic lossy-line case.

**E4: Periodic line and traveling-wave EO response — unassigned; after E2/E3.**

Implement unit-cell ABCD composition, periodic length/fraction conventions,
effective propagation, source/load mismatch, spatial EO integration and bandwidth
relative to an explicit reference frequency. Handle no threshold crossing as a
bound, not an invented bandwidth; state finite-sweep and Bragg limitations.

Write ownership: new `engine/src/loaded-line.mjs`, `engine/src/eo-response.mjs`
and dedicated tests. Acceptance: uniform-cell reduction, matched/lossless and
velocity-matched limits, analytic mismatch/loss curves, stable frequency sweep,
and explicit handling of finite-cell versus homogenized results. Integration
into shared runner/contract is a coordinated handoff.

**E5: Literature regressions and convergence — unassigned; after E2–E4.**

Start with Chen 2022, then independently reviewed Kharel/Valdez/other eligible
configs. Run mesh, domain, optical-window and model-sensitivity studies; separate
measured and paper-simulated targets. Record pass/fail, convention corrections,
and model/disclosure limitations in a DevLog. Do not commit solver arrays, fitted
parameters or stored response curves. Promote validation status only for the
specific level actually demonstrated and independently reviewed.

### U — Browser application

**U1: Cross-section simulator and baseline app — codex-main; ready_for_review.**

Deliver config selection, safe local static asset lookup, editable YAML, geometry
preview, section/resolution controls, cancellable worker, target status and
provenance/missing-input views. Fill About, link the existing scorecard and document
setup. Keep calculated values in memory and distinguish stale results after edits.

Acceptance: app tests, Svelte checks and production build pass; Chrome smoke covers
table/explore/about/sim, a real worker run, cancellation, invalid input, edited-input
staleness, unknown config and navigation; check narrow viewport and non-root base
path. UI must communicate the implemented physical scope and incomplete targets.
No external deployment is included in this tranche.

**U2: Table/explore correctness and usability — unassigned; ready in disjoint files.**

Audit representative selection, qualifier propagation in derived quantities,
voltage comparability, RF-loss frequency context, measured-only filtering,
Pareto treatment of bounds, URL/back navigation, empty states, table expansion,
CSV export and plot provenance. Confirm the specified VπL·α and other plot options
against the original decisions; record deliberate departures rather than silently
substituting another FOM. Add targeted tests for incorrect scientific comparisons.

Write ownership: `app/src/routes/table/`, `app/src/routes/explore/`, the relevant
chart/filter/drawer components, `logic.ts`, `charts.ts`, and new dedicated tests.
`logic.test.ts` and U1's narrow header layout scope are released for a U2 claim.
Scorecard, worker and sim route are released for U3 after its physics dependencies;
About and the package manifest remain shared maintenance files. Coordinate shared
`types.ts`, other global layout/CSS and `build_views.py` changes with their owners.

Acceptance: materially comparable values retain conventions, bounds and source
basis; missing values are not plotted as zero; filter/sort/URL/CSV regression tests
and representative desktop/mobile smoke checks pass.

**U3: Full-chain visualization — unassigned; after E4/E5 and U1 release.**

Add computed EO/RF traces, paper targets and tolerance comparisons, stage errors,
frequency/reference labels and an honest reproduction score. Separate evaluated
coverage from target pass fraction. Never display an unevaluated target as passed.
Write ownership by handoff: sim route, worker types, scorecard and new result
components. Acceptance includes synthetic known-limit curves and incomplete-run
states as well as reviewed paper configs.

### Q/R — Independent review and release

**Q1: Evidence audit — claude-data-lane via fresh-context auditors.**
Read primary sources independently. Recheck identity/version, headline numbers,
units, conventions, bounds, organization names and numerical figure locators.
Write `data/_staging/audits/<scope>-<agent-id>.md`; do not edit the author's data
while auditing. Author fixes findings; auditor verifies corrections. A second
fresh-context pass before release satisfies the original twice-audited requirement.

**Q2: Numerical audit — unassigned; initial review ready now.**
Independently derive/check energy and charge capacitance, symmetry, tensor frame,
mode normalization and target comparability. Later review RF loss, ABCD/EO
response and convergence claims. Write `DevLog/audits/<scope>-<agent-id>.md` and
new `engine/tests/audit-<agent-id>.test.mjs` if needed. Read-only on implementation
files. Acceptance: reproducible findings, fixes verified, and no conflation of
analytic correctness with paper reproduction.

**R1: Release integration — unassigned; after accepted tranches.**
Run the combined checks, a second independent audit, clean-install and deployment
path checks, provenance/license inventory and public-source/path review. Confirm
the chosen publication policy for the tracked source cache before any public
spin-off. Prepare a reviewable release; no remote publishing is implied by this
coordination plan. Update README and completion statuses based on evidence.

## 3. Batch dispatch map

Cached means `references/<paper_id>/text.md` exists; it does not establish that
the source is complete, correctly identified or already audited.

| Task | Papers in the existing manifest | Cached queued papers |
|---|---|---|
| D1.01 | deng2026, qiu2026, ogiso2024, porto2026, tanaka2026 | all except qiu2026 |
| D1.02 | tran2026, wang2018, weigel2018, he2019, boynton2020 | tran2026 |
| D1.03 | kharel2021, liu2021, pan2021, arabjuneghani2022, mao2022 | kharel2021 |
| D1.04 | valdez2022, xu2022, meng2023, renaud2023, valdez2023 | valdez2022 |
| D1.05 | valdez2023a, li2025a, liu2025, liu2025b, murai2025 | liu2025b |
| D1.06 | rahman2025, zhang2025, didier2026, lee2026, li2026aa | lee2026 |
| D1.07 | su2026, xue2026, zhang2026b, zheng2026, powell2024 | none |
| D1.08 | wang2024a, wang2024b, wang2025, cai2025, chen2026 | none |
| D1.09 | li2026, li2026a, li2026ba, lin2025, niels2026 | none |
| D1.10 | sayem2026c, valdez2026, xu2026a, xu2026b, derose2012 | none |
| D1.11 | yue2025, dong2026, liu2026b, liu2026c, wang2026a | none |

The eight priority-1 manual-download papers are eltes2019, gupta2023, ogiso2017,
ogiso2020, wang2018a, yang2022, zhang2021b and zhang2022. Preserve exact filenames
in `data/manual_downloads.md`; assign ingestion only after a source arrives.

Prioritize D1.01 for cached-source throughput and D1.03/D1.04 for dielectric-TWE
inputs. Then D1.05/D1.06, followed by the source-dependent batches. A batch can
finish with documented access blockers; report distilled and blocked counts
separately. Do not repeatedly re-fetch blocked sources.

## 4. Suggested parallel launch schedule

This is a proposal for task allocation, not a record of agents already running.
Use at most four simultaneous roles in the current runtime. External sessions
must follow the same file ownership and fetch coordination.

| Wave | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Exit/handoff |
|---|---|---|---|---|---|
| A: foundation | codex-main: E1 + U1 | D0 tooling | D1.01, cached papers first | Q1 pilot audit or Q2 numerical audit | Reliable contracts/tooling, first staged batch, independent findings |
| B: parallel growth | E2 optics/EO owner | E3 RF owner | D1.03 then D1.04 | U2 or rotating Q1/D2 reviewer | EO/RF modules independently tested; reviewed batches integrated serially |
| C: coupled model | E4 periodic/EO response | Additional D1 batch | Additional D1 batch | Q1/D2 rotating integrator | Complete chain and growing verified database |
| D: qualification | E5 regressions | U3 full-chain UI | Q2 independent numerical audit | Remaining D1/Q1 | Convergence and review evidence |
| E: release | R1 integrator | Second fresh-context evidence audit | Second numerical/UI audit | Remaining source blockers | Reviewable release with truthful coverage |

Until D0 fixes/tests locking, use **one download owner across all agents**;
other distillers can work on cached PDFs. Crossref calls also need coordinated
pacing. Keep the existing batch-specific arXiv API restrictions until explicitly
revised by the tooling owner based on the current source-access policy.

## 5. Shared-file and handoff rules

- Shared worktree: no branch switching, broad reset/clean, whole-repo formatting,
  or commits bundling another agent's changes. Use isolated worktrees if desired,
  but claim paper IDs and network access globally even across worktrees.
- One owner per output file. Claim a batch, not an arbitrary overlapping subset
  of paper IDs. Auditors write findings separately; authors apply corrections.
- Only D2 applies canonical merges and regenerates `atlas.json`. Other agents
  may build temporary views for tests; they must not overwrite the shared artifact.
- E1 owns SPEC/runtime/schema reconciliation now. E2/E3 write proposed contract
  changes in their own reports before shared interfaces change. No undocumented
  new convention, field or silent physical default.
- A schema change must update schema, validator, skill, view builder, app types
  and tests as needed in one coordinated tranche. Assign a temporary interface
  owner before implementing it; do not leave consumers half-migrated.
- Source files are read-only inputs during extraction/audit unless that source's
  owning batch is repairing its cache. Public records never contain private paths.
- No solver outputs are committed. DevLogs record gates/pass/fail and limitations.
- Status lifecycle: `unassigned → claimed → in_progress → ready_for_review → complete`;
  `waiting` names a dependency and `blocked` names missing source/input. API failure
  is not completion. Announce when ownership is released or transferred.

Claim template for `coordination/claims/<agent-id>.md`:

```text
Agent:
Task IDs:
Status / updated date:
Exact owned write paths:
Read-only inputs:
Explicit exclusions:
Dependencies / requested interface changes:
Acceptance checks:
Handoff report path:
```

Handoff report must contain task IDs, changed files, implementation/extraction
summary, exact commands and outcomes, unresolved limitations, evidence/contract
changes, and files released to the next owner. Include the base revision/worktree
when working in isolation. Do not claim tests ran if they only appear in a plan.

## 6. codex-main takeover and immediate handoff boundary

I am taking **C0, E1 and U1 only**. Exact files and exclusions are in
[`coordination/claims/codex-main.md`](../coordination/claims/codex-main.md).
Implementation was paused for planning and resumed at the user's request on
2026-10-01. Live progress and verification are in
[DevLog-003](DevLog-003-cross-section-progress.md). No other task is assigned to
this agent by implication.

Delivered implementation (initial commit `ef32d46`, data integration `8029881`,
and the documented follow-up):

- Shared YAML parser, config guards, runner, CLI, input schema and type declarations.
- Twenty engine tests; optical material/resource guards and analytic baseline.
- Sim UI/worker, geometry preview, static config copying, About and scorecard link.
- Five app logic tests, README and engine documentation/license text.
- Incomplete-draft geometry/disclosure inspection with strict solve blocking.

Author acceptance checks completed; exact commands/results are in DevLog-003:

1. Runtime/SPEC/schema reconcile the E1 subset, resource ceilings and honest
   target eligibility; all seven available paper drafts were checked at the boundary.
2. Engine 20/20, app 5/5, Python 28/28, validator 0 errors, Svelte 0 errors/warnings.
3. Root and `/eo-atlas` production builds and Chrome interaction suites pass,
   including actual worker/Node agreement, cancellation and incomplete drafts.
4. Narrow-viewport layout verified automatically and visually.
5. Review handoff delivered; optics released to E2, U1 integration files to U3,
   app data tests/header scope to U2. Shared interfaces still need coordination.

E1/U1 await independent review rather than being marked complete. Chen remains
unvalidated; its metal-intersecting optical window and periodic physics remain
unresolved. Other agents can claim E2, E3, Q2 or U2 within the recorded file
boundaries. D0/D1/D2/Q1 are already owned by the data lane. No further user
permission is required for ordinary local work already within project scope;
file/interface conflicts require agent coordination.
