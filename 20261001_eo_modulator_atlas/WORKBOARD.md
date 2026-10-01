# Workboard

Updated 2026-10-01. Detailed scope, dependencies and acceptance checks:
[DevLog-002](DevLog/DevLog-002-work-plan-and-ownership.md).

**Current claim:** `codex-main` (this conversation) owns C0, E1 and U1.
Implementation resumed at the user's request on 2026-10-01. The existing uncommitted
engine/simulator changes belong to that claim and are work in progress. Live notes:
[E1/U1 progress](DevLog/DevLog-003-cross-section-progress.md).
All other tasks below are **unassigned**, not implicitly delegated.

| ID | Tranche | State | Owner | Dependency / handoff |
|---|---|---|---|---|
| C0 | Work plan, claims, current-state record | complete | codex-main | This board + detailed plan + claim file |
| D0 | Ingestion tooling and policy alignment | ready | unassigned | Serialize fetching until locking is tested |
| D1.01–D1.11 | Priority-1 paper batches, one claim per batch | ready, source access varies | unassigned | Existing batch manifests; see cached papers in plan |
| D2 | Canonical data integration and view refresh | waiting for staged batches | unassigned | D1 report + independent evidence review |
| E1 | Cross-section runner, config boundary, analytic baseline | in progress | codex-main | Freeze input/result contracts; finish current checks |
| E2 | Optical model limits, EO tensor overlap and voltage conventions | waiting for E1 handoff | unassigned | May research/design now; runtime edits after release |
| E3 | Uniform RF line, conductor/dielectric loss | design ready; integration waits | unassigned | E1 section outputs; explicit model contract |
| E4 | Periodic loaded line and traveling-wave EO response | waiting | unassigned | E2 + E3; loading/reference-plane contract |
| E5 | Paper regressions and convergence studies | waiting | unassigned | E2–E4 + reviewed paper inputs |
| U1 | Browser cross-section simulator and current app baseline | in progress | codex-main | E1; route, worker, docs and smoke checks |
| U2 | Table/explore correctness and usability | ready in owned files | unassigned | Avoid U1-owned scorecard and app config |
| U3 | Full-chain results and reproduction scorecard | waiting | unassigned | E4/E5 contract + U1 release |
| Q1 | Independent pilot/batch evidence audit | ready | unassigned | Read-only inputs; write audit report only |
| Q2 | Independent numerical review | review ready; final gate waits | unassigned | E1 diff; later E2–E5; separate audit files |
| R1 | Release integration, second audit, public packaging | waiting | unassigned | Accepted D/E/U tranches; rights policy review |

## Claim protocol

1. Read this board, the detailed plan, and every active file under
   `coordination/claims/`. Existing active claims take precedence over a stale
   board row or an older DevLog TODO.
2. Claim an unassigned task in `coordination/claims/<agent-id>.md`, using the
   template in the plan. Name exact write paths and exclusions; announce the
   claim to the coordinating conversation. This is task coordination, not a
   request for new user authorization.
3. Work only within those paths. If another active claim overlaps, resolve the
   ownership conflict before editing the contested files. Do not quietly expand
   scope into shared files.
4. Keep your claim and handoff report current. The coordinator updates this
   board; avoid simultaneous whole-file rewrites by multiple agents.
5. Report `ready_for_review` before `complete`. Completion requires the stated
   acceptance checks and review, not just files on disk or a zero CLI exit code.

No agents have been spawned by this continuation. The user may assign the
unclaimed tranches to other conversations without overlapping this claim.
