# Workboard

Updated 2026-10-01. Detailed scope, dependencies and acceptance checks:
[DevLog-002](DevLog/DevLog-002-work-plan-and-ownership.md).

**Current claims:** `codex-main` holds C0, the E1 contract and U2a comparison correctness; E1/U1 are
`ready_for_review` with successor file releases recorded in its claim.
`claude-data-lane` owns D0,
D1.01–D1.11, D2 and Q1. Progress: [E1/U1](DevLog/DevLog-003-cross-section-progress.md)
and [data lane](DevLog/DevLog-004-data-lane-progress.md). Current canonical snapshot:
8 papers / 16 devices / 19 organizations after p1_01. Other tranches remain
unassigned. No implicit delegation.

| ID | Tranche | State | Owner | Dependency / handoff |
|---|---|---|---|---|
| C0 | Work plan, claims, current-state record | complete | codex-main | This board + detailed plan + claim file |
| D0 | Ingestion tooling and policy alignment | implementation reported complete | claude-data-lane | DevLog-004 records lock/breaker tests; single prefetch owner retained |
| D1.01–D1.11 | Priority-1 paper batches, one claim per batch | in progress; p1_01 integrated, p1_02/03/09 reports present | claude-data-lane | Max 3 concurrent; batch-specific Q1 corrections before D2 |
| D2 | Canonical data integration and view refresh | in progress; p1_01 integrated | claude-data-lane | Serial merges; p1_09 audit corrections remain outstanding |
| E1 | Cross-section runner, config boundary, analytic baseline | ready_for_review | codex-main | Author gates pass; Q2 independent audit pending; DevLog-003 handoff |
| E2 | Optical model limits, EO tensor overlap and voltage conventions | ready to claim | unassigned | E1 handoff available; optics.mjs released; coordinate shared interfaces |
| E3 | Uniform RF line, conductor/dielectric loss | design ready; integration waits | unassigned | E1 section outputs; explicit model contract |
| E4 | Periodic loaded line and traveling-wave EO response | waiting | unassigned | E2 + E3; loading/reference-plane contract |
| E5 | Paper regressions and convergence studies | waiting | unassigned | E2–E4 + reviewed paper inputs |
| U1 | Browser cross-section simulator and current app baseline | ready_for_review | codex-main | Root/base-path Chrome and narrow viewport pass; incomplete drafts preview safely |
| U2a | Comparison correctness: bounds, plot validity, export context | in progress | codex-main | Approximately 20-minute block; DevLog-005 plan and exact file claim |
| U2b | Remaining representative/filter/navigation and usability audit | waiting for U2a file release | unassigned | Coordinate generated-view contracts with D2; no overlapping app writes |
| U3 | Full-chain results and reproduction scorecard | waiting | unassigned | U1 files released; still needs E4/E5 contract |
| Q1 | Independent pilot/batch evidence audit | claimed (fresh-context auditors, read-only) | claude-data-lane | Read-only inputs; write audit report only |
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
The [E1/U1 follow-up register](DevLog/DevLog-003-cross-section-progress.md#cross-agent-review-and-follow-up)
records outstanding p1_09 provenance corrections and batch interface proposals.
