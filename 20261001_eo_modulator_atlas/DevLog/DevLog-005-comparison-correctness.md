---
title: EO atlas - U2a comparison correctness
date: 2026-10-01
status: in_progress
owner: codex-main
tasks: [U2a]
---

# DevLog-005: Comparison correctness

User requested another approximately 20-minute work block. Baseline `09f3daa`;
clean worktree at start. E1/U1 remain ready for independent review. Ownership:
[codex-main claim](../coordination/claims/codex-main.md). Data lane still owns
canonical integration and generated views; no subagents launched.

## Plan and categories

| Tranche | Budget | Deliverable | Gate |
|---|---|---|---|
| U2a.1 audit | ~5 min | Trace source qualifiers through metrics, charts and CSV; reproduce concrete failures | Findings distinguish app defects from data-contract follow-ups |
| U2a.2 fixes | ~10 min | Axis-specific omission rules; honest derived bounds, frontier/statistic scope and export context | Targeted synthetic regressions plus actual integrated data |
| U2a.3 verification | ~5 min | Type/unit/build checks and focused browser interactions; handoff | No known regression in baseline smoke; explicit remaining work |

## Progress

- Started 17:23 local. Read current claims and committed data/implementation state.
- Found panel C uses an artificial x=0 then requires positive x and y, dropping
  all material-summary points. Every chart applies that positivity rule even on
  linear axes; toggling log/linear cannot restore valid zero values.
- Found Pareto construction includes bandwidth bounds as exact points. Group
  statistics likewise average bound thresholds as measured values.
- Derived metric qualifier presentation takes the first input qualifier. This
  does not invert denominator bounds for FOM or handle conflicting directions.
  CSV exports qualifiers but omit metric basis, source context and device IDs.

## Checks and handoff

Pending implementation and verification. U2b will cover the remaining
representative-ranking, measured-only semantics, URL/back and mobile usability
work; changes to Python-derived views require coordinated ownership with D2.
