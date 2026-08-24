# Photonics Lineage Execution Tracker

Last updated: 2026-08-23

## Target

Execute the first major release defined in `photonics_lineage_research_plan.md`: a regenerable, evidence-backed person-level genealogy with at least 100 organizations/labs and 200 validated people, plus A/B-only lineage visualizations.

## Operating decisions

- The planning document remains the requirements source; execution status lives here.
- Research is merged only when each material claim has a direct source URL and evidence grade.
- A/B evidence is eligible for the default graph. C evidence is retained but excluded by default. D/X claims remain unresolved and must not silently become graph edges.
- Person-level records are canonical. Institution-level talent flows are derived artifacts.
- Dates use ISO `YYYY-MM-DD` when known and a year-only field when the evidence is not more precise.
- Unknown values remain explicit nulls; they are never inferred merely to complete a record.
- Research workstreams write isolated initial datasets before coordinator review and canonical merge.

## Current milestone: M0 — foundation and first evidence tranche

| Workstream | Owner | Status | Deliverables |
|---|---|---|---|
| Coordination and tracker | root | In progress | `PROGRESS.md`, integration review |
| Phase 0 schema and seeds | schema_foundation agent | In progress | `schema.yaml`, `edge_types.yaml`, `source_policy.md`, seed YAMLs |
| Phase 1 priority acquisitions | acquisition_research agent | In progress | `research/acquisitions_initial.yaml`, briefing |
| Phase 2 Intel SiPh first tranche | intel_diaspora agent | In progress | `research/intel_siph_people_initial.yaml`, briefing |

## Milestones

- [ ] M0: schema frozen; seed datasets parse; first acquisition and Intel research tranches reviewed
- [ ] M1: canonical merge pipeline and validation checks
- [ ] M2: Bell Labs/coherent, UCSB, MIT/Columbia, legacy-company, and China workstreams
- [ ] M3: 100 organizations/labs and 200 validated people with conflict resolution
- [ ] M4: first A/B-only Sankey and person-level interactive graph
- [ ] M5: research briefing, lineage memos, validation report, and release audit

## Validation gates

- All YAML/JSON parses successfully.
- IDs are stable, unique, and referentially valid.
- Every employment/training/founder/leadership claim has at least one source.
- Source grade is attached to the supported claim, not only to a person or organization globally.
- Acquisitions distinguish announcement date, close date, headline value, and contingent consideration.
- Aggregate flows retain the underlying named people and weight contributions.
- Organization aliases and corporate-successor relationships do not erase historically distinct teams.

## Execution log

### 2026-08-23

- Read the complete 1,364-line research plan.
- Found no blocking ambiguity; adopted the plan's first-major-release definition as the target.
- Started three isolated parallel workstreams: schema/foundation, priority acquisitions, and Intel SiPh diaspora.
- Noted that the Git worktree contains unrelated untracked sibling projects; they are out of scope and will remain untouched.

## Open issues

- The exact rendering stack is intentionally deferred until canonical schema and initial data reveal graph size and browser-performance requirements.
- LinkedIn-only career claims may be captured as grade C but require corroboration before default visualization.
- Several 2025–2026 acquisitions may have announced but not closed; status must be verified from current primary sources.

## Next coordinator actions

1. Review schema semantics and parseability.
2. Audit a sample of acquisition and Intel claims against cited sources.
3. Reconcile workstream formats into canonical datasets.
4. Add automated integrity checks before launching the next research wave.
