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
| Phase 0 schema and seeds | schema_foundation agent | Complete; validator in progress | `schema.yaml`, `edge_types.yaml`, `source_policy.md`, seed YAMLs |
| Phase 1 priority acquisitions | acquisition_research agent | Initial tranche complete | `research/acquisitions_initial.yaml`, briefing |
| Phase 2 Intel SiPh first tranche | intel_diaspora agent | Canonical initial tranche complete | `research/intel_siph_people_initial.yaml`, briefing |
| Bell Labs / coherent optics | acquisition_research agent | Initial tranche complete | 16 people, 27 edges, briefing |
| UCSB / Bowers | schema_foundation agent | Initial tranche complete | 18 people, 36 edges, briefing |
| MIT / Columbia ecosystem | acquisition_research agent | Initial tranche complete | 20 people, 33 edges, briefing |
| Legacy-company diaspora | acquisition_research agent | In progress | Legacy people dataset and briefing |
| M1 integrity tooling | schema_foundation agent | Complete | `scripts/validate_data.rb`, usage notes |
| Canonical data merge | intel_diaspora agent | MIT/Columbia merge complete | `canonical/*.yaml`, merge notes |
| First A/B graph build | schema_foundation agent | Complete (strict dated-evidence prototype) | graph builder, derived JSON, self-contained HTML, build report |
| China / Huawei diaspora | intel_diaspora agent | In progress | Bilingual China people dataset and briefing |
| Stanford academic lineages | schema_foundation agent | Plan/seed update in progress | Stanford plan section and seed records; research tranche follows |

## Milestones

- [x] M0: schema frozen; seed datasets parse; first acquisition and Intel research tranches reviewed
- [x] M1: canonical merge pipeline and validation checks
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
- Froze schema v0.1 with 101 organization seeds, 47 person seeds, and 15 explicit edge types. All IDs are unique and the YAML parses.
- Completed the initial 10-target acquisition tranche with explicit unresolved fields rather than inferred values.
- Spot-audited GlobalFoundries' 2025 filing: it directly supports InfiniLink's 2025-11-14 close at $48M and AMF's 2025-11-18 close for $453M cash.
- Corrected the Intel staging tranche's source grading: the Optica team roster is B-grade professional-society evidence under the project policy, not A-grade primary evidence.
- Started the Bell Labs / coherent-optics workstream while M1 integrity tooling proceeds in parallel.
- Completed Intel's initial tranche with 20 people and 37 atomic edges: 21 A-grade and 16 B-grade.
- Added and independently reran a Ruby-stdlib integrity validator; all 101 organization seeds, 47 person seeds, and 15 edge types pass.
- Started the first schema-compliant canonical merge and the UCSB / Bowers evidence workstream.
- Completed Bell Labs/coherent initial research: 16 people and 27 A/B atomic edges. Unsupported Mintera→Acacia and Bell Labs→Nubis group transfers remain explicitly excluded.
- Completed UCSB/Bowers initial research: 18 people and 36 A-grade atomic edges, with UCSB attendance kept distinct from specific Bowers-group membership and Aurrion employment kept distinct from founding.
- Started MIT/Columbia photonic-compute and optical-I/O research.
- Completed the first canonical merge: 49 sources, 46 organizations, 43 people, 10 acquisition events, and 70 atomic edges. Evidence mix is 49 A / 17 B / 4 C; the four C-grade edges remain asserted and are excluded from the default graph.
- Canonical merge validation found zero duplicate IDs, orphan references, missing base fields, or missing edge-type attributes. The ambiguous grouped Finisar/Coherent employment claim was excluded.
- Began merging the completed Bell Labs and UCSB tranches and extending the reusable validator across canonical sources, events, and edges.
- Completed MIT/Columbia initial research: 20 people, 33 atomic edges, and 13 A-grade sources. University degrees remain distinct from evidenced PI-group training, and two tempting but unsupported founder claims are recorded as non-edges.
- Started the legacy commercial-company diaspora workstream.
- Merged Bell Labs and UCSB into canonical data: 77 sources, 58 organizations, 72 people, 10 events, and 126 edges. Seven exact relationship duplicates were skipped.
- Extended canonical validation across sources, events, endpoint types, evidence/status compatibility, required attributes, and chronology. It passes with zero errors and one documented InfiniLink warning (the first public item found post-dates close).
- Started the MIT/Columbia canonical merge and a reproducible A/B-only graph prototype.
- Merged MIT/Columbia into canonical data: 90 sources, 67 organizations, 91 people, 10 events, and 158 edges. All 32 new edges are A-grade/validated; explicit non-edges remain excluded.
- Started the China/Huawei/overseas-returnee workstream with individual-name validation as a hard gate.
- Built the first deterministic A/B-only graph artifacts. The evidence browser contains 141 validated person→organization edges across 84 people and 48 organizations.
- The strict dated-flow view contains only 5 institution flows / 6 person contributions; 79 possible cross-organization pairs are excluded because ordering is missing or non-conclusive. This sparsity is disclosed in the HTML and build report rather than masked with inferred direction.
- Independently reran canonical validation and the graph builder successfully against the 91-person / 67-organization / 158-edge snapshot.
- Added Stanford to scope as a first-class academic lineage spanning the Fan, Vučković, Solgaard, Miller, Jim Harris, Steve Harris, Byer, and Fejer schools; plan and discovery seeds are being updated before research begins.

## Open issues

- The exact rendering stack is intentionally deferred until canonical schema and initial data reveal graph size and browser-performance requirements.
- LinkedIn-only career claims may be captured as grade C but require corroboration before default visualization.
- Several 2025–2026 acquisitions may have announced but not closed; status must be verified from current primary sources.

## Next coordinator actions

1. Merge Bell Labs and UCSB staging records into canonical datasets.
2. Complete automated validation of canonical sources, events, and edges.
3. Finish MIT/Columbia research and launch legacy-company and China workstreams.
4. Derive the first A/B-only aggregate flow set and render a graph prototype.
