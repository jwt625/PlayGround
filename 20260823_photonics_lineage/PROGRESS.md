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
| Legacy-company diaspora | acquisition_research agent | Initial tranche merged | 25 people, 39 staging edges, briefing |
| M1 integrity tooling | schema_foundation agent | Complete | `scripts/validate_data.rb`, usage notes |
| Canonical data merge | intel_diaspora agent | MIT/Columbia merge complete | `canonical/*.yaml`, merge notes |
| First A/B graph build | schema_foundation agent | Complete (strict dated-evidence prototype) | graph builder, derived JSON, self-contained HTML, build report |
| China / Huawei diaspora | intel_diaspora agent | Initial tranche merged | 20 people, 25 edges, bilingual briefing |
| Stanford academic lineages | acquisition_research agent | Initial tranche merged | 21 people, 31 A-grade edges, 24 primary sources |
| Temporal evidence recovery | schema_foundation agent | Complete and applied | 10 dated-edge updates, impact report |
| imec / Ghent ecosystem | acquisition_research agent | Initial tranche merged | 23 people, 49 A/B edges, Caliopa acquisition |
| Caltech / Scherer / Luxtera | intel_diaspora agent | Initial tranche merged | 20 people, 39 A/B staging edges, 18 sources |
| UK / Southampton / Bookham / Rockley | acquisition_research agent | Initial tranche merged | 20 people, 36 A/B staging edges, 2 events |
| A*STAR IME / AMF | intel_diaspora agent | Initial tranche merged | 20 people, 36 A-grade staging edges, formal spinout |
| ETH / Leuthold / Polariton | acquisition_research agent | Initial tranche merged | 20 people, 37 A/B staging edges, 1 event |
| Recent active startups | intel_diaspora agent | Initial tranche merged | 29 people, 62 A-grade edges, 4 spinouts |
| Release coverage audit | acquisition_research agent | Complete | 88-seed completeness matrix and ranked top-30 gaps |
| Historical acquisitions gap | acquisition_research agent | Complete and merged | 10 acquisition events, 25 A-grade sources |
| China incumbent companies | intel_diaspora agent | Initial tranche merged | 20 people, 60 A/B edges, 12 sources |
| Reproducible release reports | schema_foundation agent | Complete | Validation report and metrics generator |
| Priority gap closure | acquisition_research agent | Paused for visualization correction | Next 10 uncovered high-value claims |
| Sankey replacement | intel_diaspora agent | Complete | Python/Plotly Sankey, PNG, interactive HTML |
| Institution dedup audit | acquisition_research agent | Complete | Manual core-node audit and 11 display clusters |
| China incumbent expansion | intel_diaspora agent | Paused; TODO only | TFC, AFR/光库科技, InnovSemi/易缆微, SUNA/苏纳光电 |
| Jim Harris diaspora depth pass | acquisition_research agent | Paused; TODO only | Zongjian→Apple, Hong Liu→Google, Yijie Huo→Vertilite, full alumni roster |
| Visualization optimization | root | Next priority | Interactive graph and Sankey UX/layout |

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
- Completed the Stanford plan/seed update across primary questions, lab scope, Phase 4, visualization outputs, Workstream H, methods, briefing, final deliverables, and immediate actions. Added Stanford plus eight group nodes and all eight requested people as explicitly unverified discovery seeds.
- Seed validation now passes with 110 organizations, 55 people, and 15 edge types. A dedicated Stanford evidence tranche is in progress.
- Completed the legacy-company staging tranche with 25 people, 39 edges, and 17 A/B sources; no blanket acquisition-based employee transfers were asserted.
- Completed the China staging tranche with 20 people, 25 A/B edges, 25 sources, and 5 explicit discovery non-edges; generic team-origin claims remain excluded.
- Started canonical integration of both completed tranches and a targeted A/B temporal-evidence pass against the graph's 79 excluded organization pairs.
- Merged legacy and China/Huawei staging data into canonical form. Canonical counts are now 127 sources, 82 organizations, 129 people, 10 events, and 213 edges; validation passes with the same single documented InfiniLink warning.
- Completed Stanford staging research with 21 people, 31 A-grade edges, and 24 primary sources; canonical integration is in progress.
- Started imec/Ghent/Caliopa/Luceda/Huawei Belgium research.
- Merged Stanford into canonical data and rebuilt the graph. Canonical counts are now 151 sources, 94 organizations, 149 people, 10 events, and 244 edges; validation passes.
- The rebuilt interactive evidence graph contains 227 A/B edges across 143 people and 79 organizations, with 9 strictly dated institutional flows.
- Completed a temporal-evidence patch with 10 proposed date updates from 9 A/B sources; canonical application is in progress. The initial impact simulation recovers Intel SiPh→ANELLO, Bell Labs→Nubis, and UCSB Bowers→Nexus flows.
- Started Caltech/Scherer/Luxtera research.
- Completed imec/Ghent/Caliopa/Luceda/Huawei Belgium staging research: 23 people, 49 A/B edges, all six evidenced Luceda founders, and the EUR 7 million Caliopa acquisition. Canonical merge is in progress.
- Applied all 10 temporal updates. Strictly ordered person contributions increased from 10 to 14, aggregate institution flows from 9 to 12, and exclusions fell from 108 to 104; validation and graph rebuild pass.
- Started the Southampton/Surrey/Bookham/Rockley UK lineage workstream.
- Merged imec/Ghent into canonical data, including the EUR 7 million Caliopa→Huawei acquisition while preserving its unresolved 2009-versus-2010 formation conflict. Canonical counts reached 100 organizations, 172 people, 168 sources, 11 events, and 293 edges.
- Rebuilt the graph to 271 A/B person-organization edges, 166 visible people, 85 visible organizations, and 15 strictly ordered institution flows.
- Completed Caltech/Scherer/Luxtera staging research with 20 people and 39 A/B edges; canonical merge is in progress.
- Started A*STAR IME/AMF/Singapore foundry research.
- Merged Caltech/Scherer/Luxtera into canonical data. Counts are now 103 organizations, 189 people, 184 sources, 11 events, and 329 edges; validation passes and the graph has 307 A/B edges with 17 strict institutional flows.
- Completed UK/Southampton/Bookham/Rockley staging research with 20 people, 36 A/B edges, and 2 events; canonical merge is in progress.
- Started ETH/Leuthold/Polariton research.
- Merged the UK tranche. Canonical counts reached 111 organizations, 206 people, 202 sources, 13 events, and 360 edges, passing the plan's top-level organization and people count thresholds.
- The rebuilt graph now contains 336 A/B person-organization edges across 200 visible people and 95 visible organizations, with 19 strict institutional flows.
- Completed A*STAR IME/AMF staging research with 20 people, 36 A-grade edges, and one formal spinout; canonical merge is in progress.
- Started coverage of eight recent active-startup seeds whose founders and technical leadership remain thin or absent.
- Merged A*STAR/IME/AMF data. Canonical counts are now 113 organizations, 225 people, 210 sources, 13 events, and 394 edges; Guo-Qiang/Patrick Lo was resolved as one identity and no blanket process/team transfer was inferred.
- The graph rebuild now contains 370 A/B edges across 219 people and 98 visible organizations, with 23 strict institutional flows.
- Completed ETH/Leuthold/Polariton staging research with 20 people and 37 A/B edges; canonical merge is in progress.
- Started a plan-wide release coverage audit to rank missing founder, technical-leadership, upstream-affiliation, acquisition, and date claims.
- Merged ETH/Leuthold/Polariton into canonical data. Counts are now 117 organizations, 240 people, 223 sources, 13 events, and 424 edges; the graph contains 400 A/B edges and 30 strict flows.
- Completed the recent-active-startup staging tranche with 29 people, 62 A-grade edges, four sourced spinout/lab-origin edges, and eight explicit non-edges; canonical merge is in progress.
- Completed a reproducible audit of 88 major seeds. It found 26/54 company seeds with founder coverage, 12/54 with CTO/technical-leadership coverage, and 10/20 planned acquisitions currently canonical.
- Started research for the 10 missing historical acquisitions and a focused China-incumbent leadership tranche.
- Merged eight recent-startup lineages. Canonical counts are now 141 organizations, 268 people, 245 sources, 13 events, and 486 edges; the graph contains 458 A/B edges across 262 people and 125 organizations.
- Started a deterministic validation/metrics report generator so subsequent merges can refresh release status without manual recounting.
- Completed deterministic release-report tooling and generated validation/metrics reports. Current audit surfaces 3 named conflicts, 1 chronology warning, 207 graph sequence exclusions, and no duplicate IDs or source URLs.
- Completed research for all 10 missing historical acquisitions with 25 A-grade sources and explicit announced-versus-closing valuation bases; canonical merge is in progress.
- Started a focused closure pass on the next 10 highest-value gaps not already covered by the active China-incumbent workstream.
- Merged all 10 historical acquisitions. Canonical now has 145 organizations, 268 people, 270 sources, 23 events, and 496 edges; acquisition coverage is 21 events, 19 announcement dates, 19 effective dates, and 18 disclosed values.
- Completed the China-incumbent staging tranche with 20 people and 60 A/B edges across InnoLight, Eoptolink, Accelink, Source Photonics, Broadex, and Hisense Broadband; canonical merge is in progress.
- Started the planned China-lineage synthesis memo from the validated datasets.
- Rejected the initial hand-built SVG flow sketch as the final visualization. Started a Python/Plotly Sankey replacement with real layout, founder-lineage and strict-chronology modes, named-person hover provenance, and acquisition exploration.
- Started a manual audit of core institutional identities. Time-aware canonical entities such as Bell Labs, Lucent Bell Labs, Alcatel-Lucent Bell Labs, and Nokia Bell Labs will remain distinct in source data but may collapse into a documented Sankey display cluster; true duplicates will be resolved separately.
- Completed the manual core-institution audit. The map contains 11 non-overlapping clusters over 36 canonical organization IDs; Bell Labs eras collapse only for talent display, while canonical employment eras and acquisition counterparties remain intact.
- Replaced the hand-built SVG with a Python/Plotly Sankey. The self-contained HTML has founder-lineage and strict-chronology modes, Plotly interaction, named-person/evidence hover, a searchable evidence table, and an acquisition table. Founder mode has 63 flows / 104 deduplicated contributions; strict mode has 52 flows / 60 contributions.
- Archived the rejected hand-built HTML and generated a 2250×1500 PNG from the Plotly figure.
- Merged the China-incumbent tranche. Canonical totals are now 172 organizations, 288 people, 282 sources, 23 events, and 556 edges; validation and deterministic report generation pass.
- Recorded four additional China-incumbent targets as TODOs only: TFC Optical, AFR/光库科技, InnovSemi/易缆微, and SUNA/苏纳光电. No records from the interrupted follow-up research were ingested.
- Recorded a required Jim Harris-group depth correction as TODO only: validate Zongjian at Apple, Hong Liu at Google, Yijie Huo as a Vertilite co-founder, and systematically mine the full alumni roster. The existing two-alumnus sample is explicitly considered incomplete.
- Stopped all research and ingestion at the user's request. The canonical snapshot remains frozen at 172 organizations, 288 people, 282 sources, 23 events, and 556 edges while visualization optimization becomes the active priority.

## Open issues

- Visualization usability now takes priority over further dataset expansion; layout, filtering, labeling, and exploration behavior need optimization against the frozen canonical snapshot.
- LinkedIn-only career claims may be captured as grade C but require corroboration before default visualization.
- Several 2025–2026 acquisitions may have announced but not closed; status must be verified from current primary sources.
- Stanford Harris-group coverage is known incomplete; the named Apple/Google/Vertilite leads and full-roster pass remain TODOs and are not canonical claims.
- The four newly requested China incumbents remain TODOs; preliminary identity resolution was not ingested.

## Next coordinator actions

1. Optimize the interactive visualization using the frozen canonical snapshot.
2. Improve Sankey layout, labeling, filters, hover evidence, and institution-cluster controls based on user feedback.
3. Keep all research, source ingestion, and canonical-data expansion paused until explicitly resumed.
4. When research resumes, execute the Jim Harris depth correction and four-company China-incumbent TODOs before lower-priority expansion.
