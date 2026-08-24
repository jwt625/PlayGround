# A/B Graph Build Report

Generated reproducibly from the canonical YAML snapshot. The output contains no build timestamp; input SHA-256 hashes in `graph_ab.json` identify the exact source state.

## Result

| Metric | Count |
|---|---:|
| Canonical organizations | 67 |
| Canonical people | 91 |
| Canonical edges | 158 |
| Canonical person→organization edges | 146 |
| Validated A/B person→organization edges | 141 |
| Person→organization edges removed by A/B/status filter | 5 |
| People in evidence graph | 84 |
| Organizations in evidence graph | 48 |
| People with fewer than two lineage institutions | 43 |
| People eligible for cross-institution sequencing | 41 |
| Person-level ordered contributions | 6 |
| Aggregated institution→institution talent flows | 5 |
| Excluded cross-organization person pairs | 79 |

Evidence grades: `A` 121, `B` 20.

Edge types: `employment` 36, `executive_role` 9, `founder` 47, `publication_affiliation` 1, `research_training` 22, `technical_leadership` 26.

## Ordering policy

A talent flow is generated only when the latest possible date represented by an earlier edge's `end_date` is strictly before the earliest possible date represented by a later edge's `start_date`. Year and month precision are expanded to full intervals. Equal years/months therefore do not establish order. Biography list order, acquisition timing, current-role fields, and undated role language are not used to infer a sequence.

The HTML is a conservative Sankey-like institution-flow view plus a searchable table of every included person→organization evidence edge. Because only 5 institution flows meet the ordering rule, it should be read as a **dated-evidence prototype**, not an ecosystem-completeness map.

## Derived institution flows

- **Alcatel-Lucent Bell Labs → Ciena** — weight 1.0; Peter Winzer
- **Bell Labs → Alcatel-Lucent Bell Labs** — weight 1.0; Robert Tkach
- **Lucent Bell Labs → ETH Zurich Institute of Electromagnetic Fields** — weight 1.0; Jürg Leuthold
- **Massachusetts Institute of Technology → Ayar Labs** — weight 6.0; Milos Popovic, Vladimir Stojanovic
- **MIT Photonics and Modern Electro-Magnetics Group → Lightelligence** — weight 3.0; Yichen Shen

## Exclusions

- `missing_prior_end_or_later_start`: 72
- `overlapping_or_nonconclusive_date_precision`: 7

Full person/pair exclusions and their contributing edge IDs are stored in `graph_ab.json` under `sequence_exclusions`. People with fewer than two distinct lineage institutions never become sequence candidates and are counted separately above. Eligible person→organization relationships outside the configured lineage-bearing types remain visible in the evidence browser but are not used to derive flows.

## Validation

The builder aborts on missing canonical collections, schema-version disagreement, or missing evidence-source references for included edges. After writing, the build was checked by reparsing JSON, checking node/edge/flow references, and confirming that every included person edge is validated with grade A or B.
