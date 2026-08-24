# Canonical Validation and Release Report

Generated deterministically from canonical YAML and `generated/graph_ab.json`. This report does not run the standalone validator; it performs release-oriented consistency and freshness checks and should be paired with `ruby scripts/validate_data.rb --canonical canonical`.

## Snapshot

| Collection | Count |
|---|---|
| Organizations | 172 |
| People | 288 |
| Sources | 282 |
| Events | 23 |
| Edges | 556 |

Schema version: `0.1.0`. Graph input hashes current: **yes**.
Canonical duplicate IDs: **0**. Duplicate source URLs: **0**.

## Evidence

| Grade | Sources | Edges |
|---|---|---|
| A | 253 | 491 |
| B | 27 | 60 |
| C | 2 | 5 |

Edge statuses: `asserted` 5, `validated` 551.

## Edge types

| Edge type | Count |
|---|---|
| acquisition | 21 |
| advisor | 1 |
| collaboration | 3 |
| employment | 121 |
| executive_role | 46 |
| founder | 117 |
| patent_affiliation | 5 |
| publication_affiliation | 23 |
| research_training | 83 |
| spinout | 10 |
| technical_leadership | 126 |

## Default validated A/B graph

| Metric | Count |
|---|---|
| Canonical person→organization edges | 523 |
| Included validated A/B edges | 518 |
| Excluded by grade/status | 5 |
| People in graph | 282 |
| Organizations in graph | 153 |
| Ordered person contributions | 60 |
| Institution flows | 52 |
| Sequence exclusions | 255 |

Coverage: 97.9% of canonical people and 89.0% of canonical organizations appear in the default graph.

Sequence exclusions: `missing_prior_end_or_later_start` 214, `overlapping_or_nonconclusive_date_precision` 41.

## Acquisition completeness

| Metric | Count |
|---|---|
| Canonical acquisition events | 21 |
| Canonical acquisition edges | 21 |
| Edges linked to canonical events | 21 |
| Events with announced date | 19 |
| Events with effective date | 19 |
| Events with disclosed value | 18 |
| Events with ≥2 participants | 21 |
| Events with A/B evidence | 21 |

Missing announced date: `event_acquisition_aurrion_juniper_2016`, `event_caliopa_acquisition_huawei_2013`.

Missing effective date: `event_acquisition_enosemi_amd`, `event_acquisition_polariton_marvell`.

Missing disclosed value: `event_acquisition_elenion_nokia_2020`, `event_acquisition_enosemi_amd`, `event_acquisition_polariton_marvell`.

## Conflicts, warnings, and limitations

Named open or preserved conflicts: `conflict_caliopa_formation_year`, `conflict_polariton_seed_technology_tfln`, `conflict_xiaoguang_tu_ime_inphi_overlap`.

- `event_acquisition_infinilink_globalfoundries`: announced_date is after effective_date; inspect event notes for a publication-proxy explanation.

- The default graph includes only validated grade-A/B person-to-organization edges; lower-grade and asserted claims remain canonical but are excluded.
- Institution flows require strict date ordering. Missing or overlapping date precision produces explicit sequence exclusions rather than inferred order.
- Canonical conflicts and negative claims are currently narrative in MERGE_NOTES.md; only explicitly named conflict IDs and section counts are machine-surfaced.
- Acquisition completeness reports disclosure, not an assumption that unknown effective dates or values are zero.

MERGE_NOTES contains 10 explicitly labeled non-edge/excluded-claim sections. Negative claims remain narrative until a canonical structured conflicts/non-edges collection is introduced.

## Definition of done

- [x] **canonical_collections_loaded** (`pass`): Five canonical YAML collections parsed with schema 0.1.0.
- [x] **canonical_ids_unique** (`pass`): No duplicate canonical IDs found.
- [x] **canonical_source_urls_unique** (`pass`): No duplicate canonical source URLs found.
- [x] **graph_matches_canonical_inputs** (`pass`): All graph input SHA-256 hashes match.
- [x] **default_graph_is_validated_ab** (`pass`): Graph evidence filter is {"statuses"=>["validated"], "grades"=>["A", "B"]}.
- [x] **acquisition_events_and_edges_linked** (`pass`): 21 of 21 acquisition edges link to canonical acquisition events.
- [ ] **open_conflicts_disclosed** (`warn`): 3 explicitly named conflicts remain open or preserved.
- [x] **graph_exclusions_disclosed** (`pass`): 255 sequence exclusions are reported by reason.
- [x] **deterministic_generation** (`pass`): Reports contain no runtime timestamp and use sorted IDs/count keys plus canonical input hashes.

## Input fingerprints

| Input | SHA-256 | Matches graph |
|---|---|---|
| organizations | `86a015c2459568615be61ee20e602be5a4db2e2ae63fe51070f6c7f088af9f12` | yes |
| people | `38bf33ceafff3f2da68928277057fc935dd4717fa53dd4178116bc63d24ee628` | yes |
| sources | `bb9f11f662540a262e702e5b4c68ea4e3959cb7744b997eaa9696323779deba8` | yes |
| events | `a5e6520ddc2d603e2b6c73831358b300103c0712866308852f024e6ba5a1f88e` | yes |
| edges | `a031f6561ac2a4230798293ea11a8983d21d877a908610422d87a0848fad3a5c` | yes |
