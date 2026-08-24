# Canonical merge notes

Date: 2026-08-23  
Schema: `schema.yaml` version 0.1.0  
Inputs: `research/acquisitions_initial.yaml` and `research/intel_siph_people_initial.yaml`

## Merge result

| Collection | Count |
|---|---:|
| Sources | 90 |
| Organizations | 67 |
| People | 91 |
| Acquisition events | 10 |
| Atomic edges | 158 |

Edge composition:

| Edge type | Count |
|---|---:|
| acquisition | 10 |
| founder | 50 |
| technical_leadership | 27 |
| employment | 37 |
| executive_role | 9 |
| spinout | 2 |
| publication_affiliation | 1 |
| research_training | 22 |

Evidence/status composition:

- 133 grade-A edges, 20 grade-B edges, and 5 grade-C edges.
- 153 edges are `validated`; the five C-grade edges are retained as `asserted` and are excluded from the default graph.
- The C-grade edges are Ahmed F. Aboul-Ella → InfiniLink, Botros George → InfiniLink, Preet Virk → Celestial AI, Phil Winterbottom → Celestial AI, and Phil Winterbottom → Bell Labs.

## Normalization decisions

- All canonical entity, source, event, and edge IDs use the prefixes required by `schema.yaml`.
- Acquisition dates and values were moved from nested staging records into ten event records. Each target→acquirer relationship is also represented as one atomic `acquisition` edge referencing its event.
- Enosemi's and Polariton's effective dates remain null. Their official announcements say the transaction/team transfer was completed but do not establish an exact legal closing date.
- InfiniLink's `announced_date` preserves the staging date but is explicitly labeled as the first dated public item found, which post-dates the filing-established close.
- Dollar figures were converted from USD millions to numeric USD amounts. Transaction-basis caveats remain in event notes and `stock_component` text.
- The Intel Photonics Technology Lab and Intel Silicon Photonics Solutions Group are temporarily represented by one time-evolving business-unit node. This preserves the staging file's common lineage node, but a later organization-history pass may split them.
- Optica's Intel team roster remains grade B. The primary 2007 paper affiliation, official company biographies, filings, releases, and official conference biographies remain A where the staging evidence supports that classification.
- `src_sukna_infinilink` and `src_engine_celestial_ai` were downgraded from staging grade B to grade C because the frozen policy classifies startup databases/investor portfolio pages as weak evidence. Dependent edges are `asserted`, not `validated`.
- Mutable pages have null archive URLs/content hashes because the staging work did not capture archives or hashes.

## Claims deliberately excluded from this first merge

The coordinator requested a complete, internally valid checkpoint over exhaustive mapping. These staging claims were not promoted to canonical edges:

- Juthika Basak → “Finisar / Coherent”: the staging source groups two historically distinct employers and supplies no transition dates. Both organizations are retained as seeds, but no grouped career edge was created.
- Ahmed F. Aboul-Ella → Mixel Egypt: only C-grade evidence and an imprecise historical role.
- Guo-Qiang Lo → A*STAR IME and IDT: valid leads, but the long IME role string contains multiple promotions/periods and should be split after date recovery.
- Loi Nguyen → Cornell/Lester Eastman and Honeywell; Ford Tamer → Broadcom and Agere/Lucent: these require careful `research_training` versus employment/executive decomposition and time-aware organization naming.
- David Lazovsky → Intermolecular/Khosla Ventures and post-close Marvell; individual Celestial AI histories for Preet Virk and Phil Winterbottom beyond the retained C-grade target roles.
- Ben Rubovitch's DustPhotonics founder/board claims, Avigdor Willenz's chairman edge, Ronnen Lovinger's board edge, and other DustPhotonics upstream histories. Ronnen Lovinger's CEO and post-close Credo roles are included.
- Guilhem de Valicourt → “IPG Photonics / Lumentum” remains excluded because the grouped employer claim must not be collapsed. The Bell tranche subsequently resolved Peter Winzer and Guilhem de Valicourt's Bell Labs edges, Dan Harding's Broadcom edge, and Peter Winzer's post-close Ciena role.
- Enosemi founders' upstream Luminous, Elenion/Nokia, Ayar, A*STAR/IME, University of Washington, and Gennum/Semtech paths, plus Matt Streshinsky's post-close AMD role. The formal company acquisition and three Enosemi founder edges are included.
- Individual ETH/Leuthold research-training edges for the Polariton founders. The three founder edges and formal ETH Institute → Polariton spinout are included.

The associated people and organizations may exist as canonical seed entities even when their more complex edges are deferred; this avoids inventing simplified relationships.

## Staging/schema mismatches encountered

1. Staging IDs lacked entity prefixes and staging edges named organizations directly rather than referencing organization IDs.
2. Staging evidence used direct URL arrays; canonical evidence requires stable `src_` records and claim-specific source references.
3. Staging dates were scalar strings; canonical dates require `{value, precision}` objects.
4. Acquisition staging combined event, technology, people, integration, and unresolved claims in one object. Canonical form separates event facts, organization entities, people, and atomic edges.
5. Some staging roles combine founder and executive semantics. The first merge generally uses a founder edge with title metadata, avoiding duplicate simultaneous role edges unless a separately meaningful downstream executive edge is needed.
6. Some acquisition values are announcement values, accounting consideration, or maximum contingent value rather than comparable purchase prices. These distinctions remain explicit and must not be used as homogeneous Sankey weights.
7. The staging files lacked archive URLs, content hashes, exact locators for several sources, and many employment dates. Canonical fields remain null instead of being inferred.

## Validation performed

- All five canonical YAML files parse with YAML aliases enabled.
- No duplicate source, organization, person, event, or edge IDs.
- All edge endpoints resolve to canonical people or organizations.
- All evidence references resolve to canonical sources.
- All acquisition edges resolve to canonical events.
- All organization parents and event participants resolve.
- Every edge contains the schema-required base fields and edge-type-required attributes.

No staging file, `PROGRESS.md`, or script was modified during this merge.

## Bell Labs and UCSB merge

Inputs: `research/bell_labs_people_initial.yaml` and `research/ucsb_bowers_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 49 | 28 | 77 |
| Organizations | 46 | 12 | 58 |
| People | 43 | 29 | 72 |
| Events | 10 | 0 | 10 |
| Edges | 70 | 56 | 126 |

The Bell tranche contributed 12 new people and 23 new edges. The UCSB tranche contributed 17 new people and 33 new edges. Added edge evidence comprises 52 grade A, 3 grade B, and 1 grade C.

### Deduplication and remapping

- Reused canonical people: Peter Winzer, Guilhem de Valicourt, Dan Harding, Phil Winterbottom, and Brian Koch.
- Reused canonical organizations: Nubis Communications (`org_nubis`), Juniper Networks (`org_juniper`), Aurrion, Quintessent, Ciena, Celestial AI, generic historical Bell Labs, and ETH Zurich's Institute of Electromagnetic Fields (`org_eth_ief`).
- Added separate time-aware organization nodes for Lucent Bell Labs, Alcatel-Lucent Bell Labs, and Nokia Bell Labs. These were not collapsed into generic Bell Labs.
- Reused source artifacts by URL for the Ciena Nubis announcement, Ciena transaction deck, Engine Ventures Celestial AI page, and Quintessent team page.
- The Bell staging ID `src_optica_winzer_2026` collided with an existing source ID for a different Optica artifact. The new official speaker biography was added as `src_optica_winzer_speaker_2026`; the existing source record was preserved.
- Seven exact relationship duplicates were not added: Peter Winzer → Nubis founder, Guilhem de Valicourt → Nubis founder, Dan Harding → Nubis CEO, Phil Winterbottom → Celestial AI CTO, and Brian Koch → Quintessent/Aurrion/Juniper.

### Exclusions and conservative grading

- The Bell staging source `src_engine_celestial_profile_2026` is the same Engine Ventures URL already classified canonically as `startup_database`, grade C. It was not duplicated or upgraded. Phil Winterbottom's newly added Bell Labs edge was downgraded from staging B/validated to canonical C/asserted.
- Guilhem de Valicourt's grouped “IPG Photonics / Lumentum” history remains excluded; neither employer receives a fabricated atomic edge.
- UCSB alumni-roster claims were used only for Bowers-group affiliation/training, as staged. They were not expanded into unsupported degree dates or downstream employment.
- No Aurrion → Juniper team-transfer edge was created. The staged annual report establishes acquisition and employee awards but does not name a transferred team.

### Post-merge validation

The full canonical dataset was rechecked for YAML syntax, unique IDs, source references, entity endpoints, organization parents/predecessors/successors, event participants, event references, advisor/founder person references, edge-type-required attributes, and evidence-grade/status mapping.

## MIT and Columbia merge

Input: `research/mit_columbia_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 77 | 13 | 90 |
| Organizations | 58 | 9 | 67 |
| People | 72 | 19 | 91 |
| Events | 10 | 0 | 10 |
| Edges | 126 | 32 | 158 |

All 32 added edges are grade A and validated. They comprise 11 research-training, 14 founder, 3 employment, and 4 technical-leadership edges.

### Deduplication and normalization

- Reused `person_vivek_raghunathan`, `org_ayar_labs`, and canonical `org_xscape` rather than creating staging's duplicate `org_xscape_photonics` node.
- Skipped one exact relationship duplicate: Vivek Raghunathan → Xscape Photonics, co-founder/CEO. The existing canonical edge remains authoritative and prevents duplicate Sankey weight.
- Added distinct organization nodes for MIT, three MIT research groups, Columbia University, two Columbia research groups, Lightmatter, and Lightelligence.
- No source URL or source-ID collisions occurred in this tranche.
- No event records were added because the staging file contains only people-level training, employment, founder, and technical-leadership claims.

### Explicit non-edges preserved

- **Rejected:** Yichen Shen was a Lightmatter founder. MIT Sloan documents his participation on the 2017 competition team, but MIT's 2024 company history names Nicholas Harris, Darius Bunandar, and Thomas Graham as founders. No Shen → Lightmatter founder edge exists.
- **Asserted but not promoted:** Marin Soljačić was a Lightelligence founder. MIT says Yichen Shen teamed with Soljačić during founding, but the reviewed A-grade wording does not explicitly call Soljačić a founder. No Soljačić → Lightelligence founder edge exists.

The staging file reported no unresolved conflicts. University-degree claims continue to target MIT rather than a PI group unless the source explicitly supports lab membership.

### Validator result

`ruby scripts/validate_data.rb --canonical canonical` passed after merge. It reports 67 organizations, 91 people, 90 sources, 10 events, and 158 edges. The sole warning is unchanged from the pre-merge baseline: InfiniLink's public-item proxy announcement date follows its filing-established effective date, and the event notes document that exception.
