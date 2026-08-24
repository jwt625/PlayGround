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

## Legacy-company and China/Huawei merge

Inputs: `research/legacy_company_people_initial.yaml` and `research/china_photonics_people_initial.yaml`

### Before and after

| Collection | Before | Legacy added | China/Huawei added | After |
|---|---:|---:|---:|---:|
| Sources | 90 | 13 | 24 | 127 |
| Organizations | 67 | 5 | 10 | 82 |
| People | 91 | 18 | 20 | 129 |
| Events | 10 | 0 | 0 | 10 |
| Edges | 158 | 30 | 25 | 213 |

The resulting edge set contains 175 grade-A, 33 grade-B, and 5 grade-C edges. All 208 A/B edges are `validated`; the five pre-existing C-grade edges remain `asserted`. No C-, D-, or X-grade China/Huawei edge was promoted.

### Legacy-company normalization and deduplication

- Reused seven existing people: Vikram Sadagopan, Vivek Raghunathan, Matt Streshinsky, Ari Novack, Juthika Basak, Loi Nguyen, and Ford Tamer.
- Reused four source artifacts by canonical URL: the Xscape team page, Xscape's Vikram Sadagopan appointment, Enosemi's launch release, and OFC's Juthika Basak biography. Thirteen distinct legacy sources were added.
- Reused canonical `org_xscape` and `org_rockley` rather than staging's duplicate `org_xscape_photonics` and `org_rockley_photonics` IDs. Added Bookham, Kotura, NeoPhotonics, JDSU, and Oclaro from the frozen seed catalog.
- Skipped nine semantic edge duplicates already represented canonically: Vikram Sadagopan → Luxtera/Xscape; Vivek Raghunathan → Rockley/Xscape; Matt Streshinsky → Enosemi; Ari Novack → Enosemi; Juthika Basak → AMD; Loi Nguyen → Inphi; and Ford Tamer → Inphi.
- Added the remaining 30 source-backed legacy career edges. The one grade-B legacy edge, Donald Scifres → JDSU, remains `validated` as required by the frozen policy.
- Preserved the staging non-edges as exclusions: no unnamed Luxtera → Cisco team transfer; no Andrew Rickman → Oclaro edge inferred from Bookham/Oclaro succession; and no blanket NeoPhotonics → Lumentum personnel-continuity claim.

### China/Huawei normalization and identity handling

- Added 20 named people and 25 atomic edges: ten people at LightIC, Inxun, NVISION, Liangyin, Aluksen, and Beeplux, plus ten Huawei/HiSilicon optical-interconnect leaders.
- Added 24 unique sources from 25 staging source records. The duplicated CIOE conference-guide URL was normalized to one source record and is referenced by both supported claims.
- Added ten organizations: LightIC, Inxun, NVISION Technology, University of Hong Kong, Liangyin Technology, Cadence Design Systems, Aluksen, Beeplux Semiconductor, Huawei Technologies, and HiSilicon Optoelectronics. Existing Intel and Lucent Bell Labs nodes were reused.
- Modeled HiSilicon Optoelectronics as a Huawei business unit while preserving `海思光电子` / `海思光电` naming in aliases/notes. Conference-program affiliation remains point-in-time evidence, not an inferred employment interval.
- Canonical person records retain native Chinese names as `zh` aliases, romanized public forms where available, researcher-supplied-pinyin provenance in notes, and staging identity discriminators. This prevents romanization-only identity merges, including the documented 梁亦铂 / similarly rendered-name collision risk.
- All grade-B China claims remain `validated`, consistent with `source_policy.md`. Professional-society programs and strong industry interviews remain B; official company, university, government, and organizer conference artifacts remain A.

### Explicit China/Huawei non-edges and deferred claims

The following five generic provenance statements remain excluded from canonical edges and are preserved here as discovery leads only:

1. LightIC's unnamed team reportedly drew from MIT, Berkeley, Intel, Apple, and Uber.
2. NVISION's unnamed core team reportedly drew from Intel and Bell Labs.
3. Beeplux's unnamed core team reportedly drew from Intel, imec, and Xanadu.
4. Aluksen's unnamed silicon-photonics team reportedly included Intel, Broadcom, Huawei, or Cisco-related backgrounds.
5. Inxun's interview references partially named or unnamed founders with Intel/NeoPhotonics backgrounds.

No `team_transfer`, `spinout`, or person-level antecedent edge was created from those statements. Beeplux's Guo Peng and additional Futurewei, NeoPhotonics, InnoLight, Cisco, and Broadcom returnee paths also remain deferred because this tranche lacks clean named-person A/B claims.

### Post-merge validation

`ruby scripts/validate_data.rb --canonical canonical` passes with 82 organizations, 129 people, 127 sources, 10 events, and 213 edges. Referential integrity, required edge attributes, ID uniqueness, evidence references, and grade/status mapping pass. The sole warning is unchanged: `event_acquisition_infinilink_globalfoundries` uses a documented public-item proxy announcement date that follows the filing-established effective date.

## Stanford photonics merge

Input: `research/stanford_photonics_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 127 | 24 | 151 |
| Organizations | 82 | 12 | 94 |
| People | 129 | 20 | 149 |
| Events | 10 | 0 | 10 |
| Edges | 213 | 31 | 244 |

All 31 added Stanford edges are grade A and `validated`. They comprise 8 technical-leadership, 13 research-training, 6 founder, 2 employment, and 2 patent-affiliation edges.

### Deduplication and normalization

- Reused the existing `person_dirk_englund` identity and retained his MIT current role. His Stanford thesis source was added to the same person record, and his Vuckovic-lab PhD relationship is represented as a separate research-training edge.
- Added one Stanford University node and eight distinct Stanford academic-group/program nodes, with the lab nodes parented to Stanford University. No pre-existing canonical Stanford organization or source URL collided with the staging records.
- Added Flexcompute, SkyCool Systems, and Silicon Light Machines as distinct company nodes. No formal Stanford spinout relationship was inferred for any of them.
- Deduplicated the repeated `optomechanics` technology tag in the staged Solgaard Lab organization record.
- Preserved staging organization and person IDs because they already follow schema v0.1 prefix conventions and do not collide with canonical IDs.

### Patent-affiliation boundary and explicit non-edges

- Marc Jankowski and Carsten Langrock are linked to Stanford University with `patent_affiliation` edges using the Stanford Office of Technology Licensing artifact. Neither person is assigned a Fejer-group training edge; co-inventorship does not establish training.
- Tyler Hughes and Momchil Minkov are not labeled Flexcompute founders. The reviewed Stanford roster supports a Flexcompute destination but not founder status.
- Silicon Light Machines is not labeled a formal Stanford spinout. Olav Solgaard's official material supports dissertation-related technology and co-founding, but not the schema's formal-spinout or technology-transfer criteria.

### Post-merge validation

`ruby scripts/validate_data.rb --canonical canonical` passes with 94 organizations, 149 people, 151 sources, 10 events, and 244 edges. The sole warning remains the documented InfiniLink announcement/effective-date proxy exception.

## Temporal-evidence merge

Input: `research/temporal_edges_patch.yaml`

### Applied and skipped proposals

All ten date-field proposals were applied to their existing canonical edge IDs; none were skipped. The merge added no people, organizations, events, or edges.

| Edge ID | Applied field | Canonical value |
|---|---|---|
| `edge_mario_paniccia_intel_siph` | `end_date` | 2016, year precision |
| `edge_mario_paniccia_anello_founder` | `start_date` | 2019-11, month precision |
| `edge_peter_winzer_employment_bell_labs` | `end_date` | 2019, year precision |
| `edge_peter_winzer_nubis_founder` | `start_date` | 2020-09, month precision |
| `edge_guilhem_de_valicourt_employment_bell_labs` | `end_date` | 2017, year precision |
| `edge_guilhem_de_valicourt_nubis_founder` | `start_date` | 2020-09, month precision |
| `edge_zhang_bowers_training` | `end_date` | 2017-03, month precision |
| `edge_fang_bowers_training` | `end_date` | 2008-03-01, day precision |
| `edge_tran_bowers_training` | `end_date` | 2019-03, month precision |
| `edge_ronnen_lovinger_credo` | `start_date` | 2026-05-28, day precision |

Six new source artifacts were added. Three patch-local source records were deduplicated by canonical URL and mapped to existing IDs: `src_temporal_anello_paniccia_profile` → `src_anello_paniccia_profile`, `src_temporal_ciena_nubis_deck` → `src_ciena_nubis_deck`, and `src_temporal_credo_dust_close` → `src_credo_dust_close`. Every affected edge retains its previous evidence references and adds or reuses the direct date-supporting reference.

The ANELLO 2019/2020 wording is preserved explicitly: 2019-11 is the directly dated CEO/chairman appointment on the combined founder/executive edge, not a claimed legal-incorporation date. Fang's 2008 dissertation date still does not strictly order UCSB before Aurrion's year-only 2008 founding date. Tran's 2019-03 dissertation date similarly remains nonconclusive against year-only 2019 Nexus employment. Acquisition-day continuity remains excluded from strict-before talent flow.

### Graph impact and post-merge validation

Against the current 149-person Stanford-merged canonical snapshot, the reproducible A/B graph changed as follows:

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 227 | 227 | 0 |
| Ordered person contributions | 10 | 14 | +4 |
| Aggregated institution→institution flows | 9 | 12 | +3 |
| Excluded cross-organization person pairs | 108 | 104 | -4 |

The four newly ordered person contributions are Mario Paniccia (Intel Silicon Photonics → ANELLO), Peter Winzer (Bell Labs → Nubis), Guilhem de Valicourt (Bell Labs → Nubis), and Chong Zhang (UCSB Bowers Group → Nexus Photonics). Winzer and de Valicourt aggregate onto the same institution flow, hence four person contributions but three new aggregate flows.

`ruby scripts/validate_data.rb --canonical canonical` passes with 94 organizations, 149 people, 157 sources, 10 events, and 244 edges. The sole warning remains the documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes and regenerates `generated/graph_ab.json`, `generated/photonics_lineage_sankey.html`, and `generated/graph_build_report.md`.

## imec / Ghent Photonics Research Group merge

Input: `research/imec_ghent_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 157 | 11 | 168 |
| Organizations | 94 | 6 | 100 |
| People | 149 | 23 | 172 |
| Events | 10 | 1 | 11 |
| Edges | 244 | 49 | 293 |

All 49 staged atomic edges were merged: 44 person→organization relationships and five organization→organization relationships. The resulting canonical edge set contains 254 grade-A, 34 grade-B, and five grade-C edges. All 288 A/B edges are `validated`; the five pre-existing grade-C edges remain `asserted`.

### Entity and source normalization

- Reused canonical `org_huawei` rather than creating a Belgium-specific or acquisition-subsidiary duplicate. Huawei Tech Investment is retained in the acquisition event notes as the direct acquiring subsidiary.
- Reused canonical `org_nubis` for Lukas Elsinger's destination, replacing staging's seeded alias ID `org_nubis_communications`.
- Promoted the frozen seed IDs `org_imec`, `org_ghent_university`, `org_ghent_photonics_research_group`, `org_caliopa`, and `org_luceda_photonics`; this avoids parallel identities when the seed catalog is merged later.
- Added `org_psiquantum`, the staging file's only net-new organization record.
- Added all 11 distinct staging artifacts. No canonical URL or source-ID collision existed, and a post-merge canonical-URL check reports zero duplicates.
- No staged person name or person ID collided with the 149-person baseline.

### Caliopa acquisition and formation conflict

Added `event_caliopa_acquisition_huawei_2013` and `edge_caliopa_acquired_by_huawei`. Huawei's official 2013 financial filing establishes acquisition of 100% of Caliopa on 2013-08-06 for **EUR 7,000,000**. The event keeps `announced_date` null because the reviewed A-grade sources establish completion, not a separately dated announcement.

No canonical `conflicts.yaml` collection existed at merge time, so `conflict_caliopa_formation_year` is preserved here and on the organization record rather than silently resolved. `org_caliopa.founded_date` remains null and `research_status` is `conflicted`:

1. EVS's A-grade director biography reports Martin De Prycker as Caliopa founder/CEO from 2009 through 2013; this may include pre-incorporation founder activity.
2. The official Ghent PRG spin-off page says PRG-imec researchers founded Caliopa in September 2010.

The dated founder edge retains the directly supported 2009–2013 De Prycker interval, while the two formal institutional spinout edges retain the separately supported 2010-09 date. Neither is promoted to the organization's canonical formation date while the conflict remains open.

### Explicit non-edges

- No `team_transfer` edge was created from the Caliopa acquisition or Huawei's generic Belgian-R&D integration language.
- Dirk Taillaert, Joost Brouckaert, William Chen, Irfan Ansari, and Martijn Tassaert are not labeled Caliopa founders; reviewed sources support affiliations but not founder status.
- Luceda's six founders are not all assigned PRG training. Only people independently present on the official PRG roster receive PRG relationships.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 227 | 271 | +44 |
| People in evidence graph | 143 | 166 | +23 |
| Organizations in evidence graph | 79 | 85 | +6 |
| Ordered person contributions | 14 | 17 | +3 |
| Aggregated institution→institution flows | 12 | 15 | +3 |
| Excluded cross-organization person pairs | 104 | 125 | +21 |

The three newly dated contributions are Wim Bogaerts from Ghent University to the PRG, imec, and Luceda Photonics. The additional undated, source-backed multi-institution careers increase exclusions as expected under the strict date-order rule.

`ruby scripts/validate_data.rb --canonical canonical` passes with 100 organizations, 172 people, 168 sources, 11 events, and 293 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. The graph rebuild passes with 271 eligible A/B person→organization edges, 17 ordered person contributions, 15 aggregate flows, and 125 exclusions.

## Caltech / Scherer Nanofabrication Group merge

Input: `research/caltech_scherer_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 168 | 16 | 184 |
| Organizations | 100 | 3 | 103 |
| People | 172 | 17 | 189 |
| Events | 11 | 0 | 11 |
| Edges | 293 | 36 | 329 |

The merge added 32 grade-A and four grade-B edges. All are `validated`. The added edges comprise 23 research-training, eight founder, two technical-leadership, two executive-role, and one employment relationship.

### Deduplication and normalization

- Reused canonical `org_luxtera`, `org_cisco`, `org_xscape`, `org_elenion`, and `org_luminous_computing`. Added only `org_caltech`, `org_caltech_scherer_nanofab_group`, and `org_tesselmax`.
- Reused existing people `person_jelena_vuckovic`, `person_ron_horan`, and `person_vikram_sadagopan`. Jelena Vuckovic's Scherer-group roster source and training edge were added without changing her existing Stanford current affiliation.
- Mapped staging source `src_cisco_ron_horan` to canonical `src_cisco_horan_2021` and `src_xscape_vikram_appointment_caltech` to `src_xscape_vikram_appointment` because each pair has the same URL. A post-merge canonical URL check reports no duplicates introduced by this merge.
- Skipped three semantic duplicate edges: staged `edge_ron_horan_executive_cisco` duplicated the same canonical edge ID; staged `edge_vikram_sadagopan_employment_luxtera` maps to canonical `edge_vikram_sadagopan_luxtera`; and staged `edge_vikram_sadagopan_executive_xscape` maps to canonical `edge_vikram_sadagopan_xscape`. The existing Ron Horan Cisco edge title was enriched with the directly supported Client Optics Group detail.

### Degree, group-training, and explicit non-edges

- Caltech degree attendance and Scherer-group membership remain distinct atomic relationships. A Caltech thesis that directly names Axel Scherer as adviser supports both a degree-level `research_training` edge to Caltech and a separate adviser/group edge to the Scherer Nanofabrication Group.
- A Scherer alumni roster alone supports group alumni membership only. It does not establish a Caltech degree, adviser relationship, or tenure dates. This boundary applies especially to Tom Baehr-Jones and to roster-only alumni Jelena Vuckovic, Guangxi Wang, Marko Loncar, and Dongyoon Oh.
- No Caltech degree or formal Scherer-group training was inferred for Alex Dickinson, Thierry Pinguet, Richard Seligman, Ron Horan, or Vikram Sadagopan. Richard Seligman's grade-B Caltech employment edge is not a training edge.
- Cary Gunn's thesis acknowledgments do not establish that any acknowledged person worked at or founded Luxtera. Luxtera employment and founder status are represented only for people independently supported by direct sources.
- No Scherer-group cohort transfer to Luxtera was created, and no Luxtera workforce transfer to Cisco was inferred from Cisco's 2019 acquisition. The graph contains only person-specific relationships; no `team_transfer` edge was added.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 271 | 307 | +36 |
| People in evidence graph | 166 | 183 | +17 |
| Organizations in evidence graph | 85 | 89 | +4 |
| Ordered person contributions | 17 | 19 | +2 |
| Aggregated institution→institution flows | 15 | 17 | +2 |
| Excluded cross-organization person pairs | 125 | 151 | +26 |

The two newly orderable contributions are Michael Hochberg from Caltech to Luminous Computing and from the Scherer Nanofabrication Group to Luminous Computing. Other source-backed Caltech/Scherer careers remain excluded when their dates do not prove strict ordering.

`ruby scripts/validate_data.rb --canonical canonical` passes with 103 organizations, 189 people, 184 sources, 11 events, and 329 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes with 307 eligible A/B person→organization edges, 19 ordered person contributions, 17 aggregate flows, and 151 exclusions.

## UK silicon-photonics lineage merge

Input: `research/uk_silicon_photonics_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 184 | 18 | 202 |
| Organizations | 103 | 8 | 111 |
| People | 189 | 17 | 206 |
| Events | 11 | 2 | 13 |
| Edges | 329 | 31 | 360 |

The 31 net-new edges comprise 28 grade-A and three grade-B relationships, all `validated`: 12 technical-leadership, seven executive-role, four founder, four research-training, two employment, and two collaboration edges.

### Deduplication and canonical endpoint normalization

- Reused canonical `org_bookham`, `org_rockley`, `org_oclaro`, and `org_xscape`; staging references `org_rockley_photonics` and `org_xscape_photonics` were normalized to `org_rockley` and `org_xscape`. Bookham and Oclaro were promoted from seed-only records with direct SEC evidence rather than duplicated.
- Added the four staged organizations plus four required seed dependencies: University of Surrey, University of Southampton, Avanex, and Source Photonics. The staged organizations are the Surrey and Southampton silicon-photonics groups, Pointcloud, and Morelight Technologies.
- Reused `person_andrew_rickman`, `person_vivek_raghunathan`, and `person_alain_couder`; added the other 17 staged people. Rickman's full-name alias, Surrey thesis, and additional SEC evidence were merged into his existing record.
- Mapped `src_xscape_team_2026_uk` to existing `src_xscape_team` and `src_oclaro_couder_2013` to existing `src_oclaro_release_2013` because each pair resolves to the same URL. The other 18 source artifacts are distinct, and the full canonical source set has no duplicate URLs.
- Skipped five duplicate person edges: Rickman's Bookham and Rockley founder edges, Vivek Raghunathan's Rockley technical-leadership and Xscape founder edges, and Alain Couder's Oclaro executive edge. Rickman's existing founder edges received the additional direct evidence. Couder's existing Oclaro interval was refined to begin on the supported 2009-04-27 name-change date, paired with a separate Bookham interval ending that day.

### Program movement, corporate succession, and explicit non-edges

- `event_surrey_group_move_southampton_2012` records the supported 2012 program relocation, and the Southampton group names the Surrey group as its predecessor. Graham Reed has individually supported relationships to both programs. The source's statement that 13 people moved does not identify the other twelve, so no unnamed or inferred person memberships and no `team_transfer` edge were created.
- The Reed program edges both use year precision for 2012. They preserve the transition but do not create a strict-before graph flow because equal-year intervals are nonconclusive under the graph ordering policy.
- `event_bookham_avanex_form_oclaro_2009` records the announced and effective dates of the Bookham–Avanex transaction and Oclaro name change. Corporate predecessor/successor links provide organization history; they do not create employee-transfer edges.
- Andrew Rickman's Bookham role ended in 2004, so the 2009 corporate event does not establish Oclaro employment for him. No collective Bookham-to-Rockley alumni transfer was inferred from Rickman's serial-founder history or Rockley's 2021 leadership roster.
- Goran Mashanovich's Surrey PhD remains a university-level degree edge because the reviewed source does not name Graham Reed, an adviser, or the specific Surrey group. Callum Littlejohns's Surrey undergraduate degree likewise does not imply Surrey silicon-photonics-group membership.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 307 | 336 | +29 |
| People in evidence graph | 183 | 200 | +17 |
| Organizations in evidence graph | 89 | 95 | +6 |
| Ordered person contributions | 19 | 21 | +2 |
| Aggregated institution→institution flows | 17 | 19 | +2 |
| Excluded cross-organization person pairs | 151 | 163 | +12 |

The two new strictly ordered contributions are Andrew Rickman from the Surrey Silicon Photonics Research Group to Kotura and to Rockley Photonics. The two organization-to-organization collaboration edges remain visible canonically but are not person→organization graph inputs.

`ruby scripts/validate_data.rb --canonical canonical` passes with 111 organizations, 206 people, 202 sources, 13 events, and 360 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes with 336 eligible A/B person→organization edges, 21 ordered person contributions, 19 aggregate flows, and 163 exclusions.

## A*STAR / IME / Advanced Micro Foundry merge

Input: `research/astar_ime_amf_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 202 | 8 | 210 |
| Organizations | 111 | 2 | 113 |
| People | 206 | 19 | 225 |
| Events | 13 | 0 | 13 |
| Edges | 360 | 34 | 394 |

All 34 net-new edges are grade A and `validated`: 21 publication-affiliation, six technical-leadership, three employment, three patent-affiliation, and one founder edge. One existing grade-B founder edge was upgraded to grade A with direct A*STAR and paper-biography evidence; the pre-existing formal spinout edge was enriched rather than duplicated.

### Deduplication and identity resolution

- Reused canonical `org_astar_ime`, `org_amf`, `org_globalfoundries`, `org_inphi`, and `org_marvell`. Added only the A*STAR umbrella organization and the National Semiconductor Translation and Innovation Centre. IME is now explicitly parented to A*STAR.
- Enriched the existing AMF node with its legal-name alias, directly supported 2017 incorporation year, technology scope, website, and launch source. The existing GlobalFoundries parent relation remains unchanged.
- Resolved staged `person_patrick_guoqiang_lo` to canonical `person_guo_qiang_lo`. Patrick Guo-Qiang Lo and Patrick Lo are retained as aliases on one identity; the resolution is supported by the distinctive IME-to-AMF founder/leadership sequence, Guo-Qiang Lo paper form, and patent authorship. No second person node was created.
- Skipped staged `edge_patrick_lo_founder_amf` as a semantic duplicate of `edge_guo_qiang_lo_amf_founder`. The canonical edge now carries the supported 2017 founder year, A-grade evidence, and the unresolved President-versus-later-CTO title evolution.
- Skipped staged `edge_ime_spinout_amf` as a semantic duplicate of `edge_spinout_ime_amf`; the canonical edge now carries the 2017 date, both directly supported named founders, and three A-grade sources.
- All eight staged source URLs are distinct from the canonical URLs present before this merge. Both GlobalFoundries acquisition releases are retained because they are separate official publication endpoints and dates. The post-merge canonical URL audit reports no duplicates.

### Formal spinout, chronology conflict, and explicit non-edges

- `edge_spinout_ime_amf` represents the formally supported 2017 IME-to-AMF spinout and technology continuity. It does not assert that every IME paper author joined AMF, that the 2017 silicon-photonics author list transferred as a cohort, or that every IME process was covered by a formal IP license or assignment.
- Only Xianshu Luo and Guo-Qiang Lo are listed as founders on the formal spinout edge because the reviewed evidence names them. Chao Li receives separate publication-affiliation edges at IME and AMF; those artifacts do not establish founder status or an employment interval.
- `conflict_xiaoguang_tu_ime_inphi_overlap` remains open in this merge record: an official IME biography says Xiaoguang Tu joined Inphi in 2016, while a 2017 Optica paper prints an IME affiliation. The 2017 claim remains a point-in-time `publication_affiliation`, not an extension of IME employment. Publication lag, collaboration, or dual affiliation remain unresolved alternatives.
- The 2025 AMF acquisition release generically references skilled talent, but it does not name an acquisition-wide personnel roster. No `team_transfer` edge and no universal AMF-person-to-GlobalFoundries continuity were created. Only Guo-Qiang Lo has a named post-acquisition GlobalFoundries role in the reviewed evidence.
- Patent inventorship and AMF assignee linkage remain `patent_affiliation` edges; they do not establish ordinary employment or founder status for Shawn Yohanes Siew or other inventors.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 336 | 370 | +34 |
| People in evidence graph | 200 | 219 | +19 |
| Organizations in evidence graph | 95 | 98 | +3 |
| Ordered person contributions | 21 | 26 | +5 |
| Aggregated institution→institution flows | 19 | 23 | +4 |
| Excluded cross-organization person pairs | 163 | 168 | +5 |

The new strictly ordered flows are AMF→NSTIC through Xianshu Luo; IME→AMF through Guo-Qiang Lo and Xianshu Luo; IME→Marvell through Xiaoguang Tu; and IME→NSTIC through Xianshu Luo. Publication and patent edges contribute only where their point dates satisfy the graph's strict ordering rule; they are not silently retyped as employment.

`ruby scripts/validate_data.rb --canonical canonical` passes with 113 organizations, 225 people, 210 sources, 13 events, and 394 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes with 370 eligible A/B person→organization edges, 26 ordered person contributions, 23 aggregate flows, and 168 exclusions.

## ETH Zurich / Leuthold / Polariton merge

Input: `research/eth_leuthold_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 210 | 13 | 223 |
| Organizations | 113 | 4 | 117 |
| People | 225 | 15 | 240 |
| Events | 13 | 0 | 13 |
| Edges | 394 | 30 | 424 |

The 30 net-new edges comprise 24 grade-A and six grade-B relationships, all `validated`: 15 employment, eight research-training, four technical-leadership, one advisor, one publication-affiliation, and one collaboration edge.

### Deduplication and canonical endpoint normalization

- Reused canonical `org_eth_ief` for staging's `org_eth_leuthold_group`, `org_polariton` for `org_polariton_technologies`, and existing imec, Lucent Bell Labs, Nokia Bell Labs, and Marvell nodes. Added ETH Zurich as the parent of IEF plus Karlsruhe Institute of Technology, Technische Universität Berlin, and CREOL at the University of Central Florida.
- Reused `person_jurg_leuthold`, `person_claudia_hoessbacher`, `person_wolfgang_heni`, `person_benedikt_baeuerle`, and `person_qian_hu`; added the other 15 staged people. Existing native-character aliases for Hössbacher and Bäuerle were preserved.
- Mapped staged `src_eth_leuthold_bio_2026_ethlineage` to existing `src_eth_leuthold_bio_2026` because the URLs are identical. The other 13 source artifacts are distinct, including separate German- and English-language ETH acquisition pages and separate Marvell investor/newsroom publication endpoints. The post-merge canonical source set has no duplicate URLs.
- Skipped seven semantic duplicate edges: Leuthold's Lucent Bell Labs employment and ETH/IEF leadership; the three existing Polariton founder edges; the ETH/IEF→Polariton formal spinout; and the Polariton→Marvell acquisition edge. The corresponding canonical identities and relationships remain singular.

### Acquisition-event deduplication and unknown terms

The staged `event_polariton_acquisition_marvell_2026` duplicates canonical `event_acquisition_polariton_marvell`, so no second event was created. The existing event was enriched with the two new direct sources. Its announced date remains 2026-04-22; `effective_date` remains null because Marvell announced the acquisition on April 22 while ETH described it as completed on April 23 without supplying a separate legal close date. Transaction value and structure also remain null. The canonical acquisition edge likewise remains a single relationship with a null start date.

### Plasmonic/TFLN taxonomy conflict and explicit non-edges

- `conflict_polariton_seed_technology_tfln` remains documented as an open staging-to-canonical taxonomy discrepancy. The unverified seed classified Polariton as thin-film lithium niobate, while reviewed Marvell, ETH, and Polariton sources describe plasmonic active devices integrated with silicon-photonics platforms and provide no TFLN support.
- Canonical `org_polariton` therefore retains the evidence-backed `silicon_photonics`, `coherent_optics`, and `optical_io` tags, while its notes explicitly preserve the plasmonic mechanism and the absence of a dedicated schema taxonomy term. The unsupported TFLN alternative is not silently promoted or deleted from the conflict record in this merge note.
- The acquisition's generic statement that Marvell added a team does not identify retained individuals. No person→Marvell retention edges and no `team_transfer` edge were created.
- Nicholas Güsken's source supports a Polariton engineering role only; it does not establish Leuthold-group training. Qian Hu's record remains at Nokia Bell Labs, with a separate Polariton collaboration edge for the joint trial rather than an employment migration. Valery Shklover's project relationship terminates at ETH Zurich because the source does not establish IEF-group membership.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 370 | 400 | +30 |
| People in evidence graph | 219 | 234 | +15 |
| Organizations in evidence graph | 98 | 102 | +4 |
| Ordered person contributions | 26 | 34 | +8 |
| Aggregated institution→institution flows | 23 | 30 | +7 |
| Excluded cross-organization person pairs | 168 | 177 | +9 |

The new strictly ordered flows are IEF→imec, IEF→Polariton, IEF→TU Berlin, IEF→UCF CREOL, and ETH Zurich→IEF, KIT, and Lucent Bell Labs. The IEF→Polariton aggregate contains two named contributors, Claudia Hoessbacher and Benedikt Baeuerle, producing eight person contributions across seven aggregate flows.

`ruby scripts/validate_data.rb --canonical canonical` passes with 117 organizations, 240 people, 223 sources, 13 events, and 424 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes with 400 eligible A/B person→organization edges, 34 ordered person contributions, 30 aggregate flows, and 177 exclusions.

## Recent photonics-startup lineage merge

Input: `research/recent_startups_people_initial.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 223 | 22 | 245 |
| Organizations | 117 | 24 | 141 |
| People | 240 | 28 | 268 |
| Events | 13 | 0 | 13 |
| Edges | 424 | 62 | 486 |

All 62 staged edges were added as distinct grade-A, `validated` claims: 18 technical-leadership, 18 founder, ten employment, eight executive-role, four research-training, and four spinout relationships. The four organization-level origin edges are CEA-Leti→Scintil, Oxford Advanced Nanoscale Engineering Group→Salience Labs, University of Oxford→Lumai, and UPV iTEAM→iPronics.

### Deduplication and normalization

- Reused canonical Aurrion, Juniper Networks, Cisco, Lumentum, Inphi, and Intel nodes. Added the other 24 staged organizations, including eight startup nodes and their directly referenced institutional or corporate predecessors.
- Reused `person_volkan_kaman`, preserving his existing UCSB Bowers-group edge and adding the separately supported Aurrion and OpenLight roles. His canonical record now includes both Bowers and OpenLight sources. The other 28 staged people were added as distinct identities.
- No staged source ID or exact URL collided with the 223-source baseline. All 22 source artifacts were added, and the post-merge canonical source set has zero duplicate URLs.
- No staged edge ID or semantic person→organization relationship duplicated the canonical baseline. Current startup affiliations were normalized into person records only where the same official company artifact directly supports the named role.

### Institutional origins and eight explicit non-edges

- OpenLight is not modeled as a wholesale Aurrion or Juniper team transfer. Volkan Kaman and John Parker have individual Aurrion paths; generic company-history language does not enumerate a transferred cohort.
- CEA-Leti→Scintil is a formal spinout relationship, but the reviewed material does not establish a license of CEA-Leti's entire silicon-photonics patent portfolio. No `ip_license` edge was inferred.
- Ranovus biographies support named individual paths from CoreOptics, Cisco, Inphi, Lumentum, Bell-Northern Research, and BTI Systems. They do not establish that all CoreOptics founders or Cisco staff moved together.
- Avicena's founders are not modeled as a Kaiam cohort transfer. Rob Kalman and Alex Tselikov have individual Kaiam paths, while Bardia Pezeshki's biography discusses Kaiam separately.
- Salience Labs has a supported Oxford ANE/Feldmann lab-origin spinout edge, but no Oxford IP-license or named lab-team transfer is asserted.
- Lumai is a supported Oxford spinout. That organization-level origin does not establish individual advisor relationships or ownership of all Oxford optical-computing IP.
- iPronics is a supported UPV iTEAM spinout, but only José Capmany, Daniel Pérez López, Ivana Gasulla, and Prometheus Das Mahapatra receive individually sourced iTEAM relationships. No complete-group transfer is inferred.
- SiPhi's masked CTO “Dr Y.” remains unresolved and is not represented as a person. Marcus Yang is resolved independently through the SiPhi-linked handle and EPIC's full-name Intel biography; no surname-initial matching was used.

No `team_transfer`, cohort-transfer, or inferred `ip_license` edge was added anywhere in this tranche.

### Graph impact and validation

| Graph metric | Before | After | Change |
|---|---:|---:|---:|
| Validated A/B person→organization edges | 400 | 458 | +58 |
| People in evidence graph | 234 | 262 | +28 |
| Organizations in evidence graph | 102 | 125 | +23 |
| Ordered person contributions | 34 | 35 | +1 |
| Aggregated institution→institution flows | 30 | 31 | +1 |
| Excluded cross-organization person pairs | 177 | 207 | +30 |

The sole newly strict-orderable flow is CoreOptics→Ranovus through Hamid Arabzadeh. The other named upstream/startup pairs remain excluded where their sources do not establish both a prior endpoint and later start under the graph's conservative ordering rule. The four organization-level spinout edges are canonical evidence but are not person→organization graph inputs.

`ruby scripts/validate_data.rb --canonical canonical` passes with 141 organizations, 268 people, 245 sources, 13 events, and 486 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. `ruby scripts/build_graph.rb --canonical canonical` passes with 458 eligible A/B person→organization edges, 35 ordered person contributions, 31 aggregate flows, and 207 exclusions.

## Historical acquisition-event merge

Input: `research/historical_acquisitions_patch.yaml`

### Before and after

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 245 | 25 | 270 |
| Organizations | 141 | 4 | 145 |
| People | 268 | 0 | 268 |
| Events | 13 | 10 | 23 |
| Edges | 486 | 10 | 496 |

All ten staged acquisition events and their ten one-to-one acquisition edges were added as distinct grade-A, `validated` records. The transactions cover Luxtera→Cisco, Acacia→Cisco, Kotura→Mellanox, Mellanox→NVIDIA, Aurrion→Juniper, Elenion→Nokia, NeoPhotonics→Lumentum, Finisar→II-VI, Oclaro→Lumentum, and Lightwire→Cisco.

### Sources, organizations, and successor normalization

- All 25 staged source artifacts have distinct URLs relative to the pre-merge canonical set and were added. The post-merge canonical source audit reports zero duplicate URLs.
- Added Mellanox Technologies, NVIDIA, II-VI Incorporated, and Lightwire. Existing targets and acquirers—including Cisco, Juniper, Nokia, Lumentum, Luxtera, Acacia, Kotura, Aurrion, Elenion, NeoPhotonics, Finisar, and Oclaro—were reused.
- II-VI remains the named 2019 Finisar acquirer. The canonical organization chain records Finisar→II-VI and II-VI→Coherent Corp.; Coherent is the later corporate successor and is not substituted retroactively as the acquisition counterparty.

### Announced terms, closing values, and workforce boundary

- Event `headline_amount` preserves the source-supported transaction measure. Announcement headline values, per-share formulations, closing-date consideration, accounting purchase price, assumed awards, earnouts, and retention incentives are not silently treated as interchangeable. Where sources use different measurement bases, the event's `stock_component` and notes preserve that distinction.
- Nine of the ten added events have an announced date; Aurrion→Juniper retains a null announcement/signing date because the reviewed source establishes only the 2016-08-09 completion. All ten added events have an effective date.
- Nine of the ten added events have a disclosed headline value. Elenion→Nokia retains null value fields because the transaction terms were not disclosed.
- Issuer statements about employees joining an acquirer or a transaction adding skilled personnel remain acquisition integration context. They do not enumerate named retained people, establish continued employment intervals, or justify `team_transfer` edges. No acquisition-wide cohort or team-transfer edge was created.
- Outstanding transaction-specific gaps remain explicit, including named-person retention across all ten deals; final adjusted or aggregate cash for several transactions; Aurrion's public announcement date; and Mellanox's optical-subteam disposition separate from the whole-company transaction.

### Acquisition completeness and graph impact

| Acquisition metric | Before | After | Change |
|---|---:|---:|---:|
| Acquisition events | 11 | 21 | +10 |
| Acquisition edges | 11 | 21 | +10 |
| Edges linked to canonical events | 11 | 21 | +10 |
| Events with announced date | 10 | 19 | +9 |
| Events with effective date | 9 | 19 | +10 |
| Events with disclosed value | 9 | 18 | +9 |
| Events with at least two participants | 11 | 21 | +10 |
| Events with A/B evidence | 11 | 21 | +10 |

The default person-lineage graph remains unchanged at 458 validated A/B person→organization edges, 262 people, 125 organizations, 35 ordered person contributions, 31 aggregate flows, and 207 exclusions. Acquisition events and organization→organization edges refresh the graph input hashes but do not create person nodes, person edges, or derived talent flows.

`ruby scripts/validate_data.rb --canonical canonical` passes with 145 organizations, 268 people, 270 sources, 23 events, and 496 edges. The only warning remains the pre-existing documented InfiniLink announcement/effective-date proxy exception. The graph rebuild and `ruby scripts/build_reports.rb` both pass; release reports show 21 acquisition events, 21 linked acquisition edges, 19 announced dates, 19 effective dates, and 18 disclosed values.

## China incumbent people merge — 2026-08-23

| Collection | Before | Added | After |
|---|---:|---:|---:|
| Sources | 270 | 12 | 282 |
| Organizations | 145 | 27 | 172 |
| People | 268 | 20 | 288 |
| Events | 23 | 0 | 23 |
| Edges | 496 | 60 | 556 |

The China-incumbent tranche adds 60 individually sourced career, founder, technical-leadership, executive, and training edges spanning InnoLight, Eoptolink, Accelink, Source Photonics, Broadex, and Hisense Broadband. All 60 are validated A/B claims: 46 grade A and 14 grade B. Existing Agere, Finisar, Inphi, Intel, JDS Uniphase, NeoPhotonics, Oclaro, Source Photonics, and UCSB nodes were reused. Source URLs and person IDs were audited before merge; no duplicate source URL or existing-person identity was introduced.

### Native names, aliases, and identity boundaries

- Native Chinese names and company aliases are retained as structured aliases, including 中际旭创, 新易盛, 光迅科技, 博创科技, 海信宽带, and the staged Chinese person-name forms. Researcher-supplied pinyin or renderings remain marked as alternate/native aliases rather than silently replacing the source's canonical display name.
- Common-name risks such as 张军, Frank Chang, and David Li retain explicit role/company identity discriminators in the person records. Biography-specific identity notes are preserved for all 20 people.
- Hisense Broadband and “Hisense Broadband group” are one canonical organization endpoint; the wording difference does not create a second legal entity. “JDS” and “JDS Uniphase” likewise resolve to the existing canonical JDS Uniphase node.

### Explicit China-incumbent non-edges

- No broader InnoLight cohort or team-transfer edge is inferred from seven individually named biographies or their shared U.S. optical-component employers.
- Jianying Zhou's grouped biography does not establish identical roles or a public-company succession path across Oclaro, Finisar, Nortel, and JDS. Only named employer edges are retained, with missing title and chronology uncertainty explicit.
- Broadex's ownership relationship with YOFC does not establish a YOFC team transfer. Only Jinkuan Tang's individually sourced YOFC-to-Broadex path is represented.

No generic employee transfer, cohort edge, acquisition continuity, or public-company succession inference was added. Organization status labels are descriptive node metadata and do not create lineage edges.

### Graph and validation impact

The default A/B graph grows from 458 to 518 validated person→organization edges, from 262 to 282 people, and from 125 to 153 organizations. Strictly ordered person contributions increase from 35 to 60, aggregate institution flows from 31 to 52, and disclosed sequence exclusions from 207 to 255. The increase in exclusions reflects preservation of biography gaps rather than imputed ordering.

`ruby scripts/validate_data.rb --canonical canonical`, `ruby scripts/build_graph.rb`, and `ruby scripts/build_reports.rb` pass. The validator reports 172 organizations, 288 people, 282 sources, 23 events, and 556 edges; its sole warning remains the pre-existing documented InfiniLink publication-proxy exception. Acquisition completeness remains unchanged at 21 events, 21 linked acquisition edges, 19 announced dates, 19 effective dates, and 18 disclosed values.
