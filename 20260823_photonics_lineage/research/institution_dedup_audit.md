# Institution and company identity audit for visualization

As of 2026-08-23. The machine-readable result is `config/institution_display_map.yaml`. This audit is display-only: it does not change canonical organizations, edges, events, or source claims.

## Bottom line

The audited canonical snapshot has **no duplicate organization IDs that should be merged destructively**. The apparent collisions fall into three different classes:

1. **Alias already normalized to one canonical node:** Intel Photonics Technology Lab is a former-name alias on `org_intel_silicon_photonics`, not a second organization.
2. **Nested institutions:** a university or umbrella organization and a directly evidenced lab, group, or institute are different claim targets. They may collapse in a high-level person-affiliation Sankey, but must remain separate in evidence and lineage views.
3. **Time-aware or transaction-aware entities:** Bell Labs eras, acquired companies, and corporate successors remain separate canonically. Selected families may collapse only for a present-family person-affiliation Sankey.

The display map contains **18 non-overlapping clusters and 51 unique canonical organization IDs**. Sixteen clusters permit Sankey-only aggregation; Ghent–imec–PRG and JDSU–Oclaro–Lumentum are deliberately non-collapsible. Ten university/internal-lab clusters allow aggregation in both `founder` and `strict` modes. Corporate families, industrial-lab histories, and the A*STAR/IME umbrella pair are not newly collapsed in `strict`. Every allowed collapse requires per-person deduplication so parent and child affiliations do not inflate weights.

## Audit decisions

| Display family | Canonical decision | Sankey-only collapse | Reason |
|---|---|---:|---|
| Bell Labs | Keep generic Bell Labs plus Lucent, Alcatel-Lucent, and Nokia-era lab nodes distinct | Yes | Biographies resolve eras at different precision. Collapse removes display fragmentation but cannot assign an unresolved generic edge to a specific era. |
| Intel / Intel SiPh / PTL | Keep Intel parent and one silicon-photonics business-unit node; PTL remains an alias | Yes | Legal-employer and technical-unit claims differ. There is no duplicate PTL node to merge. |
| MIT and named labs | Keep MIT plus Englund, Ram, and Soljacic groups distinct | Yes | An MIT degree does not prove lab membership. University-level Sankeys may aggregate after person deduplication. |
| Stanford and named groups | Keep Stanford plus eight group/program nodes distinct | Yes | Advisor and group edges carry information that a university affiliation does not. |
| UCSB and Bowers group | Keep both | Yes | UCSB attendance and Bowers-group training are not equivalent. |
| Columbia and named groups | Keep Columbia plus Lipson and Bergman groups distinct | Yes | Both groups have explicit Columbia parent links; group membership still requires direct evidence. |
| Caltech and Scherer Nanofabrication Group | Keep both | Yes | The group is school-contained and explicitly parented to Caltech; collapse is display-only. |
| ETH Zurich and IEF | Keep both | Yes | IEF is an internal ETH institute, while an ETH degree alone does not establish Leuthold-group membership. |
| Oxford and ANE | Keep both | Yes | ANE is an explicitly parented Oxford academic group. |
| UPV and iTEAM Photonics | Keep both | Yes | iTEAM is an explicitly parented UPV academic unit. |
| Surrey and Silicon Photonics Research Group | Keep both | Yes | Surrey affiliation and group membership remain different canonical claims. |
| Southampton and Silicon Photonics Group | Keep both | Yes | Southampton affiliation and group membership remain different canonical claims. |
| imec / Ghent / PRG | Keep all three and do not aggregate by default | No | PRG is joint, but Ghent degrees, PRG membership, and imec employment are separate claims; a single node would create false equivalence. |
| A*STAR / IME | Keep umbrella and operating institute distinct | Yes | Technical papers and AMF spinout evidence specifically name IME; high-level institution views may use A*STAR IME. |
| Finisar / II-VI / Coherent | Keep acquisition target, 2019 acquirer, and later successor identity distinct | Yes | Present-family talent views may aggregate, but the Finisar transaction must continue to target II-VI, not Coherent. |
| JDSU / Oclaro / Lumentum | Keep all three; do not aggregate yet | No | Oclaro→Lumentum is canonical, but the JDSU branch into Lumentum is not explicitly represented. Collapsing now would overstate continuity. |
| Mellanox / NVIDIA | Keep both | Yes | Mellanox→NVIDIA is an acquisition, not a duplicate or an optical-team transfer. |
| Luxtera / Cisco | Keep both | Yes | Luxtera→Cisco is an acquisition, and only named person evidence can support individual continuity. |

## True duplicates versus entities that only look duplicated

### Already canonicalized alias

- `org_intel_silicon_photonics` is named **Intel Silicon Photonics Solutions Group** and carries **Intel Photonics Technology Lab** as a former-name alias. The staging history treated these labels as one time-evolving Intel photonics node pending finer organizational history. Creating `org_intel_ptl` would duplicate the current canonical decision.

### Time-aware Bell Labs identities

`org_bell_labs`, `org_lucent_bell_labs`, `org_alcatel_lucent_bell_labs`, and `org_nokia_bell_labs` are not duplicates. The Bell Labs memo explicitly preserves corporate-era identity and notes that several biographies only say “Bell Labs.” Peter Winzer's 2000–2020 span, for example, should not be mechanically split across corporate eras without a CV-quality source. The display collapse therefore means “same research lineage brand for this view,” not “same legal employer at all dates.”

### Parent institution versus lab or group

- MIT degrees and broad affiliations terminate at `org_mit`; direct lab evidence terminates at `org_mit_englund_group`, `org_mit_ram_group`, or `org_mit_soljacic_group`.
- Stanford university affiliations terminate at `org_stanford_university`; direct evidence supports eight separate group/program nodes. Patent co-inventorship and a Stanford degree do not establish PI-group training.
- UCSB affiliation and `org_ucsb_bowers_group` membership remain distinct. The UCSB memo specifically excludes Bowers-group inference for several UCSB alumni.
- Columbia University remains distinct from the Lipson Nanophotonics Group and Bergman Lightwave Research Laboratory.
- Caltech remains distinct from `org_caltech_scherer_nanofab_group`; the latter requires group-level evidence even though both may display as Caltech photonics.
- ETH Zurich, Oxford, UPV, Surrey, and Southampton remain distinct from their explicitly parented IEF, ANE, iTEAM, and silicon-photonics group nodes.
- A*STAR is the umbrella; `org_astar_ime` is the operating institute printed in papers and named in the AMF spinout evidence.

These are nested identities, not duplicate employers. Display aggregation is acceptable only after deduplicating the same person within a cluster.

### Joint Ghent–imec research ecosystem

`org_ghent_photonics_research_group` is documented as a joint Ghent University–imec group, but that does not make `org_ghent_university`, `org_imec`, and PRG interchangeable. The source memo distinguishes Ghent degrees, PRG roster membership, and imec employment. This family is included in the map for audit completeness with aggregation disabled.

### Corporate histories

- **Finisar → II-VI → Coherent:** the canonical acquisition edge correctly targets II-VI on 2019-09-24. `org_coherent` is the later successor identity. A present-family Sankey may collapse the three, but historical and transaction views may not.
- **Oclaro → Lumentum:** the canonical acquisition edge closes on 2018-12-10. `org_jdsu` is also a distinct historical company, and the present canonical snapshot does not encode the full JDSU/Lumentum separation. This three-node family remains non-collapsible.
- **Mellanox → NVIDIA:** the canonical edge closes on 2020-04-27. The acquisition does not prove movement of a separately identifiable optical team.
- **Luxtera → Cisco:** the canonical edge closes on 2019-02-06. The acquisition announcement supports planned employee integration, but the lineage memo only treats named person continuity as person evidence.

## Structural issues found, not repaired

1. `org_ucsb_bowers_group` is logically a UCSB lab but its canonical `parent_organization_id` is null. The display map groups it with UCSB without mutating the canonical record.
2. The four Bell Labs era nodes do not carry explicit predecessor/successor or parent-company relationships. That is appropriate for this display audit, but a future time-resolved corporate-history layer should encode supported era transitions.
3. The canonical snapshot records `org_ii_vi → org_coherent` through predecessor/successor fields but has no explicit rebrand/successor event edge. The historical acquisition report already recommends adding one.
4. The canonical snapshot does not yet encode a sourced JDSU-to-Lumentum separation/successor relationship. This is why the JDSU/Oclaro/Lumentum grouping is audit-only.
5. The Ghent PRG note says the group is joint with imec, while its single canonical parent is Ghent University. A multi-parent relationship cannot be expressed through the current `parent_organization_id` field; no duplicate node should be introduced to work around that limitation.
6. `org_cornell_eastman_group` has no separate Cornell University node in the canonical snapshot, and `org_ucf_creol` has no separate University of Central Florida node. They cannot form validated parent/lab display clusters without first adding and sourcing those parent organizations canonically; this audit does not invent them.

## Safe visualization contract

For an allowed cluster, rewrite only person-affiliation Sankey endpoints to the display label and retain the original organization ID on every contributing atomic edge. Deduplicate by `(person_id, display_cluster_id, time_slice)` before computing weights. `aggregation_modes` is an allow-list: the ten university/internal-lab clusters are enabled for both `founder` and `strict`; Bell Labs eras, Intel/SiPh, A*STAR/IME, and acquisition/successor families are enabled only for `founder`. Do not apply this map to acquisition, spinout, IP-transfer, legal-employer, advisor, or chronology views. Evidence tables should always show the original canonical organization name and ID.

The non-collapsible families remain useful in the configuration because they make the audit decision executable: a renderer can recognize the family but must leave its endpoints unchanged.
