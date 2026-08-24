# imec / Ghent photonics lineage — initial tranche

As of 2026-08-23, the tranche contains **23 named people, 49 atomic edges, 11 A/B sources, one acquisition event, and one explicit date conflict**. Forty-eight edges are supported by A-grade primary evidence; one Ghent-degree edge uses a B-grade Optica conference biography.

## Decision-useful findings

| Node | Directly supported lineage | Confidence and boundary |
|---|---|---|
| Ghent Photonics Research Group | The official roster establishes current professors Roel Baets, Peter Bienstman, Wim Bogaerts, Gunther Roelkens, and Dries Van Thourhout, plus named former staff. | High for membership category. The mutable roster usually omits former roles and intervals, so most alumni edges retain null dates and an unresolved role. |
| imec foundry platform | imec's biography of Wim Bogaerts documents his role in building the silicon-photonics platform, full-time Ghent/imec work from 2016, and Luceda co-founding. PRG alumni destinations directly identify Peter De Heyn, Ashwyn Srinivasan, and Joris Van Campenhout at imec. | High for the recorded facts. Destination labels do not establish exact titles, teams, or start dates. |
| Luceda Photonics | PRG's spin-off page names Wim Bogaerts, Erwin De Baetselier, Pieter Dumon, Martin Fiers, Joris Geessels, and Pierre Wahl as founders in June 2014. Separate roster evidence supports PRG membership only for Bogaerts, Dumon, Fiers, and Geessels. | High. No PRG training is inferred for De Baetselier or Wahl. The institutional spinout edges do not imply that every founder worked for every contributing institution. |
| Caliopa | PRG says its researchers and imec formed Caliopa in September 2010. Individual evidence establishes PRG→Caliopa connections for Dirk Taillaert and Joost Brouckaert and PRG→“Caliopa-Huawei” affiliations for William Chen, Irfan Ansari, and Martijn Tassaert. | High for affiliation, low specificity for roles/dates. None of these five receives a founder edge because the reviewed primary sources do not explicitly identify them as founders. |
| Martin De Prycker | EVS's regulatory filing identifies De Prycker as Caliopa founder/CEO during 2009–2013 and documents Ghent degrees. | High source quality, but the 2009 role start conflicts with PRG's September 2010 formation date. Both alternatives are preserved; 2009 may represent pre-incorporation activity. |
| Huawei acquisition | Huawei's filing records acquisition of 100% of Caliopa on 2013-08-06 for EUR 7 million. PRG and Huawei describe Belgian R&D integration. | High for ownership, close date, value, and R&D context. There is **no `team_transfer` edge** because no reviewed source names a retained group moving together. |
| Downstream design/foundry ecosystem | The official PRG roster directly maps Koen Alexander to PsiQuantum, Lukas Elsinger to Nubis, Yuting Shi to Lumentum, and three named alumni to imec. | High for destination affiliation; exact roles and career intervals remain unresolved. |

## Degree versus group membership

The dataset deliberately models these as different claims:

- Wim Bogaerts's 2004 PhD and Pieter Dumon's Ghent MSc/PhD terminate at `org_ghent_university`; neither source identifies a dissertation adviser.
- Dirk Taillaert's individual PRG page names Roel Baets as promoter, identifies a November 2004 PhD, and gives a 1999–2010 group interval. His doctoral edge therefore terminates at `org_ghent_photonics_research_group`, with a separate postdoctoral edge.
- Martin De Prycker's Ghent degrees terminate at the university. His later Caliopa founder role is not evidence of PRG membership.
- A PRG former-staff listing is modeled as employment/group membership, not automatically as a degree or adviser relationship.

## Corporate event versus people flow

```text
PRG + imec --formal spinout evidence--> Caliopa --100% acquisition--> Huawei
       named alumni affiliations             2013-08-06, EUR 7m

No derived claim: “the Caliopa team transferred to Huawei on close.”
```

The “Caliopa-Huawei” labels attached to William Chen, Irfan Ansari, and Martijn Tassaert show a person-level affiliation with that combined historical label, but do not resolve legal employer or pre/post-acquisition timing. Stevan Stankovic has a separate PRG→Huawei destination edge; the dataset does not attribute that move to the Caliopa transaction.

## Unresolved items

- Find Caliopa incorporation records, original shareholder filings, or a contemporaneous founder announcement to resolve the 2009-versus-September-2010 date conflict and name the complete founder group.
- Locate archived Caliopa/Huawei Belgium biographies for Taillaert, Brouckaert, Chen, Ansari, and Tassaert to establish roles and exact intervals.
- Retrieve PRG individual pages or theses for downstream alumni to replace destination-only employment edges with degree/adviser and career dates where available.
- Add first-party PsiQuantum, Nubis, Lumentum, and imec biographies for the named alumni before treating mutable roster destinations as current indefinitely.
- Resolve Luceda founder roles and intervals individually. Only Pieter Dumon's 2025 CEO role is supported in this tranche; other titles are intentionally null.
- Investigate Luceda's 2025 Semitronix transaction separately. It is outside the requested Caliopa→Huawei context and is not introduced from a secondary article here.

## Validation

The complete file was loaded against the frozen schema, edge catalog, and seeded organizations:

```text
ruby scripts/validate_data.rb \
  --organization-file organizations_seed.yaml \
  --organization-file research/imec_ghent_people_initial.yaml \
  --people-file research/imec_ghent_people_initial.yaml \
  --source-file research/imec_ghent_people_initial.yaml \
  --event-file research/imec_ghent_people_initial.yaml \
  --edge-file research/imec_ghent_people_initial.yaml

Validation passed
people: 23
sources: 11
events: 1
edges: 49
```

Edge mix: five technical-leadership, 27 employment, five research-training, seven founder, four institutional-spinout, and one acquisition edge. No team-transfer or inferred/X-grade edge is present.
