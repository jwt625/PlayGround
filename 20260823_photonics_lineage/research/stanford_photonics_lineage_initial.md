# Stanford photonics lineage — initial tranche

As of 2026-08-23, this tranche records **21 people, 12 organizations, 31 atomic edges, and 24 A-grade primary sources**. It centers the eight requested Stanford faculty branches and deliberately separates specific PI-group training from university-level affiliation, co-inventorship, and commercial roles.

## What the evidence supports

| Branch | Directly evidenced people | Supported lineage or commercialization signal | Confidence / boundary |
|---|---|---|---|
| Shanhui Fan | Fan; Zongfu Yu; Aaswath Raman; Eli Goldstein; Tyler Hughes; Momchil Minkov | Yu directly identifies Fan as his PhD adviser and as a Flexcompute co-founder. Stanford rosters establish Raman, Goldstein, Hughes, and Minkov as group alumni; SkyCool's own page establishes Raman and Goldstein as co-founders. | High for the recorded edges. Hughes/Minkov are **not** assigned founder status; their Flexcompute roles and dates remain unresolved. |
| Jelena Vuckovic | Vuckovic; Dirk Englund; Andrei Faraon; Ilya Fushman; Hatice Altug | Stanford-hosted theses establish the Englund, Faraon, and Fushman doctoral connections. A Stanford thesis acknowledgment directly identifies Altug as a graduated group member. | High for Englund/Faraon/Fushman. Altug is recorded as `training_type: other` because the reviewed artifact does not itself establish degree type or dates. |
| Olav Solgaard | Solgaard | Solgaard's CV establishes a Stanford PhD in 1992, Stanford faculty appointment from 1999, and Silicon Light Machines co-founder interval 1994–2001. His faculty page ties the dissertation work to SLM. | High. The degree edge terminates at Stanford University, not a locally invented Bloom-group node. No formal Stanford-spinout edge is asserted. |
| David A. B. Miller | Miller | Stanford's bio documents his semiconductor-optics and optical-interconnect program and earlier Bell Labs interval. | High for Stanford technical leadership. This tranche does not reconstruct Miller's student roster or duplicate the Bell Labs lineage dataset. |
| James S. Harris Jr. | Harris; Seth Bank; Vijit Sabnis | Harris's official profile covers III-V/MBE/VCSEL optical-interconnect work. The Stanford-hosted former-student roster directly places Bank and Sabnis in the Harris group. | High for group membership; Bank's 2006 thesis completion is explicit. Sabnis's degree type and dates remain unresolved here. |
| Stephen E. Harris | Stephen Harris | Stanford documents his lasers, quantum-electronics, atomic-physics, and nonlinear-optics research program. | High for faculty leadership; no student lineage is asserted without a direct roster/thesis source. |
| Robert L. Byer | Byer; Martin Fejer | Byer's own Stanford-hosted PhD-student list names Fejer and a 1986 thesis, providing a direct Byer→Fejer training bridge. | High. This is the cleanest nonlinear-optics academic-lineage edge in the tranche. |
| Martin M. Fejer | Fejer; Marc Jankowski; Carsten Langrock | Stanford documents Fejer joining Applied Physics in 1986 and his nonlinear/guided-wave optics program. Stanford OTL names Fejer, Jankowski, and Langrock as innovators on nanophotonic periodically poled lithium niobate. | High for Fejer's leadership and the three inventor affiliations. Co-inventorship is **not** treated as proof that Jankowski or Langrock trained in Fejer's group. |

## Cross-domain coverage

- Integrated and computational photonics: Fan, Yu, Hughes, Minkov, and the Vuckovic branch.
- Optical interconnects: Miller's documented program and James Harris's VCSEL/III-V work.
- Nonlinear optics and lithium niobate: the Byer→Fejer training edge plus the Fejer/Jankowski/Langrock Stanford technology disclosure.
- Lasers and III-V: James Harris, Bank, Sabnis, Stephen Harris, Byer, and Fejer.
- Quantum/nanophotonics: Vuckovic, Englund, Faraon, Fushman, and Altug.
- MEMS and beam steering: Solgaard's Stanford lab and Silicon Light Machines commercialization.
- Adjacent startups: Flexcompute, SkyCool Systems, and Silicon Light Machines, with founder edges only where an official bio, CV, or company history explicitly says founder/co-founder.

## Important non-edges

1. The Fan-group roster's Flexcompute destinations do not prove that Tyler Hughes or Momchil Minkov founded the company. No founder edge is created for either person.
2. Stanford OTL's co-inventor list does not establish Fejer-group training for Marc Jankowski or Carsten Langrock. Their edges are university-level `patent_affiliation` records.
3. Solgaard's dissertation-to-Silicon-Light-Machines narrative is strong technical-origin evidence, but it does not by itself meet the schema definition for a formal spinout or technology-transfer edge.
4. Stanford affiliation is not treated as a specific PI-group relationship. Solgaard's Stanford PhD therefore maps to the university; his CV names David Bloom, but this tranche does not create an unevidenced Bloom-group organization.

## Unresolved fields and next research

- Obtain first-party founder-date evidence for Flexcompute, SkyCool Systems, and exact company formation timing rather than backfilling dates from secondary databases.
- Replace the historical Fan roster's destination-only Flexcompute evidence with current company biographies for Hughes and Minkov, including exact roles and intervals.
- Find dissertation records or official lab biographies for Vijit Sabnis and Hatice Altug to resolve degree type, adviser wording, and dates at supported precision.
- Expand Miller, Stephen Harris, Byer, Solgaard, and Fejer alumni rosters using Stanford theses/CVs before claiming downstream company mixing nodes.
- Resolve Silicon Light Machines' acquisition/successor history in the legacy-company workstream, keeping the corporate event distinct from Solgaard's person-level founder edge.
- The schema taxonomy lacks explicit quantum-photonics, nonlinear-photonics, and optical-sensing labels. Records use the nearest frozen Phase-1 categories; the prose and source locators preserve the technical specificity.

## Validation

The YAML parses successfully and passes the repository validator when loaded as both the local organization collection and the people collection:

```text
ruby scripts/validate_data.rb \
  --organization-file research/stanford_photonics_people_initial.yaml \
  --people-file research/stanford_photonics_people_initial.yaml

Validation passed
organizations: 12
people: 21
```

An additional integrity check found 31 unique edge IDs and no missing source references. All 31 asserted edges are grade A and `validated`; date fields remain null where the reviewed source does not state a defensible date.
