# Southampton / Surrey / Bookham / Rockley lineage — initial tranche

As of 2026-08-23, this tranche contains **20 named people, 36 atomic A/B edges, 20 sources, and two contextual corporate/program events**. Thirty-three edges are A-grade and three are B-grade. The model keeps university training, lab membership, corporate succession, and person migration separate.

## Core lineage

```text
Graham Reed's Surrey silicon-photonics program (1989–2012)
        │
        ├── Andrew Rickman PhD, 1994 ──> Bookham founder ──> Rockley founder
        │
        └── program relocated in 2012 ──> Southampton Silicon Photonics Group
                                              ├── CORNERSTONE/foundry leadership
                                              ├── Pointcloud founders
                                              └── formal Rockley research partnership

Bookham + Avanex ── corporate merger/name change, 2009-04-27 ──> Oclaro
        No automatic person-migration edges across this event
```

Andrew Rickman's 1994 Surrey thesis directly names Graham Reed and Philip Walker as supervisors. Southampton's institutional history independently calls Rickman Reed's student and connects Reed-group guided-wave work to Bookham. This is strong academic-commercial lineage evidence, but it does not establish a formal university spinout transaction or a 1988 technology transfer.

## Evidence-backed branches

| Branch | Named people and edges | Confidence / boundary |
|---|---|---|
| Surrey | Graham Reed; Andrew Rickman; Goran Mashanovich; Callum Littlejohns | Reed's group leadership and Rickman's Reed-supervised PhD are direct. Mashanovich's Surrey PhD and Littlejohns's Surrey BEng terminate at the university because their sources do not identify Reed's lab. |
| Southampton | Reed; David Thomson; Frederic Gardes; Goran Mashanovich; Callum Littlejohns; Milos Nedeljkovic | Current university profiles directly establish roles and group membership. Littlejohns's Southampton ORC silicon-photonics PhD is separately recorded from his Surrey undergraduate degree. |
| Pointcloud | Remus Nicolaescu; Graham Reed; David Thomson | Optica's first-person Reed interview identifies Nicolaescu as CEO/originator and Thomson as co-founder; the Future Photonics Hub identifies Reed as co-founder. Founding date remains unresolved. |
| Bookham | Andrew Rickman; Michael Scott; Liam Nagle; James Haynes; Adrian Meldrum; Alain Couder | SEC filings provide founder and time-bounded technical/executive roles. Only Couder has a separately evidenced continuous Bookham→Oclaro leadership path. |
| Rockley | Andrew Rickman; Amit Nagra; Roozbeh Parsa; Hooman Abediasl; Ben Ver Steeg; Cas Wierzynski; Ara Nazarian; Vivek Raghunathan | Roles come from 2021 Rockley/SEC artifacts, except Raghunathan's official Xscape bio. These are historical snapshots, not claims of current 2026 employment. |
| Downstream mixing | Vivek Raghunathan→Xscape; Ben Ver Steeg→Morelight; Reed/Thomson/Nicolaescu→Pointcloud | Founder edges are person-specific. Rockley does not receive an inferred Bookham-team edge, and Xscape does not receive an inferred Rockley-team transfer. |

## Time-aware corporate history

Bookham's company history is represented independently of people:

- Rickman founded Bookham in 1988 and led it through 2004.
- Michael Scott joined as CTO in December 2002; Liam Nagle joined as COO in November 2002.
- James Haynes's Bookham sequence begins in June 2003; Adrian Meldrum's begins in 2001.
- Alain Couder joined Bookham as CEO in August 2007.
- Bookham and Avanex announced their merger agreement on 2009-01-27. The combination closed on 2009-04-27, when Bookham changed its name to Oclaro.
- Couder receives separate Bookham and Oclaro edges because an official biography supports continuous leadership. Rickman receives no Oclaro edge because his Bookham leadership ended in 2004.

The merger event does not place Scott, Nagle, Haynes, or Meldrum at Oclaro. Their Bookham end dates remain null where the reviewed filings do not establish departure.

## Surrey versus Southampton

Reed's 2023 Optica interview states that Southampton recruited his entire 13-person Surrey group in 2012. The dataset records a program-level event and predecessor relationship, but does **not** generate twelve unnamed person migrations. Only Reed is currently supported with direct edges on both sides.

This treatment avoids two common errors:

1. A Surrey degree does not automatically imply membership in Reed's silicon-photonics group.
2. Current Southampton collaboration does not imply that a person was part of the 2012 Surrey relocation.

## Unresolved research

- Identify the other twelve members of the 2012 Surrey→Southampton move through archived Surrey rosters, Southampton appointment releases, CVs, or theses before creating person edges.
- Find official Bookham departure dates for Michael Scott, Liam Nagle, James Haynes, and Adrian Meldrum; do not use the 2009 company transaction as a proxy.
- Expand Bookham's technical inventor layer using patents and dated product-team biographies rather than executive titles alone.
- Resolve the founding date and complete founder roster of Pointcloud through incorporation or first-party company records.
- Replace Rockley's 2021 point-in-time management roster with dated employment intervals and trace post-restructuring destinations.
- Resolve Morelight Technologies' technology and corporate history; only Ben Ver Steeg's founder/CTO link is supported here.
- The frozen taxonomy lacks an optical-sensing category. Mid-IR and wearable-sensing records use the nearest permitted labels, with specialization preserved in titles and notes.

## Validation

```text
ruby scripts/validate_data.rb \
  --organization-file organizations_seed.yaml \
  --organization-file research/uk_silicon_photonics_people_initial.yaml \
  --people-file research/uk_silicon_photonics_people_initial.yaml \
  --source-file research/uk_silicon_photonics_people_initial.yaml \
  --event-file research/uk_silicon_photonics_people_initial.yaml \
  --edge-file research/uk_silicon_photonics_people_initial.yaml

Validation passed
people: 20
sources: 20
events: 2
edges: 36
```

Edge mix: 13 technical-leadership, seven founder, four research-training, two employment, eight executive-role, and two collaboration edges. No inferred/X-grade edge or person-level team-transfer edge is present.
