# Recent active photonics startups: initial people and lineage tranche

**As of:** 2026-08-23  
**Structured data:** `research/recent_startups_people_initial.yaml`

## Result

The tranche covers all eight requested startups with **29 named founders or technical leaders**, **62 atomic A-grade edges**, and **four narrowly scoped formal/lab spinout edges**. Every included person has both a directly supported startup role and at least one upstream employment, research, or degree edge. Dates remain null where a source gives a role without a defensible interval.

| Startup | Named people | Strongest directly supported upstream paths |
|---|---:|---|
| OpenLight | 4 | Aurrion→OpenLight (Volkan Kaman, John Parker); Cisco→OpenLight (Adam Carter); Infinera→OpenLight (Chris Barnard) |
| Scintil Photonics | 3 | CEA-Leti→Scintil (Sylvie Menezo); Tronics→Scintil (Pascal Langlois); TI→Scintil (Jim Theodoras) |
| Ranovus | 7 | CoreOptics→Ranovus (Hamid Arabzadeh, Georg Roell, Christoph Schulien); Lumentum, Inphi, BNR, and BTI individual paths |
| Avicena | 4 | IBM Watson→Avicena (Bardia Pezeshki); Kaiam→Avicena (Rob Kalman, Alex Tselikov); Samsung foundry→Avicena (Marco Chisari) |
| Salience Labs | 2 | Oxford ANE postdoc→Salience (Johannes Feldmann); OSE EIR→Salience (Vaysh Kewada) |
| Lumai | 4 | Oxford research/PhD/professor paths for Xianxin Guo, James Spall, and Alexander Lvovsky; Newcastle PhD→Lumai for Enzo D'Alessandro |
| iPronics | 4 | Named UPV iTEAM researchers José Capmany, Daniel Pérez López, Ivana Gasulla, and Prometheus Das Mahapatra→iPronics |
| SiPhi | 1 | Intel silicon-photonics product leadership→SiPhi (Marcus Yang); masked CTO excluded |

## Institutional lineage

Four institution edges meet the strict spinout gate:

- **CEA-Leti→Scintil Photonics (2018):** CEA-Leti's own history names the founders, the originating project, and reliance on its patent portfolio. This supports a formal spinout, not a specific patent-license agreement.
- **Oxford ANE Group→Salience Labs (2021):** Oxford's group roster establishes Johannes Feldmann's photonic-computing postdoc and subsequent Salience co-founder/CTO role. The edge is classified `lab_origin`; it is not an Oxford-wide IP claim.
- **University of Oxford→Lumai (2021):** Lumai explicitly calls itself an Oxford spinout. No IP-license terms or advisor relationships are inferred.
- **UPV iTEAM PRL→iPronics (2019):** UPV explicitly calls iPronics a spin-off from iTEAM's photonics laboratory, and its research yearbook names the founding researchers.

OpenLight's Aurrion/Juniper heritage is represented through named people only. The available official page says the Aurrion technology forms OpenLight's foundation, but this does not by itself define a formal corporate spinout, legal IP transfer, or wholesale team transfer.

## Evidence discipline

Official company biographies, official press releases, university pages, and the UPV research yearbook are grade A under `source_policy.md`. Each career statement is atomic and uses only the precision printed by its source. Company biographies that list several prior employers do not justify interpolated dates or continuous employment.

Degree and publication evidence are kept distinct from employment. In particular, Lumai biographies support Oxford degrees or research roles at the institution level, while the Oxford ANE roster supports Johannes Feldmann's specific group membership. No PI/advisor edge is inferred.

## Explicit exclusions

The structured dataset preserves eight non-edges covering unsupported team transfers, IP-license scope, advisor inference, and identity resolution. The most important is SiPhi: its website identifies the CTO only as **“Dr Y.”** The generic career description is insufficient to name or merge that person. Marcus Yang is included with an identity caution because the displayed “Marcus Y.” is paired with the public handle `ymarc`; a full official biography remains preferable.

## Gaps and next queue

1. Find incorporation, patent-assignment, or technology-transfer records for OpenLight and Scintil to define legal platform/IP continuity without inferring it from marketing histories.
2. Add current biographies for iPronics founders beyond Daniel Pérez López; the university artifact proves founding and iTEAM origin but does not establish everyone's present operating role.
3. Locate Oxford technology-transfer records for Salience and Lumai that distinguish university equity, IP ownership, and license scope.
4. Resolve SiPhi's leadership only after the company publishes full names; do not use surname-initial matching.
5. Add named Avicena personnel from the Nanosys fab acquisition only if an official source enumerates individuals; the acquisition's “engineering team” language is not person-level evidence.

## Graph-use recommendation

Founder, employment, technical-leadership, research-training, and narrowly defined spinout edges are appropriate for the conservative A/B layer. Publication/training edges should retain their lower talent-flow semantics, and none of the eight rejected transfer/license/identity claims should contribute to derived flow widths.
