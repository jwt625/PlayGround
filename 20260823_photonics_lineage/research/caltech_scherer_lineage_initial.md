# Caltech / Axel Scherer / Luxtera lineage: initial tranche

**Status:** evidence-backed staging dataset  
**As of:** 2026-08-23  
**Structured data:** `research/caltech_scherer_people_initial.yaml`

## Result

This initial tranche identifies **20 named people and 39 atomic edges**: 35 grade A and 4 grade B. Fifteen people are directly identified with Caltech's Scherer-led Nanofabrication Group; nine of those have thesis-level evidence naming Scherer as advisor. Eight people have individually supported Luxtera roles, of whom five are explicitly supported as founders/co-founders.

| Layer | Named evidence | Boundary enforced |
|---|---|---|
| Scherer-group leadership | Axel Scherer | Official Caltech roster establishes PI leadership; no dates are inferred. |
| Thesis-backed Scherer training | Cary Gunn, Michael Hochberg, Chuan-Cheng Cheng, Jingqing Huang, Uday Khankhoje, Oskar Painter, Michael Shearn, Jeremy Witzens, Tomoyuki Yoshie | Each Caltech degree edge is separate from the Scherer-group membership edge. |
| Roster-only group alumni | Tom Baehr-Jones, Jelena Vuckovic, Guangxi Wang, Marko Lončar, Dongyoon Oh | Recorded as group affiliation/training type `other`; no degree, advisor, or dates inferred. |
| Explicit Luxtera founders | Axel Scherer, Cary Gunn, Tom Baehr-Jones, Michael Hochberg, Alex Dickinson | Founder status requires explicit founder language; it is not inferred from employment or acknowledgments. |
| Other named Luxtera roles | Thierry Pinguet, Ron Horan, Vikram Sadagopan | Process-development, commercial leadership, and early technical-team employment remain distinct from founder status. |
| Caltech project support | Richard Seligman | University employment/project participation only; no Luxtera or Scherer-training edge. |

## Defensible paths

- **Axel Scherer → Caltech Nanofabrication Group → Luxtera:** Caltech's current roster establishes group leadership; a Caltech institutional feature explicitly identifies Scherer as a Luxtera co-founder.
- **Cary Gunn → Caltech PhD + Scherer group → Luxtera:** Gunn's primary dissertation describes pre-company Nanofabrication Group work and the founding of Luxtera to commercialize the silicon-photonics platform. This is the strongest lab-to-company path in the tranche.
- **Michael Hochberg → Caltech PhD + Scherer group → Luxtera → Elenion → Luminous:** the Caltech dissertation proves Scherer supervision; Luminous's official appointment biography explicitly supplies the three company claims.
- **Tom Baehr-Jones → Scherer group → Luxtera → Elenion → Tesselmax:** the Caltech roster proves group alumni status without a degree inference; SBIR.gov and Tesselmax's official bio support the founder edges.
- **Ron Horan: Luxtera → Cisco:** Cisco directly supplies the 2011 Luxtera start, VP Sales/Marketing roles, and post-acquisition Client Optics product-management role.
- **Vikram Sadagopan: Luxtera → Xscape:** Xscape's official appointment announcement calls him an early Luxtera technical-team member and establishes his 2025 VP Engineering appointment.

## Evidence discipline

All career relationships are atomic. The file deliberately does not collapse the following distinctions:

1. A Caltech PhD is not automatically Scherer-group membership.
2. A Scherer-group roster entry does not automatically establish a Caltech degree or Scherer as formal advisor.
3. Luxtera employment, technical leadership, and founder status are separate claims.
4. Cisco's acquisition of Luxtera does not prove that every Luxtera employee continued at Cisco.
5. Multiple people moving through the same institutions do not establish a `team_transfer` edge.

The U.S. Air Force Luxtera history is graded B: although hosted by an official government publisher, it is a retrospective narrative rather than a contemporaneous filing, patent, or individual primary biography. Caltech theses, official university pages, SBIR award records, and official company biographies/announcements are grade A.

## Explicit non-edges

- Thesis acknowledgments are not a Luxtera employee or founder roster.
- No Scherer-group → Luxtera cohort transfer is asserted.
- No intact Luxtera → Cisco workforce transfer is asserted.
- No Caltech degree or Scherer-advisor edge is inferred for people whose evidence proves only a company role or general project participation.

## Gaps and next queue

1. Locate a primary Luxtera corporate artifact or patent roster for Alex Dickinson and Thierry Pinguet; both currently rely on a grade-B government retrospective.
2. Resolve Luxtera formation dates and founder titles from contemporaneous filings rather than secondary histories.
3. Add named Cisco continuity only from official biographies, patents, or dated conference records; do not expand Ron Horan into a team claim.
4. Investigate downstream paths for the thesis-backed alumni—especially Oskar Painter, Marko Lončar, Jelena Vuckovic, and Chuan-Cheng Cheng—using official university/company biographies.
5. Revisit Tom Baehr-Jones's Caltech degrees only with a Caltech transcript, thesis, official alumni bio, or equivalent primary artifact. The current group roster supports membership but not degree details.

## Suitable graph use

The 39 A/B edges are suitable for a conservative lineage layer, subject to their uncertainty notes. Degree-to-university and training-to-group edges should remain visually distinct. No non-edge should contribute to a Sankey weight, and the repeated-person paths must not be aggregated into a team transfer without new named-cohort evidence.
