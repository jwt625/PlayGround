# A*STAR IME / Advanced Micro Foundry lineage: initial tranche

**Status:** evidence-backed staging dataset  
**As of:** 2026-08-23  
**Structured data:** `research/astar_ime_amf_people_initial.yaml`

## Result

This tranche identifies **20 named researchers, founders, inventors, and technical leaders**, **36 atomic edges**, and one directly supported formal spinout. All included edges are grade A. The dataset uses primary paper affiliations aggressively to expand the named research base without converting paper bylines into employment histories.

| Layer | Evidence-backed result |
|---|---|
| Institutional lineage | A*STAR is the umbrella; IME is the operating research institute. AMF's own release and A*STAR's history explicitly describe AMF as a 2017 IME spinout carrying forward the silicon-photonics platform. |
| Named founders | Xianshu Luo and Patrick Guo-Qiang Lo have official IME biographies supporting IME→AMF paths. A primary research-paper biography also identifies Guo-Qiang Lo as AMF founder and CTO; identity is resolved to Patrick Guo-Qiang Lo rather than duplicated. |
| IME→AMF person path | Chao Li has primary-paper affiliations at IME in 2017 and AMF in 2024. This is a dated affiliation sequence, not an inferred continuous employment or team-transfer claim. |
| Later destination | Xiaoguang Tu has an official IME biography supporting IME→Inphi→Marvell, where he leads the Singapore silicon-photonics team. Patrick Lo has a named 2026 GlobalFoundries technical-fellow role. |
| Technical base | The 2017 IME paper contributes twelve named publication affiliates; the 2024 optical-engine paper adds six IME affiliates and three AMF affiliates, with organizational affiliations kept separate. |
| Patent lineage | AMF patent US20250164829A1 links Patrick Lo, Xianshu Luo, and Shawn Yohanes Siew to AMF as inventors/assignee only; no employment inference is made from the patent. |

## Strongest paths

- **Patrick Guo-Qiang Lo: IME technical leadership (2004–2017) → AMF co-founder/president and CTO evidence → GlobalFoundries Senior Fellow.** The source set also resolves the Guoqiang Lo / Guo-Qiang Lo / Patrick Lo name variants through the distinctive career and patent sequence.
- **Xianshu Luo: IME research scientist (2010–2017) → AMF co-founder and research leadership (2017–2023) → IME principal scientist/head (from 2023) + NSTIC platform leadership.** This is the cleanest returnee loop in the tranche.
- **Xiaoguang Tu: IME → Inphi → Marvell.** IME's official 2026 biography supplies the sequence and current Distinguished Engineer/team-lead role.
- **Chao Li: IME publication affiliation (2017) → AMF publication affiliation (2024).** The two primary artifacts support a person-level institutional sequence, but not employment dates.

## Formal process lineage

The institutional edge is deliberately narrow: `org_astar_ime → org_amf`, `edge_type: spinout`, `spinout_basis: formal_spinout`, dated 2017. AMF's official release says it is a spin-off from IME/A*STAR; A*STAR's feature says the silicon-photonics program began at IME in 2006 and later became the AMF spin-off. This supports formal spinout and process continuity.

It does **not** establish:

- a named wholesale team transfer;
- a legal IP-license or assignment instrument;
- that every IME silicon-photonics process moved to AMF;
- that every AMF employee continued at GlobalFoundries after the 2025 acquisition.

## A*STAR versus IME

Paper bylines that print “Institute of Microelectronics, A*STAR” target the IME organization node. A*STAR remains the parent/umbrella node, not a duplicate employer edge. This prevents the same publication from being counted twice and preserves institute-level technical specificity.

## Publication and chronology discipline

The 2017 and 2024 Optica pages are primary paper artifacts, so their affiliation edges are grade A. They establish a named affiliation at publication, not a full employment interval. One chronology tension is preserved explicitly: Xiaoguang Tu's official bio says he joined Inphi in 2016, while the 2017 paper prints an IME affiliation. That may reflect publication lag, continuing collaboration, or dual affiliation; the dataset keeps the paper edge and does not extend IME employment beyond 2016.

## Gaps and next queue

1. Obtain AMF incorporation/shareholder records or contemporaneous founder bios to resolve whether any founders beyond Xianshu Luo and Patrick Guo-Qiang Lo should be modeled.
2. Locate an A*STAR or IPOS technology-transfer instrument defining the process/IP package moved into AMF.
3. Expand Chao Li's and other possible IME→AMF paths through patents, archived staff directories, and official biographies rather than byline inference.
4. Identify named Rain Tree Photonics links to IME/AMF from official biographies; the 2024 paper alone proves collaboration and separate affiliations, not career movement.
5. Add named post-acquisition AMF→GlobalFoundries roles only when official biographies or filings become available.

## Suitable graph use

The formal spinout, explicit founder/career edges, and primary publication/patent affiliations are suitable for a conservative evidence layer. Publication-affiliation edges should not be given ordinary employment weight, and the four rejected generic-transfer claims must never contribute to a Sankey or talent-flow aggregate.
