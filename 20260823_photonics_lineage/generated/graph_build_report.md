# A/B Graph Build Report

Generated reproducibly from the canonical YAML snapshot. The output contains no build timestamp; input SHA-256 hashes in `graph_ab.json` identify the exact source state.

## Result

| Metric | Count |
|---|---:|
| Canonical organizations | 172 |
| Canonical people | 288 |
| Canonical edges | 556 |
| Canonical person→organization edges | 523 |
| Validated A/B person→organization edges | 518 |
| Person→organization edges removed by A/B/status filter | 5 |
| People in evidence graph | 282 |
| Organizations in evidence graph | 153 |
| People with fewer than two lineage institutions | 123 |
| People eligible for cross-institution sequencing | 159 |
| Person-level ordered contributions | 60 |
| Aggregated institution→institution talent flows | 52 |
| Excluded cross-organization person pairs | 255 |

Evidence grades: `A` 458, `B` 60.

Edge types: `advisor` 1, `collaboration` 1, `employment` 120, `executive_role` 46, `founder` 114, `patent_affiliation` 5, `publication_affiliation` 23, `research_training` 83, `technical_leadership` 125.

## Ordering policy

A talent flow is generated only when the latest possible date represented by an earlier edge's `end_date` is strictly before the earliest possible date represented by a later edge's `start_date`. Year and month precision are expanded to full intervals. Equal years/months therefore do not establish order. Biography list order, acquisition timing, current-role fields, and undated role language are not used to infer a sequence.

The HTML is a conservative Sankey-like institution-flow view plus a searchable table of every included person→organization evidence edge. Because only 52 institution flows meet the ordering rule, it should be read as a **dated-evidence prototype**, not an ecosystem-completeness map.

## Derived institution flows

- **Agere → InnoLight Technology Corporation** — weight 3.0; Sheng Liu
- **Agere → Opnext** — weight 1.0; Sheng Liu
- **Alcatel-Lucent Bell Labs → Ciena** — weight 1.0; Peter Winzer
- **Alcatel-Lucent Bell Labs → Nubis Communications** — weight 6.0; Guilhem de Valicourt, Peter Winzer
- **Advanced Micro Foundry → National Semiconductor Translation and Innovation Centre** — weight 1.0; Xianshu Luo
- **A*STAR Institute of Microelectronics → Advanced Micro Foundry** — weight 2.0; Guo-Qiang Lo, Xianshu Luo
- **A*STAR Institute of Microelectronics → Marvell Technology** — weight 1.0; Xiaoguang Tu
- **A*STAR Institute of Microelectronics → National Semiconductor Translation and Innovation Centre** — weight 1.0; Xianshu Luo
- **Bell Labs → Alcatel-Lucent Bell Labs** — weight 1.0; Robert Tkach
- **Bookham → Kotura** — weight 2.0; Andrew Rickman
- **Bookham → Rockley Photonics** — weight 3.0; Andrew Rickman
- **California Institute of Technology → Luminous Computing** — weight 2.0; Michael Hochberg
- **Caltech Nanofabrication Group → Luminous Computing** — weight 2.0; Michael Hochberg
- **Chengdu Kanghe Optoelectronics → Eoptolink Technology** — weight 2.0; 罗玉明
- **Chengdu Kanghe Optoelectronics → Sichuan Guangheng Communication Technology** — weight 1.0; 罗玉明
- **CoreOptics → Ranovus** — weight 3.0; Hamid Arabzadeh
- **ETH Zurich Institute of Electromagnetic Fields → imec** — weight 1.0; Christian Haffner
- **ETH Zurich Institute of Electromagnetic Fields → Polariton Technologies** — weight 6.0; Benedikt Baeuerle, Claudia Hoessbacher
- **ETH Zurich Institute of Electromagnetic Fields → Technische Universität Berlin** — weight 1.0; Maurizio Burla
- **ETH Zurich Institute of Electromagnetic Fields → CREOL, University of Central Florida** — weight 1.0; Yannick Salamin
- **ETH Zurich → ETH Zurich Institute of Electromagnetic Fields** — weight 1.0; Jürg Leuthold
- **ETH Zurich → Karlsruhe Institute of Technology** — weight 1.0; Jürg Leuthold
- **ETH Zurich → Lucent Bell Labs** — weight 1.0; Jürg Leuthold
- **Fiberxon (Chengdu) → Eoptolink Technology** — weight 1.0; 黄晓雷
- **Fiberxon (Chengdu) → Guangsheng Communications** — weight 1.0; 黄晓雷
- **Fiberxon (Chengdu) → Insten Technology** — weight 1.0; 黄晓雷
- **Ghent University → Photonics Research Group at Ghent University** — weight 1.0; Wim Bogaerts
- **Ghent University → imec** — weight 1.0; Wim Bogaerts
- **Ghent University → Luceda Photonics** — weight 3.0; Wim Bogaerts
- **Insten Technology → Eoptolink Technology** — weight 1.0; 黄晓雷
- **Intel Corporation → InnoLight Technology Corporation** — weight 3.0; Osa Chou-Shung Mok
- **Intel Corporation → Pine Photonics Communications** — weight 3.0; Osa Chou-Shung Mok
- **Intel Corporation → Uniwave Technology** — weight 3.0; Osa Chou-Shung Mok
- **Intel Silicon Photonics Solutions Group → ANELLO Photonics** — weight 3.0; Mario Paniccia
- **Leshan Radio Factory optical-communications branch → Eoptolink Technology** — weight 4.0; 高光荣, 罗玉明
- **Leshan Radio Factory optical-communications branch → Sichuan Guangheng Communication Technology** — weight 1.0; 罗玉明
- **Lucent Bell Labs → ETH Zurich Institute of Electromagnetic Fields** — weight 1.0; Jürg Leuthold
- **Massachusetts Institute of Technology → Ayar Labs** — weight 6.0; Milos Popovic, Vladimir Stojanovic
- **MIT Photonics and Modern Electro-Magnetics Group → Lightelligence** — weight 3.0; Yichen Shen
- **Pine Photonics Communications → InnoLight Technology Corporation** — weight 8.0; Hsing Hsien Kung, Sheng Liu, Weilong William Lee, Xiangzhong Wang
- **Pine Photonics Communications → Oplink Communications** — weight 1.0; Weilong William Lee
- **Pine Photonics Communications → WaveSplitter Technologies** — weight 1.0; Weilong William Lee
- **SDL → InnoLight Technology Corporation** — weight 3.0; Hsing Hsien Kung
- **SDL → Pine Photonics Communications** — weight 3.0; Hsing Hsien Kung
- **Sichuan Guangheng Communication Technology → Eoptolink Technology** — weight 2.0; 罗玉明
- **Stanford University → Silicon Light Machines** — weight 3.0; Olav Solgaard
- **Stanford University → Solgaard Lab** — weight 1.0; Olav Solgaard
- **Silicon Photonics Research Group at the University of Surrey → Kotura** — weight 2.0; Andrew Rickman
- **Silicon Photonics Research Group at the University of Surrey → Rockley Photonics** — weight 3.0; Andrew Rickman
- **University of California, Santa Barbara → Source Photonics** — weight 1.0; Yu-Heng Jan
- **UCSB Bowers Optoelectronics Research Group → Nexus Photonics** — weight 3.0; Chong Zhang
- **WaveSplitter Technologies → InnoLight Technology Corporation** — weight 1.0; Weilong William Lee

## Exclusions

- `missing_prior_end_or_later_start`: 214
- `overlapping_or_nonconclusive_date_precision`: 41

Full person/pair exclusions and their contributing edge IDs are stored in `graph_ab.json` under `sequence_exclusions`. People with fewer than two distinct lineage institutions never become sequence candidates and are counted separately above. Eligible person→organization relationships outside the configured lineage-bearing types remain visible in the evidence browser but are not used to derive flows.

## Validation

The builder aborts on missing canonical collections, schema-version disagreement, or missing evidence-source references for included edges. After writing, the build was checked by reparsing JSON, checking node/edge/flow references, and confirming that every included person edge is validated with grade A or B.
