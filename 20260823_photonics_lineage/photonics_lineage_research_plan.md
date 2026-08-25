# Photonics / Optical-Interconnect Lineage Research Plan

## 1. Objective

Build a validated, person-level genealogy of the modern optics/photonics startup ecosystem, with emphasis on silicon photonics, optical interconnects, CPO/OIO, coherent optics, integrated lasers, photonic compute, and adjacent high-speed optoelectronics.

The end goal is to reconstruct a lineage Sankey / flow graph showing how technical talent, research groups, startup teams, and acquired organizations propagated into the current photonics industry.

This is **not** limited to formal spinouts. A person having worked in a relevant technical organization counts as lineage evidence if the employment is verified and technically relevant.

The core graph should capture:

- academic lab → startup formation
- research institute → startup formation
- company → founder / technical leader → startup
- company → company personnel migration
- startup → acquisition → acquiring company
- startup → team re-formation into another startup
- IP / technology spinout or license
- advisor / co-founder relationships when technically meaningful
- major cross-pollination between industry "schools"

The analysis should be bidirectional:

1. **Downstream search from major labs, companies, and historical technical groups**
   - identify notable alumni
   - trace their later employers, startups, and acquisitions

2. **Upstream search from recent startups and acquisitions**
   - identify founders, CTOs, VPs, directors, fellows, principal engineers, early technical staff
   - reconstruct prior employment and academic history
   - identify the institutional roots behind each company

3. **Timeline completion for every included individual**
   - reconstruct the fullest publicly documented professional timeline, not only photonics-highlight roles
   - collect exact or partial start/end dates, concurrent roles, employer business units, titles, locations, education, and current status
   - inspect LinkedIn or equivalent public professional profiles whenever a credible identity match is available
   - retain LinkedIn-only facts as grade-C asserted claims pending corroboration instead of discarding the timeline information
   - record upstream and downstream coverage separately so a diaspora study is not mistaken for a complete institutional labor-flow study

---

# 2. Research Questions

## 2.1 Primary questions

1. Which companies and research groups have produced the largest number of founders and key technical leaders in modern photonics?
2. What are the strongest institutional lineages behind recent optical-interconnect startups?
3. How much of the current startup ecosystem descends from:
   - Intel Silicon Photonics
   - Bell Labs / Lucent / Alcatel-Lucent
   - UCSB / John Bowers
   - Stanford photonics groups
   - MIT photonics groups
   - Columbia photonics groups
   - Caltech / Luxtera
   - imec / Ghent
   - ETH Zurich / Leuthold
   - Bookham / Rockley
   - Acacia / Mintera
   - NeoPhotonics
   - Finisar / JDSU / Oclaro / Lumentum
   - Cisco optical organizations
   - Broadcom optical organizations
   - Huawei / Futurewei
   - A*STAR / IME
4. Which organizations act as major "mixing nodes" where talent from multiple historical schools converged?
5. Which recent acquisitions represent:
   - talent acquisition
   - technology acquisition
   - platform acquisition
   - product acquisition
   - team recombination
6. How many startups can be linked back to Intel Silicon Photonics directly or indirectly?
7. What are the most important China-based photonics startups with lineage through:
   - Intel
   - Huawei / Futurewei
   - NeoPhotonics
   - InnoLight
   - Broadcom
   - Cisco / Acacia
   - JDSU / Finisar / Oclaro / Lumentum
   - imec / Ghent
8. Which individual engineers appear repeatedly across multiple high-value companies and exits?

---

# 3. Scope

## 3.1 Technology scope

Include organizations working in:

- silicon photonics
- III-V / Si heterogeneous integration
- TFLN / thin-film electro-optics
- integrated lasers
- coherent optical transceivers
- datacenter optics
- CPO / NPO / OIO
- high-speed optical engines
- optical SerDes / photonic I/O
- photonic switching
- optical compute / photonic AI accelerators
- optical interposers
- optical network ASIC / DSP / mixed-signal ICs when tightly coupled to photonics
- microLED / optical interconnect if relevant to datacenter links
- FMCW LiDAR / integrated sensing only when the company or team has meaningful overlap with communication SiPh
- optical packaging / assembly startups if they are part of the same talent network

Avoid unrelated optics unless the people or company materially feed into the target ecosystem.

## 3.2 Time scope

Primary period:

- 1995–2026

Historical roots may extend earlier when useful, especially:

- Bell Labs
- Lucent
- Bookham
- JDSU
- Finisar
- early silicon-photonics programs
- early integrated-optics academic groups

Special emphasis:

- startups founded after ~2010
- acquisitions after ~2015
- startups / acquisitions from 2023–2026

---

# 4. Core Graph Model

The graph should be built from **people-level evidence**, even when the final visualization aggregates to institutional flows.

## 4.1 Node types

### Organization nodes
- company
- startup
- university
- academic lab
- industrial research lab
- government / national lab
- research institute
- acquired business unit

### Person nodes
- founder
- co-founder
- CEO
- CTO
- VP Engineering
- VP Photonics
- director
- fellow
- principal engineer
- distinguished engineer
- early technical employee
- technical advisor
- PI / professor
- research scientist

### Event nodes, optional
- acquisition
- spinout
- major team formation
- IP transfer / license

---

# 5. Edge Types

Use explicit edge semantics.

| Edge type | Meaning |
|---|---|
| `employment` | Person worked at organization |
| `research_training` | Person trained in lab / PI group |
| `founder` | Person founded or co-founded organization |
| `executive_role` | Person held key technical or business leadership role |
| `technical_leadership` | Principal / director / fellow / senior technical leadership |
| `spinout` | Company formally spun out of another organization |
| `ip_license` | Technology or IP licensed from source |
| `acquisition` | Company acquired by another company |
| `team_transfer` | Significant team moved together to another organization |
| `advisor` | Technical advisor or board member |
| `investment` | Strategic or venture investment, if materially relevant |
| `collaboration` | Sustained research / product collaboration |
| `publication_affiliation` | Person appears in publication under organization affiliation |
| `patent_affiliation` | Person appears as inventor / assignee linkage |

---

# 6. Visualization and Aggregation Strategy

The research database should remain person-level.

The primary visualization should be an uncertainty-aware organization exchange network, not a global Sankey. It must show confirmed directed transitions, shared-person relationships with unknown direction, and bidirectional exchange as distinct states. Selecting an organization should reveal the contributing people and their source-backed timelines.

A person-level timeline/storyline view is the primary detail view. Unknown interval endpoints should remain open or fuzzy rather than causing the affiliation to disappear. Acquisition, spinout, and team-transfer events may appear as separate event markers but must not manufacture person movement.

A Sankey may remain as a secondary, explicitly scoped **founder ancestry** view. It must not be described as a complete talent-flow map, and deep ancestry must be distinguished from consecutive career transitions.

Example:

```text
Intel Silicon Photonics
    ├── Vivek Raghunathan ──> Xscape
    ├── Yoel Chetrit ───────> DustPhotonics
    ├── Brian Koch ─────────> Aurrion / Juniper / Quintessent
    ├── Jie Sun ────────────> LightIC
    └── Mario Paniccia ─────> ANELLO
```

Possible founder-ancestry rendering:

```text
Intel SiPh ───────> Xscape
          ├───────> DustPhotonics
          ├───────> Quintessent
          ├───────> LightIC
          └───────> ANELLO
```

## 6.1 Recommended edge weights

Use unique named people as the default link width. Do not use acquisition value as lineage width and do not present role-weighted values as headcount.

Suggested talent weights:

- founder / co-founder: `3.0`
- CTO / VP / senior technical executive: `2.0`
- director / principal / fellow / key early engineer: `1.0`
- advisor / board technical role: `0.25–0.5`

Store the raw people behind every aggregate edge so it can be audited.

Role-weighted influence may be offered as a non-default analytical toggle. It must be labeled as an index, and one person must not be silently multiplied across transitive ancestry paths.

## 6.2 Transition semantics

Derived organization relationships must be classified as:

- `confirmed`: date bounds or an explicit move statement establish direction;
- `semantic`: founder or biography semantics suggest direction but dates are incomplete;
- `unknown`: the same validated person is affiliated with both organizations but order is unresolved;
- `conflicted`: sources support incompatible chronology.

Directness is separate from direction. A confirmed earlier affiliation can still be nonconsecutive ancestry if intermediate organizations are known. Only evidence-supported consecutive moves should be labeled direct talent transitions.

## 6.3 Required interaction

- Search and select any organization or person.
- Default to a one- or two-hop ego network rather than an unreadable global graph.
- Filter by time, technology, geography, edge type, evidence grade, chronology confidence, and role class.
- Toggle parent/business-unit and university/lab display clustering without rewriting canonical endpoints.
- Show unique-person counts, confirmed inbound, confirmed outbound, bidirectional exchange, and direction-unknown adjacency separately.
- On selection, expose every contributing person, atomic edge, date observation, evidence grade, and source link.

---

# 7. Evidence Standard

Every career edge should carry an evidence grade.

## A — Primary
Examples:
- official company bio
- university bio
- SEC filing
- acquisition filing
- CV
- official conference bio
- archived company page
- paper affiliation
- patent
- official press release

## B — Strong secondary
Examples:
- IEEE
- Optica
- Nature / Science profile
- reputable technical publication
- authoritative interview
- respected industry conference organizer

## C — Weak but useful
Examples:
- LinkedIn
- Crunchbase
- The Org
- company directory
- well-maintained startup database

Must be corroborated where possible.

## D — Unverified media claim
Examples:
- Chinese startup PR
- single news article
- promotional profile
- reposted biography

## X — Inferred
Do not include in validated Sankey.

The default final graph should render only `A` and `B`, with an optional layer for `C`.

---

# 8. Initial Company Seeds

## 8.1 Recent / current AI-optics and datacenter photonics

- Ayar Labs
- Lightmatter
- Xscape Photonics
- Celestial AI
- Nubis Communications
- DustPhotonics
- Enosemi
- Quintessent
- OpenLight
- Scintil Photonics
- Ranovus
- Avicena
- Hyperlume
- Lightelligence
- Luminous Computing
- Polariton Technologies
- iPronics
- Salience Labs
- Lumai
- Lumotive
- LightIC / 摩尔芯光
- SiPhi
- InfiniLink
- Advanced Micro Foundry / AMF
- Cloud Light
- Yingweixin / 英伟芯
- Inxun / 映讯芯光
- Bopu Semiconductor / 孛璞半导体
- Liangyin / 量引科技
- Aoke / 傲科光电

## 8.2 Historically important startups / companies

- Luxtera
- Kotura
- Lightwire
- Aurrion
- Elenion
- Rockley Photonics
- Acacia Communications
- Mintera
- Inphi
- Mellanox optical groups
- Juniper silicon photonics
- Cisco silicon photonics
- NeoPhotonics
- Finisar
- JDSU
- Oclaro
- Lumentum
- Bookham
- Avanex
- Opnext
- Source Photonics
- InnoLight
- Eoptolink
- Broadex
- Accelink
- Hisense Broadband
- Huawei optical / silicon photonics groups
- Futurewei
- Nokia Bell Labs
- Alcatel-Lucent Bell Labs
- Broadcom optical groups
- Marvell optical groups

---

# 9. Acquisition Seeds

The acquisition-first workstream should include at minimum:

- Luxtera → Cisco
- Lightwire → Cisco
- Acacia → Cisco
- Kotura → Mellanox
- Mellanox → NVIDIA
- Aurrion → Juniper
- Elenion → Nokia
- NeoPhotonics → Lumentum
- Oclaro → Lumentum
- Finisar → II-VI / Coherent
- Inphi → Marvell
- Cloud Light → Lumentum
- Enosemi → AMD
- Nubis → Ciena
- Hyperlume → Credo
- AMF → GlobalFoundries
- InfiniLink → GlobalFoundries
- Celestial AI → Marvell
- DustPhotonics → Credo
- Polariton → Marvell

For each acquisition capture:

- announcement date
- close date
- acquisition value
- cash / stock / earnout structure
- target employees, if disclosed
- founders / technical leadership
- core technology
- integration status
- whether team continued at acquirer
- subsequent founders who emerged from the acquired team

---

# 10. Laboratory / Research Group Seeds

## 10.1 Intel

### Intel Silicon Photonics / Photonics Technology Lab
Seed people:

- Mario Paniccia
- Haisheng Rong
- Richard Jones
- Mario Fiorentino
- Ansheng Liu
- Drew Alduino
- Robert Blum
- Yoel Chetrit
- Jie Sun
- Vivek Raghunathan
- Brian Koch
- Michael Davenport
- Matt Sysak
- Marcus Yang
- Li Ruolin / 李若林
- Nie Hui / 聂辉
- Guo Peng / 郭鹏

Expand using:

- old Intel SiPh paper author lists
- OFC / ECOC presentations
- Intel archived product pages
- patents
- conference bios
- LinkedIn / company bios for downstream history

---

## 10.2 Bell Labs / Lucent / Alcatel-Lucent / Nokia Bell Labs

Seed people:

- Peter Winzer
- Guilhem de Valicourt
- Benny Mikkelsen
- Jürg Leuthold
- Phil Winterbottom

Expand from:

- Bell Labs coherent optics groups
- optical networking teams
- photonic integration programs
- WDM / transceiver programs
- Mintera-era personnel
- Lucent optical component groups

---

## 10.3 UCSB / John Bowers

Seed people / organizations:

- John Bowers
- Alex Fang
- Brian Koch
- Alan Liu
- Michael Davenport
- Aurrion
- Quintessent
- Nexus Photonics
- Aerius Photonics
- Terabit Technologies
- Calient

Expand using:

- dissertation alumni
- lab alumni pages
- former postdocs
- UCSB tech-transfer records
- III-V/Si heterogeneous-integration paper author lists

---

## 10.4 MIT

### Dirk Englund group
- Nicholas Harris
- Darius Bunandar
- Lightmatter
- related quantum / photonic compute spinouts

### Rajeev Ram / Vladimir Stojanovic ecosystem
- Chen Sun
- Mark Wade
- Milos Popovic
- Ayar Labs
- related optical-I/O programs

### Marin Soljačić / Joannopoulos ecosystem
- Yichen Shen
- Lightelligence
- related photonic-compute founders

Expand via theses, lab pages, startup press, and MIT News.

---

## 10.5 Columbia

### Michal Lipson
- Xscape
- prior students / postdocs entering industry

### Keren Bergman
- Xscape
- optical networking alumni

### Alexander Gaeta
- Xscape
- nonlinear / frequency-comb photonics alumni

### Yoshi Okawachi
- Xscape-related lineage

Expand via:
- group alumni pages
- paper author lists
- theses
- conference bios
- LinkedIn / startup bios

---

## 10.6 Caltech

### Axel Scherer group
Seed:
- Luxtera founders / early staff
- Cary Gunn
- Alex Dickinson
- related integrated-photonics alumni

Goal:
- distinguish true Caltech lineage from later industry hires

---

## 10.7 Stanford

Treat Stanford as a multi-group research lineage rather than a single lab. Candidate downstream relationships must be established person by person; do not infer startup lineage from Stanford attendance, coauthorship, or thematic similarity alone.

### Shanhui Fan group
- nanophotonics and photonic topology
- nonreciprocal photonics
- inverse design and AI-enabled photonics
- trace doctoral students, postdocs, research staff, and later technical founders / leaders

### Jelena Vučković group
- nanoscale and quantum photonics
- inverse-designed photonic devices
- integrated photonic platforms
- distinguish formal group training from Stanford affiliation or collaboration

### Olav Solgaard group
- optical MEMS and scanning systems
- photonic sensing and integrated optical systems
- identify communications / interconnect-relevant alumni without pulling unrelated optics into scope

### David A. B. Miller ecosystem
- optical interconnects and switching
- photonic information processing
- device-to-system scaling and energy limits
- trace academic trainees and industry collaborators separately

### James S. (Jim) Harris Jr. group
- III-V semiconductor materials and epitaxy
- optoelectronic devices and integrated lasers
- identify alumni feeding semiconductor, laser, detector, and heterogeneous-integration organizations
- **TODO — deeper alumni pass (research paused):** validate Harris-group membership and downstream lineage for Zongjian (Apple), Hong Liu (Google), and Yijie Huo (Vertilite co-founder). Resolve full names and same-name ambiguity from primary sources before creating canonical people or edges.
- **TODO — roster depth:** mine the complete Harris former-student/thesis roster, then validate downstream employment, patents, technical leadership, and company formation person by person. The initial Seth Bank / Vijit Sabnis sample is not adequate coverage of this group.

### Stephen E. (Steve) Harris lineage
- nonlinear and quantum optics
- coherent light-matter interactions
- trace historically relevant students and collaborators into modern integrated photonics only where direct evidence exists

### Robert L. (Bob) Byer lineage
- lasers, nonlinear optics, and precision photonics
- technology transfer and industrial translation
- reconstruct older Stanford laser / photonics alumni paths that may act as upstream roots

### Martin M. (Marty) Fejer group
- nonlinear optics and quasi-phase matching
- ferroelectric and integrated nonlinear photonics
- examine connections to lithium-niobate and nonlinear-photonic companies as research targets, not assumed spinouts

Expand using:
- official lab rosters and archived group pages
- Stanford Profiles and faculty CVs
- dissertations, thesis-advisor records, and paper affiliations
- Stanford Office of Technology Licensing records
- patents and named license / spinout documentation
- downstream official company and conference biographies

Goals:
- build group-specific alumni lists rather than one undifferentiated Stanford node
- test which groups produced founders, early technical employees, or major industry leaders
- identify cross-group and Stanford-to-industry mixing while separating training, employment, collaboration, and licensing evidence

---

## 10.8 ETH Zurich / Jürg Leuthold

Seed:
- Wolfgang Heni
- Claudia Hoessbacher
- Benedikt Bäuerle
- Polariton Technologies

Trace:
- ETH group
- Leuthold's Bell Labs ancestry
- later startup and acquisition outcomes

---

## 10.9 imec / Ghent

Seed:
- Caliopa
- Huawei acquisition
- Luceda Photonics
- silicon-photonics foundry ecosystem
- major alumni moving to startups / fabs / design houses

Trace:
- Ghent Photonics Research Group
- imec silicon photonics
- Huawei Belgium R&D

---

## 10.10 A*STAR / IME

Seed:
- Advanced Micro Foundry
- GlobalFoundries acquisition
- Singapore SiPh foundry ecosystem
- former IME silicon-photonics leaders

Goal:
- reconstruct IME → AMF personnel / process lineage

---

## 10.11 Southampton / Surrey / Bookham / Rockley

Seed:
- Graham Reed
- Andrew Rickman
- Bookham
- Rockley Photonics

Goal:
- trace UK silicon-photonics / integrated-optics school into modern startups

---

# 11. China-Focused Seeds

Use Chinese-language primary and secondary sources aggressively.

## 11.1 Company seeds

- 华为 / Huawei
- Futurewei
- 中际旭创 / InnoLight
- 新易盛 / Eoptolink
- 博创科技 / Broadex
- 光迅科技 / Accelink
- 索尔思 / Source Photonics
- 海信宽带 / Hisense Broadband
- 天孚通信 / Suzhou TFC Optical Communication (user shorthand: Tian Fu)
- 光库科技 / Advanced Fiber Resources (Zhuhai) (AFR)
- 苏州易缆微半导体技术有限公司 / Suzhou InnovSemi (user shorthand: Suzhou Yi Lan Wei)
- 苏州苏纳光电 / SUNA Optoelectronics (user shorthand: Su Na Guang Dian)
- 摩尔芯光 / LightIC
- 映讯芯光 / Inxun
- 英伟芯
- 孛璞半导体
- 量引科技
- 傲科光电
- 曦智科技 / Lightelligence
- other recent CPO / OIO / optical-engine startups

## 11.2 Search patterns

Use queries such as:

```text
英特尔 硅光 创始人
前英特尔 硅光
Intel silicon photonics 创业
硅光 创始人 英特尔
硅光 首席工程师 创业
华为 硅光 创业
前华为 光芯片 创始人
前思科 光模块 创业
前NeoPhotonics 创始人
前Finisar 创业
前JDSU 创业
前Oclaro 创业
前Broadcom 硅光
中际旭创 离职 创业
光迅 离职 创业
```

## 11.3 Validation rule

Do not rely on generic claims such as:

> "核心团队来自 Intel, Huawei, Broadcom"

unless individual names are found and employment can be validated.

---

# 12. Person Seeds

The first broad roster should target approximately **150–250 people**.

## 12.1 Intel-centered seeds

- Mario Paniccia
- Haisheng Rong
- Richard Jones
- Mario Fiorentino
- Ansheng Liu
- Drew Alduino
- Robert Blum
- Yoel Chetrit
- Jie Sun
- Vivek Raghunathan
- Brian Koch
- Michael Davenport
- Matt Sysak
- Marcus Yang
- Li Ruolin
- Nie Hui
- Guo Peng

## 12.2 Bell Labs-centered seeds

- Peter Winzer
- Guilhem de Valicourt
- Benny Mikkelsen
- Jürg Leuthold
- Phil Winterbottom

## 12.3 UCSB-centered seeds

- John Bowers
- Alex Fang
- Alan Liu
- Brian Koch
- Michael Davenport

## 12.4 MIT / Columbia / Caltech seeds

- Nicholas Harris
- Darius Bunandar
- Mark Wade
- Chen Sun
- Vladimir Stojanovic
- Milos Popovic
- Rajeev Ram
- Yichen Shen
- Michal Lipson
- Keren Bergman
- Alexander Gaeta
- Yoshi Okawachi
- Cary Gunn
- Alex Dickinson

## 12.5 Stanford-centered seeds

- Shanhui Fan
- Jelena Vučković
- Olav Solgaard
- David A. B. Miller
- James S. Harris Jr. / Jim Harris
- Stephen E. Harris / Steve Harris
- Robert L. Byer / Bob Byer
- Martin M. Fejer / Marty Fejer

For each person, treat lab membership, advisor relationships, company formation, advisory work, IP licensing, and industry collaboration as separate claims requiring direct evidence.

## 12.6 Recent startup leadership seeds

Add founders / CTOs / photonics VPs from:

- Celestial AI
- Nubis
- Xscape
- Lightmatter
- Ayar
- Enosemi
- Dust
- Hyperlume
- AMF
- InfiniLink
- Polariton
- Quintessent
- OpenLight
- Scintil
- Ranovus
- Avicena
- Salience
- Lumai
- Lightelligence
- Cloud Light
- SiPhi
- LightIC

---

# 13. Research Phases

## Phase 0 — Schema and source discipline

### Goal
Create normalized data structures before deeper research.

### Tasks
- define organization IDs
- define person IDs
- define edge semantics
- define confidence grading
- define canonical date formats
- define source provenance format
- define technology taxonomy
- define geography taxonomy

### Deliverables
- `schema.yaml`
- `organizations_seed.yaml`
- `people_seed.yaml`
- `edge_types.yaml`
- `source_policy.md`

---

## Phase 1 — Acquisition-first reverse genealogy

### Goal
Cover all recent / important acquisitions first, especially those in the acquisition-value chart.

### Work
For each acquired target:
1. identify founders
2. identify CTO / VP Engineering / photonics leaders
3. identify early technical team
4. reconstruct prior employers / labs
5. identify later destinations after acquisition
6. capture acquisition metadata

### Priority targets
1. InfiniLink
2. Hyperlume
3. Cloud Light
4. AMF
5. Inphi
6. Celestial AI
7. DustPhotonics
8. Nubis
9. Enosemi
10. Polariton

### Deliverables
- `acquisitions.yaml`
- `acquisition_people.yaml`
- briefing memo with strongest genealogical findings

---

## Phase 2 — Intel Silicon Photonics diaspora

### Goal
Build the most complete Intel SiPh alumni graph possible.

### Method
Start from:
- old Intel paper author lists
- patents
- OFC / ECOC presentations
- archived Intel team pages
- company bios

For each person:
- academic background
- Intel role
- Intel group
- years
- technical specialty
- next employer
- startup roles
- acquisitions
- current status

### Deliverables
- `intel_siph_people.yaml`
- `intel_siph_edges.json`
- Intel SiPh lineage summary
- candidate Sankey slice

---

## Phase 3 — Historical commercial-company diaspora

### Goal
Trace high-impact legacy photonics companies forward.

### Seeds
- Luxtera
- Kotura
- Lightwire
- Aurrion
- Elenion
- Rockley
- Acacia
- Mintera
- Inphi
- NeoPhotonics
- Finisar
- JDSU
- Oclaro
- Bookham

### Method
- founder biographies
- archived leadership pages
- patents
- acquisition filings
- early employee LinkedIn profiles
- conference speaker bios

### Deliverables
- `legacy_company_people.yaml`
- `legacy_company_edges.json`
- company-to-startup downstream map

---

## Phase 4 — Academic lab spinout trees

### Goal
Map major research "schools" into industry.

### Labs
- UCSB / Bowers
- MIT / Englund
- MIT / Ram
- MIT / Soljačić
- Columbia / Lipson
- Columbia / Bergman
- Columbia / Gaeta
- Caltech / Scherer
- Stanford / Fan
- Stanford / Vučković
- Stanford / Solgaard
- Stanford / Miller
- Stanford / Jim Harris
- Stanford / Steve Harris
- Stanford / Byer
- Stanford / Fejer
- ETH / Leuthold
- imec / Ghent
- A*STAR / IME
- Southampton / Reed
- other high-yield labs discovered during research

### Deliverables
- `labs.yaml`
- `lab_alumni.yaml`
- lab → startup edge set
- short briefing for each major school
- Stanford group-by-group alumni map and a candidate-lineage validation queue

---

## Phase 5 — China / Huawei / overseas-returnee ecosystem

### Goal
Map the China-specific lineage that is poorly represented in English-language databases.

### Workstreams
- Intel → China startup founders
- Huawei / Futurewei → China startup founders
- NeoPhotonics → China
- InnoLight → startup alumni
- Broadcom → China
- Cisco / Acacia → China
- Finisar / JDSU / Oclaro / Lumentum → China
- imec / Ghent → Huawei / China

### Deliverables
- `china_photonics_people.yaml`
- `china_photonics_edges.json`
- bilingual company aliases
- evidence quality flags
- list of unresolved claims needing manual verification

---

## Phase 6 — Forward search from key people

### Goal
Discover unknown startups rather than merely validating a predetermined list.

### Method
For each seed person:
- current employer
- startup founding history
- board / advisor roles
- patents / papers after leaving source organization
- co-founders
- repeated collaborators
- team migration

This phase should surface companies that were not in the original seed list.

### Deliverables
- `newly_discovered_orgs.yaml`
- `people_forward_edges.json`
- prioritized list of newly discovered startups

---

## Phase 7 — Network validation and deduplication

### Goal
Resolve conflicting biographies and eliminate false lineage.

### Tasks
- normalize name variants
- normalize company names after mergers / rebrands
- resolve overlapping employment dates
- distinguish contractor / advisor from employee
- distinguish corporate affiliation from SiPh-specific technical affiliation
- verify acquisition dates / values
- downgrade unsupported claims
- flag uncertain group membership

### Deliverables
- `validation_report.md`
- `conflicts.yaml`
- `canonical_people.json`
- `canonical_organizations.json`
- `canonical_edges.json`

---

## Phase 8 — Sankey / graph reconstruction

### Goal
Produce the final visual lineage map.

### Outputs
1. **institution-level Sankey**
2. **person-level interactive graph**
3. **acquisition overlay**
4. **Intel-only Sankey**
5. **Bell Labs-only Sankey**
6. **academic-lab lineage Sankey**
7. **Stanford lineage Sankey**
8. **China lineage Sankey**
9. **recent AI-optics startup Sankey**

### Optional visual encodings
- edge width = talent weight
- edge opacity = evidence confidence
- node size = number of downstream founders / technical leaders
- acquisition marker = transaction event
- node outline = acquired / active / defunct
- country / region = optional categorical tag
- technology = optional node color / facet

---

# 14. Workstreams for Parallel Subagents

## Workstream A — Intel SiPh alumni
Output:
- 30–60 validated people
- downstream startups
- source citations

## Workstream B — Bell Labs / Acacia / coherent-optics lineage
Output:
- Bell Labs → Mintera → Acacia
- Bell Labs → Nubis
- Bell Labs → other startups
- post-acquisition alumni

## Workstream C — UCSB / Bowers
Output:
- academic alumni
- Aurrion
- Juniper
- Quintessent
- integrated-laser startups

## Workstream D — MIT / Columbia photonic-compute ecosystem
Output:
- Lightmatter
- Ayar
- Lightelligence
- Xscape
- related alumni

## Workstream E — Recent acquisitions
Output:
- all acquisitions 2023–2026
- founders
- technical leadership
- upstream careers

## Workstream F — China / Huawei
Output:
- Chinese-language founder biographies
- foreign-company lineage
- Huawei / Futurewei diaspora
- confidence grading
- **TODO — incumbent expansion (research/ingestion paused):** add TFC Optical, Advanced Fiber Resources / 光库科技, InnovSemi / 易缆微, and SUNA Optoelectronics / 苏纳光电. Preserve InnovSemi versus its InnovOpto subsidiary as distinct entities; treat SUNA's 有限公司→股份有限公司 change as legal-form continuity pending source-level canonical review.

## Workstream G — Legacy company diaspora
Output:
- Luxtera
- Kotura
- Rockley
- Elenion
- NeoPhotonics
- Finisar
- JDSU
- Oclaro
- Inphi

## Workstream H — Stanford photonics lineages
Output:
- group-specific alumni rosters for Fan, Vučković, Solgaard, Miller, Jim Harris, Steve Harris, Byer, and Fejer
- validated training and downstream employment / founder edges
- candidate IP-license and formal-spinout relationships requiring confirmation
- explicit separation of Stanford attendance, lab membership, collaboration, and company lineage
- **TODO — Jim Harris depth correction (research/ingestion paused):** explicitly investigate Zongjian→Apple, Hong Liu→Google, and Yijie Huo→Vertilite, then expand to the full former-student roster. Do not ingest shorthand identities until full-name disambiguation and Harris-group membership are directly sourced.

---

# 15. Search Methodology

For each person, use multiple query directions.

## 15.1 Upstream

```text
"<person>" Intel
"<person>" silicon photonics
"<person>" Bell Labs
"<person>" UCSB
"<person>" Stanford photonics
"<person>" Stanford thesis advisor
"<person>" MIT photonics
"<person>" previous company
"<person>" biography
"<person>" CV
"<person>" patent
"<person>" conference bio
```

## 15.2 Downstream

```text
"<person>" founder
"<person>" startup
"<person>" co-founder
"<person>" CTO
"<person>" current
"<person>" acquisition
"<person>" joined
```

## 15.3 Company-centered

```text
"<company>" founders
"<company>" CTO
"<company>" VP photonics
"<company>" leadership
"<company>" founding team
"<company>" acquired
"<company>" early employee
"<company>" silicon photonics team
```

## 15.4 Lab-centered

```text
"<lab>" alumni startup
"<PI>" spinout
"<PI>" startup founder
"<PI>" former student founder
"<lab>" industry alumni
```

## 15.5 Publication mining

For historically important technical groups:
- download representative papers
- extract author lists
- map affiliations
- search each author downstream

High-yield publication corpora:
- Intel SiPh
- Bell Labs coherent optics
- UCSB III-V/Si heterogeneous integration
- Stanford nanophotonics, optical-interconnect, III-V, nonlinear-optics, and optical-MEMS groups
- MIT optical I/O
- Columbia silicon / nonlinear photonics
- imec SiPh foundry work

---

# 16. Recommended Data Formats

## 16.1 `people.yaml`

```yaml
- id: person_vivek_raghunathan
  name: Vivek Raghunathan
  aliases: []
  current_role:
    organization: Xscape Photonics
    title: Co-founder / CEO
  education:
    - institution: MIT
      degree: PhD
      field: null
      confidence: A
  career:
    - organization: Intel
      group: Silicon Photonics
      role: Product integration / commercialization
      start_year: null
      end_year: null
      photonics_relevance: direct
      confidence: A
      sources:
        - url: null
          type: official_bio
    - organization: Rockley Photonics
      role: null
      photonics_relevance: direct
      confidence: A
    - organization: Broadcom
      role: Senior Principal / Product Architect
      photonics_relevance: direct
      confidence: A
  startups:
    - organization: Xscape Photonics
      relationship: founder
```

---

## 16.2 `organizations.yaml`

```yaml
- id: org_intel_silicon_photonics
  name: Intel Silicon Photonics
  aliases:
    - Intel Photonics Technology Lab
    - Intel Silicon Photonics Product Division
  type: industrial_lab
  parent: Intel
  country: US
  technologies:
    - silicon_photonics
    - integrated_lasers
    - datacenter_optics
  active_period:
    start_year: null
    end_year: null
```

---

## 16.3 `edges.json`

```json
[
  {
    "source": "org_intel_silicon_photonics",
    "target": "org_xscape_photonics",
    "edge_type": "talent_flow",
    "people": ["person_vivek_raghunathan"],
    "weight": 3.0,
    "confidence": "A",
    "notes": "Founder worked on Intel SiPh commercialization before Rockley and Broadcom."
  }
]
```

---

## 16.4 `acquisitions.yaml`

```yaml
- target: DustPhotonics
  acquirer: Credo
  announced_date: null
  close_date: null
  value:
    headline_usd: null
    upfront_usd: null
    earnout_usd: null
    stock_component: null
  technology:
    - silicon_photonics
    - optical_interconnect
  founders: []
  technical_leaders: []
  sources: []
  confidence: A
```

---

## 16.5 `sources.yaml`

```yaml
- id: src_001
  url: https://...
  title: ...
  publisher: ...
  date: ...
  source_type: official_company_bio
  evidence_grade: A
  archived: false
  notes: ...
```

---

# 17. Research Briefing Deliverable

The written briefing should contain:

## Executive summary
- major lineage clusters
- most important findings
- surprising cross-company flows
- strongest Intel diaspora examples
- strongest Bell Labs diaspora examples
- strongest validated Stanford group lineages and highest-priority candidate links
- strongest China lineage examples

## Institutional lineage sections
- Intel SiPh
- Bell Labs
- UCSB
- Stanford
- MIT
- Columbia
- Caltech
- imec / Ghent
- ETH
- legacy commercial optical companies
- China / Huawei

## Recent acquisition section
- acquisition table
- founder / technical-lead genealogy
- talent retained by acquirer
- subsequent spinouts

## Cross-pollination section
Identify companies with multiple major ancestry streams.

Examples to test:
- Xscape
- Quintessent
- Enosemi
- Celestial AI
- OpenLight

## Unknowns / unresolved claims
- conflicting dates
- weak PR-only claims
- missing founder biographies
- likely but unverified team migrations

---

# 18. Final Deliverables

## Data

1. `people.yaml`
2. `organizations.yaml`
3. `labs.yaml`
4. `acquisitions.yaml`
5. `edges.json`
6. `sources.yaml`
7. `conflicts.yaml`
8. `canonical_graph.json`

## Research outputs

9. `research_briefing.md`
10. `intel_siph_lineage.md`
11. `bell_labs_lineage.md`
12. `academic_lab_lineages.md`
13. `stanford_photonics_lineage.md`
14. `china_photonics_lineage.md`
15. `recent_acquisitions.md`
16. `validation_report.md`

## Visual outputs

17. `photonics_lineage_sankey.html`
18. `photonics_lineage_sankey.png`
19. `intel_siph_sankey.png`
20. `bell_labs_sankey.png`
21. `stanford_photonics_sankey.png`
22. `china_photonics_sankey.png`
23. optional interactive person-level graph

---

# 19. Definition of Done

The first major release is complete when:

- at least **100 relevant companies / labs** are represented
- at least **200 people** have validated career histories
- every major startup in the target scope has:
  - founders
  - CTO / technical leadership
  - major upstream affiliations
- every acquisition in the target list has:
  - announcement / close date
  - value if public
  - founder / technical leader lineage
- all Sankey edges are traceable back to named people or explicit corporate events
- all high-confidence edges have A/B evidence
- China-specific claims are individually validated rather than relying on generic "team from..." PR
- the final graph can be regenerated from structured data without manual editing

---

# 20. Immediate Next Actions

1. Freeze the schema.
2. Build the initial company / lab / person seed YAMLs.
3. Run the acquisition-first reverse genealogy.
4. Run Intel SiPh alumni mining in parallel.
5. Run Bell Labs / Acacia / coherent-optics mining in parallel.
6. Run UCSB / Stanford / MIT / Columbia lab-lineage mining in parallel, with Stanford split by group.
7. Run China / Huawei ecosystem research separately with Chinese-language sources.
8. Merge all person histories.
9. Resolve duplicate people / organizations.
10. Render an initial Sankey from only A/B evidence.
11. Inspect the first graph for missing hubs and unexpected weak links.
12. Launch a second discovery pass from newly surfaced people and organizations.

The research should be iterative: the graph itself should drive subsequent search by revealing missing intermediate nodes, improbable jumps, and high-value organizations whose alumni have not yet been fully traced.
