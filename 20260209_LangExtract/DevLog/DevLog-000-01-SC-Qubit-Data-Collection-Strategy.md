# DevLog-000-010: SC Qubit Data Collection Strategy

## Overview

Phase 1 data collection focusing on superconducting qubit fabrication processes. Target: publicly available theses from leading labs, review papers, and fabrication-focused publications.

**Scope:**
- All qubit types (transmon, fluxonium, flux qubits, etc.)
- All fabrication aspects (junction fabrication, cavity fabrication, packaging, materials)
- Historical coverage (2000+) to capture process evolution
- Storage: Local initially, move raw PDFs to NAS after processing

## Leading Labs and Research Groups

### Yale School (Rob Schoelkopf & Michel Devoret)
**Key contributions:** Circuit QED, transmon qubit, 3D cavities, cat qubits

**1st generation (direct students/postdocs):**
- Luigi Frunzio (Yale, fabrication expert)
- Steve Girvin (theory, but influential)
- Liang Jiang (now U Chicago, quantum error correction)
- Shyam Shankar (now at TIFR)
- Mazyar Mirrahimi (now ENS Paris, cat qubits)

**2nd generation:**
- From Frunzio's group: Many fabrication-focused theses
- From Devoret's recent students: Raphaël Lescanne (cat qubits), Uwe von Lüpke

**Thesis access:** Yale Graduate School, Applied Physics department
**Website:** appliedphysics.yale.edu

### UCSB/Google (John Martinis, 2008-2016 era)
**Key contributions:** Xmon qubit, quantum supremacy, high-coherence Al junctions

**1st generation:**
- Andrew Cleland (now U Chicago, previously UCSB)
- Austin Fowler (Google, error correction)
- Rami Barends (Google)
- Julian Kelly (Google)

**2nd generation:**
- From Cleland's group at UCSB: Optomechanics + qubit hybrid work
- Google AI Quantum team publications (detailed supplementary info)

**Note:** Martinis moved to Google (2014), then left (2020). Google Quantum AI continues.
**Thesis access:** UCSB Physics department, ProQuest

### ETH Zurich (Andreas Wallraff)
**Key contributions:** Circuit QED, multiplexed readout, fast gates

**1st generation:**
- Christopher Eichler (now at PSI)
- Simone Gasparinetti
- Jean-Claude Besse

**2nd generation:**
- Systematic documentation in recent theses

**Thesis access:** ETH Research Collection (excellent accessibility)
**Website:** qudev.phys.ethz.ch

### IBM Research
**Key researchers:**
- Jerry Chow (quantum control)
- Jay Gambetta (error correction, theory)
- Hanhee Paik (materials, fabrication)

**Strengths:**
- Extensive patent database (very detailed process specs)
- Regular publications on materials science aspects
- Some researchers have academic affiliations with theses

### MIT (Will Oliver, Terry Orlando)
**Key contributions:** Fluxonium, materials science, Lincoln Lab collaboration

**Thesis access:** MIT DSpace
**Website:** equs.mit.edu

### UC Berkeley (Irfan Siddiqi)
**Key contributions:** Parametric amplifiers, readout, multi-qubit systems

**Thesis access:** UC Berkeley library, ProQuest
**Website:** qnl.berkeley.edu

### Delft (Leo DiCarlo)
**Key contributions:** Multi-qubit gates, surface code implementations

**Thesis access:** TU Delft repository
**Website:** dicarlolab.tudelft.nl

### University of Maryland (Chris Lobb, Fred Wellstood)
**Key contributions:** Flux qubits, materials studies

**Thesis access:** UMD library

### Princeton (Andrew Houck)
**Key contributions:** Lattice models, photonic integration with qubits

**Thesis access:** Princeton DataSpace

### Chinese Institutions
**Better SC qubit groups than USTC:**
- Tsinghua (Luyan Sun, former Martinis postdoc)
- Zhejiang University (Wang Haohua, Zhu Xiaobo - strong fabrication)
- IOP Beijing (Yu Dapeng group)

**Note:** Jianwei Pan (USTC) primarily focuses on photonic quantum computing

## Data Collection Strategy

### Phase 1A: Seed Collection (Week 1-2)

**Goal:** 50-100 documents

**Priority 1 - Key Review Papers (3-5 papers):**
1. "Superconducting Qubits: Current State of Play" - Kjaergaard et al., Annual Review (2020)
   - Most comprehensive recent review
   - Excellent reference list organized by topic
   
2. "Materials in superconducting quantum bits" - Place et al., Nature Materials (2021)
   - Focused on fabrication and materials
   - References are goldmine for process details
   
3. "Superconducting quantum bits" - Devoret & Schoelkopf, Science (2013)
   - Classic review, historical foundation
   - ~1000 citations to branch from

4. "Building logical qubits in a superconducting quantum computing system" - Google, npj Quantum Information (2017)
   - Detailed device fabrication in supplementary info

5. IBM's coherence improvement papers (various, 2015-2023)
   - Materials science focus

**Priority 2 - High-Value Theses (20 theses):**

Distribution:
- 8 from Yale (mix of Schoelkopf/Devoret/Frunzio students)
- 5 from UCSB (Martinis era, 2008-2016)
- 4 from ETH Zurich
- 3 from MIT

Specific recommendations (if accessible):
1. Geerlings (Yale, 2013) - "Improving Coherence of Superconducting Qubits and Resonators"
2. Barends (UCSB, 2009) - "Photon-detecting superconducting resonators"
3. Eichler (ETH, 2013) - "Experimental Characterization of Quantum Microwave Radiation"
4. Reagor (Yale, 2015) - "Superconducting Cavities for Circuit Quantum Electrodynamics"
5. Place (MIT, 2020) - "Materials Engineering for High-Coherence Superconducting Qubits"

**Priority 3 - Targeted Papers (20-30 papers):**

Categories:
- Coherence improvement papers (always have fabrication details)
- New qubit type demonstrations (first transmon, first fluxonium, etc.)
- Junction fabrication methods (Dolan bridge, Manhattan style)
- Substrate and materials studies

### Phase 1B: First-Order Expansion (Week 3-4)

**From review papers:**
- Extract all fabrication-related references (expect 100-200 papers)
- Download those with "Methods" or "Supplementary" sections
- Filter for papers with process details (expect 30-50 useful ones)

**From theses:**
- Check bibliography of fabrication chapters
- Follow citations to process papers
- Identify common "recipe sources" (often cite same foundational papers)

**Lab website crawling:**
- Publications pages: Filter for "fabrication", "coherence", "materials"
- Download all supplementary information
- Alumni lists: Track to find thesis links

### Phase 1C: Second-Order Expansion (Week 5-6)

**Lab genealogy tracking:**
- From 1st gen researchers, find their students' theses
- Track lab moves (e.g., Cleland UCSB to Chicago)
- Identify fabrication specialists vs. measurement specialists

**Citation forward-tracking:**
- Use Google Scholar "Cited by" on key fabrication papers
- Recent papers often improve on older recipes
- Track evolution of specific processes

## Branching Strategies

### Approach 1: Thesis-Centric Collection

**Why theses are valuable:**
- Chapter 2-3 usually: "Fabrication" or "Device Design and Fabrication"
- Much more detail than papers (10-30 pages vs 1-2 paragraphs)
- Often include "recipes that didn't work" - valuable negative data
- Process flow diagrams
- Complete chemical lists with vendors and purities
- Equipment specifications

**Tier 1 (Must-have):**
1. Yale theses (2010-2024) - Yale Graduate School + Applied Physics
2. UCSB theses (2008-2016) - Martinis era
3. ETH Zurich dissertations - ETH Research Collection
4. MIT theses - DSpace

**Tier 2 (High value):**
1. Berkeley theses (Siddiqi group)
2. Delft theses (DiCarlo group)
3. IBM researchers with academic affiliations

### Approach 2: Review Paper Branching

**Strategy:**
- Start with 3-5 key reviews
- Extract all references in "Fabrication" or "Materials" sections
- Follow citations to papers with detailed supplementary info
- Track authors who consistently publish fabrication details

**Key sections to mine:**
- Materials and methods
- Device fabrication
- Coherence mechanisms (often discusses surface treatment)
- Loss tangent studies (substrate preparation)

### Approach 3: Lab Website Crawling

**Target information:**

1. Publications page:
   - Filter for fabrication-related keywords
   - Download supplementary information
   - Track PhD thesis links

2. Group members page:
   - Alumni list to Google Scholar to find thesis
   - Current students for recent papers

3. Resources/Protocols (rare but valuable):
   - Some groups share recipes publicly
   - Cleanroom SOPs sometimes linked

**Specific lab websites:**
- Yale Applied Physics: appliedphysics.yale.edu
- ETH Zurich Wallraff group: qudev.phys.ethz.ch
- Berkeley Siddiqi group: qnl.berkeley.edu
- MIT EQuS: equs.mit.edu
- Delft DiCarlo: dicarlolab.tudelft.nl

### Approach 4: Targeted Paper Collection

**Search categories:**

1. Coherence improvement papers:
   - "Reducing loss in superconducting qubits"
   - "Surface treatment" + "transmon"
   - "TLS" (two-level systems) + "superconducting"

2. New qubit demonstrations:
   - First transmon paper (Koch et al., 2007)
   - First 3D transmon (Paik et al., 2011)
   - Fluxonium papers (Manucharyan et al., 2009+)

3. Junction fabrication:
   - "Dolan bridge" + "Josephson"
   - "Manhattan style" + "junction"
   - "Al/AlOx/Al" + fabrication

4. Substrate and materials:
   - "Sapphire" + "qubit"
   - "Silicon" + "superconducting qubit"
   - "Tantalum" + "qubit" (recent trend)

**Google Scholar queries:**
```
"superconducting qubit" fabrication methods
"transmon" "junction fabrication"
"Josephson junction" "Dolan bridge" OR "Manhattan"
"superconducting qubit" coherence improvement
"Al/AlOx/Al" junction fabrication
```

**arXiv queries:**
```
cat:quant-ph AND (fabrication OR "process flow" OR "device fabrication")
au:Schoelkopf OR au:Devoret OR au:Martinis
```

## Data Quality Indicators

### High-value documents:
- Supplementary info >10 pages
- Contains tables of process parameters
- Has process flow diagrams
- Mentions specific equipment models
- Lists chemical vendors/purities
- Discusses "what we tried that didn't work"
- Includes SEM/optical images of fabrication steps

### Medium-value:
- Brief methods section with key parameters
- References to detailed protocols elsewhere
- Partial process flows
- Equipment types without models

### Low-value (but keep for completeness):
- "Standard fabrication techniques"
- "Devices fabricated as described in [ref]"
- No parameter details
- Only final device characterization

## Document Organization

### Local storage structure:

```
semiconductor_processing_dataset/
├── raw_documents/
│   ├── papers/
│   │   └── superconducting_qubits/
│   │       ├── transmon/
│   │       ├── fluxonium/
│   │       ├── flux_qubits/
│   │       ├── materials_studies/
│   │       └── reviews/
│   ├── supplementary/
│   │   └── [same structure]
│   └── theses/
│       ├── yale/
│       ├── ucsb/
│       ├── eth_zurich/
│       ├── mit/
│       ├── berkeley/
│       ├── delft/
│       └── others/
│
└── processed_documents/
    └── metadata/
        └── [document_id].json
```

### Metadata tracking:

Each document gets a JSON metadata file:
```json
{
  "document_id": "geerlings_yale_2013",
  "source_type": "phd_thesis",
  "institution": "Yale University",
  "author": "Geerlings, K.",
  "advisor": "Schoelkopf, R.",
  "year": 2013,
  "title": "Improving Coherence of Superconducting Qubits and Resonators",
  "url": "...",
  "download_date": "2026-02-10",
  "qubit_types": ["transmon", "3D_cavity"],
  "fabrication_topics": ["junction_fab", "cavity_machining", "surface_treatment"],
  "has_process_flow": true,
  "has_chemical_list": true,
  "fabrication_chapter": "Chapter 3",
  "page_range": "45-78",
  "quality_assessment": "high_value",
  "notes": "Excellent Dolan bridge recipe, detailed surface preparation"
}
```

## Search Strategies

### For theses:

**Yale:**
- Search: "Yale Graduate School" + "Applied Physics" + advisor name
- ProQuest: institution:"Yale University" AND advisor:"Schoelkopf"

**ETH Zurich:**
- ETH Research Collection: https://www.research-collection.ethz.ch/
- Search by advisor or group

**MIT:**
- DSpace: https://dspace.mit.edu/
- Department of Physics or EECS

**UCSB:**
- ProQuest or UCSB library
- Physics department, 2008-2016 timeframe

### For papers:

**Foundational papers to track:**
- Koch et al. (2007) - First transmon
- Paik et al. (2011) - First 3D transmon
- Barends et al. (2013) - Coherence improvements
- Manucharyan et al. (2009) - Fluxonium

**Recent materials papers:**
- Place et al. (2021) - Tantalum qubits
- Wang et al. (2022) - Surface loss mechanisms
- Any paper claiming coherence records

## Collection Progress (Updated 2026-02-10)

### Phase 1A Status: COMPLETE — 207 documents, 2.8 GB

Collection executed via parallel subagents across 3 waves on 2026-02-10.

#### Theses Collected: 111

| Institution | Count | Source | Notes |
|-------------|-------|--------|-------|
| Yale (RSL + Qulab) | 56 | rsl.yale.edu/theses, qulab.eng.yale.edu/theses/ | 31 Schoelkopf + 24 Devoret + Frunzio (2006). Spans 2005–2025. 1 failed (Roy — dead CDN link) |
| UChicago/Stanford (Schuster) | 23 | schusterlab.stanford.edu/static/pdfs/ | Includes Earnest, Oriani (cavity fab), Anferov (Nb trilayer) |
| UCSB (Martinis) | 10 | ProQuest / UCSB library | Kelly, Chen, Neill, O'Malley, White, Sank, Lucero, Bialczak, Ansmann, Raab (2009–2018) |
| ETH Zurich (Wallraff) | 8 | research-collection.ethz.ch | Eichler (2013), Norris (2024), Reuer (2024), Lacroix (2025), + 4 more. 3 initially blocked by cookie wall |
| UC Berkeley (Siddiqi) | 5 | escholarship.org | Kreikebaum, Slichter, Weber, Ramasesh + 1 more |
| MIT (Oliver) | 4 | dspace.mit.edu | Kim (2025), Lienhard, Karamlou, Sung |
| TU Delft (DiCarlo) | 2 | repository.tudelft.nl | Dickel, Marques |
| TIFR | 2 | Broad search | Shankar group |
| Paris | 1 | Broad search | ENS Paris |

**Key high-value theses for fabrication:**
- Frunzio (Yale, 2006) — "Design and Fabrication of Superconducting Circuit for Amplification and Processing of Quantum Signal" — the fabrication expert's own thesis
- Geerlings (Yale, 2013) — coherence improvements, detailed Dolan bridge recipe
- Reagor (Yale, 2015) — 3D cavity fabrication
- Ganjam (Yale, 2023) — tantalum on sapphire, materials optimization
- Kelly (UCSB, 2015) — fault-tolerant qubit fabrication (Martinis group)
- Oriani (UChicago, 2022) — Nb cavity fab, high-Q optimization
- Anferov (UChicago, 2024) — mm-wave devices, Nb trilayer junctions
- Eichler (ETH, 2013) — quantum microwave radiation

#### Papers Collected: 96

**Wave 1 (7 papers) — Seed collection:**
- 3 review papers: Kjaergaard (2020), Krantz (2019), Lee (2022)
- 2 transmon: Houck (2008, 2009)
- 1 fluxonium: Nguyen (2019)
- 1 materials: Bal (2024)

**Wave 2 — Seminal papers (9 papers):**
- Koch (2007) — first transmon
- Paik (2011) — first 3D transmon
- Manucharyan (2009) — fluxonium
- Barends (2013) — Xmon
- Place (2021) — tantalum qubits (0.3 ms)
- Devoret & Schoelkopf (2013) — SC circuits outlook
- Osman (2021) — simplified JJ fabrication
- Somoroff (2023) — millisecond fluxonium
- Wang (2022) — 0.5 ms transmon

**Wave 2 — Recent fabrication (8 papers):**
- Ganjam (2024) — ms coherence on Ta/sapphire
- Melville (2020) — TiN vs Al dielectric loss
- Smirnov (2023) — wiring surface loss
- Wang (2024) — overlap JJ fluxonium, wafer-scale
- Dunsworth (2017) — JJ fab capacitive loss
- Bu (2024) — Ta airbridges
- Rosenberg (2019) — 3D integration/packaging
- Murray (2021) — materials review (IBM)

**Wave 3 — Fabrication processes & materials (14 papers):**
- Nersisyan (2019) — Rigetti manufacturing process
- Wu (2017) — overlap junctions (NIST)
- Muschinske (2023) — Dolan vs Manhattan JJ uniformity (MIT LL)
- Fox (2024) — ammonium fluoride etch (Rigetti)
- Tsioutsios (2020) — silicon shadow masks (Yale)
- Kamal (2016) — substrate annealing (MIT LL)
- Bruno (2015) — deep etching for loss reduction (TU Delft)
- Altoe (2022) — circuit loss localization (LBNL)
- Gruenhaupt (2024) — Nb thin film structure (Google)
- Chang (2013) — TiN coherence improvement (MIT LL)
- de Leon (2022) — Nb surface oxide characterization (Princeton)
- Kurter (2024) — near-ms transmon (UMD/NIST)
- Niedzielski (2022) — TSV substrates (MIT LL)
- Osman (2023) — JJ reproducibility shadow evaporation (Chalmers)

**Wave 3 — Packaging & integration (12 papers):**
- Foxen (2018) — indium bump interconnects (Google)
- Kosen (2022) — flip-chip building blocks (Chalmers)
- Yost (2020) — qubits + TSV integration (MIT LL)
- Gold (2021) — entanglement across separate dies (Rigetti)
- Mallek (2021) — TSV fabrication process (MIT LL)
- Huang (2021) — 24-port microwave package (MIT)
- Kreikebaum (2019) — microwave packaging (Berkeley)
- Krinner (2019) — cryogenic setups for 100+ qubits (ETH)
- Van Damme (2024) — 300mm CMOS fab, 98% yield (imec)
- Field (2023) — multi-chip tunable coupler (Rigetti)
- Conner (2021) — non-galvanic flip-chip (UChicago)
- Simbierowicz (2024) — indium microspheres flip-chip (VTT)

**Schuster lab papers (42 papers):**
- 7 fabrication: Shearrow (2018, ALD TiN), Frunzio (2005, cQED fab), Lee (2019, 2D JJ barriers), Anferov (2024, Nb trilayer), Satzinger (2019, flip-chip), + 2 others
- 15 resonators/cavities: Oriani (2024, Nb coaxial Q>1.5B), Chakram (2021, seamless cavities), Romanenko (2017, Nb degradation), + 12 others
- 8 materials/device physics
- 12 qubit design

#### Paper Categories (with counts)

| Category | Count |
|----------|-------|
| transmon | 24 |
| resonators | 15 |
| qubit_design | 12 |
| materials_studies | 11 |
| materials | 8 |
| fabrication + fabrication_processes | 13 |
| fluxonium | 5 |
| reviews | 5 |
| substrate_preparation | 2 |
| packaging | 1 |

### Known Issues

1. **Duplicate subdirectories:** `materials_studies/` vs `materials/`, `fabrication/` vs `fabrication_processes/` — created by different agents, needs consolidation
2. **Institution naming:** `eth_zurich/` (empty) vs `eth/` (8 PDFs) — needs rename
3. **Duplicate papers:** A few papers were downloaded by multiple agents under different document IDs; 4 were caught and removed via SHA256 matching, but more may exist
4. **collection_tracker.csv** partially populated but JSONL manifest is the canonical tracker
5. **Roy (Yale, 2016)** thesis — dead CDN link, 0 bytes. May need manual download

### What Was NOT Collected

- **Yale theses from ProQuest** — only open-access from lab websites (but got 56 from RSL+Qulab, which is comprehensive)
- **Supplementary information files** — not yet targeted
- **Patents** — not yet targeted
- **Chinese institution theses** — Tsinghua, Zhejiang, IOP Beijing not found publicly
- **Princeton theses** — Houck group theses not found on public repositories

## Next Steps

1. **Consolidate directory structure** — merge duplicate subdirectories, fix eth naming
2. **Deduplicate** — run SHA256 comparison across all PDFs to find remaining duplicates
3. **Begin Phase 2: Text extraction** — test docling, marker, GROBID on sample papers
4. **Supplementary info** — collect SI for high-value papers (Place 2021, Ganjam 2024, etc.)
5. **Quality assessment** — manually review a sample of downloaded PDFs to verify they are complete and not HTML error pages
6. **Build initial knowledge base** — extract chemicals, equipment, abbreviations from the highest-value theses (Frunzio, Geerlings, Kelly, Oriani)

## Notes

- Storage: 2.8 GB local, will move to NAS after processing and ingestion
- Automation: Parallel subagents worked well for bulk collection; 8 agents total across waves
- Focus: All fabrication aspects, all qubit types — well covered
- Timeline: Historical coverage achieved (2005–2025 for theses, 2007–2024 for papers)
- Access: Lab websites (rsl.yale.edu, qulab.eng.yale.edu, schusterlab.stanford.edu) were by far the best sources — directly list thesis PDFs
- ETH Research Collection has cookie/session challenges that block curl; some theses required alternate URL patterns
- Tracking: JSONL manifest (`manifest_documents.jsonl`) is canonical; `collection_tracker.csv` is secondary

