# PhD Theses Collection

This directory contains PhD theses and dissertations with detailed fabrication information.

## Organization by Institution

### yale/
Yale University Applied Physics theses
- Focus: Schoelkopf, Devoret, Frunzio groups
- Topics: Circuit QED, transmon, 3D cavities, cat qubits
- Access: Yale Graduate School, ProQuest

### ucsb/
UC Santa Barbara Physics theses
- Focus: Martinis group (2008-2016 era)
- Topics: Xmon, high-coherence junctions, quantum supremacy
- Access: UCSB library, ProQuest

### eth_zurich/
ETH Zurich dissertations
- Focus: Wallraff group
- Topics: Circuit QED, multiplexed readout, fast gates
- Access: ETH Research Collection (https://www.research-collection.ethz.ch/)

### mit/
MIT theses
- Focus: Oliver, Orlando groups
- Topics: Fluxonium, materials science, Lincoln Lab collaboration
- Access: MIT DSpace (https://dspace.mit.edu/)

### berkeley/
UC Berkeley theses
- Focus: Siddiqi group
- Topics: Parametric amplifiers, readout, multi-qubit systems
- Access: UC Berkeley library, ProQuest

### delft/
TU Delft theses
- Focus: DiCarlo group
- Topics: Multi-qubit gates, surface code implementations
- Access: TU Delft repository

### others/
Theses from other institutions
- University of Maryland (Lobb, Wellstood groups)
- Princeton (Houck group)
- Tsinghua, Zhejiang, IOP Beijing (Chinese institutions)
- Other collaborating institutions

## Naming Convention

Format: `lastname_firstname_institution_year.pdf`

Examples:
- `geerlings_kyle_yale_2013.pdf`
- `barends_rami_ucsb_2009.pdf`
- `eichler_christopher_eth_2013.pdf`
- `place_alexander_mit_2020.pdf`

## Metadata Requirements

Each thesis should have a corresponding JSON file in:
`semiconductor_processing_dataset/processed_documents/metadata/lastname_firstname_institution_year.json`

Required fields:
```json
{
  "document_id": "geerlings_kyle_yale_2013",
  "source_type": "phd_thesis",
  "institution": "Yale University",
  "department": "Applied Physics",
  "author": "Geerlings, Kyle",
  "advisor": "Schoelkopf, Robert J.",
  "year": 2013,
  "title": "Improving Coherence of Superconducting Qubits and Resonators",
  "source_path": "semiconductor_processing_dataset/raw_documents/theses/yale/geerlings_kyle_yale_2013.pdf",
  "url": "https://...",
  "download_date": "2026-02-10",
  "access_type": "public_repository",
  "license_or_terms": "institutional_repository_terms",
  "file_sha256": "optional_sha256_hash",
  "qubit_types": ["transmon", "3D_cavity"],
  "fabrication_topics": ["junction_fab", "cavity_machining", "surface_treatment"],
  "has_process_flow": true,
  "has_chemical_list": true,
  "fabrication_chapter": "Chapter 3",
  "page_range": "45-78",
  "quality_assessment": "high_value",
  "quality_flags": {
    "ocr_quality": "unknown",
    "table_extraction": "unknown",
    "formula_preservation": "unknown"
  },
  "extraction_status": {
    "text_extracted": false,
    "chemicals_extracted": false,
    "process_flow_extracted": false
  },
  "notes": "Excellent Dolan bridge recipe, detailed surface preparation"
}
```

## Priority Theses (if accessible)

### High Priority:
1. Geerlings (Yale, 2013) - Coherence improvements
2. Barends (UCSB, 2009) - Photon detection, fabrication
3. Eichler (ETH, 2013) - Quantum microwave radiation
4. Reagor (Yale, 2015) - 3D cavity fabrication
5. Place (MIT, 2020) - Materials engineering, tantalum

### Medium Priority:
- Any thesis from Frunzio group (fabrication expert)
- Martinis group theses (2008-2016)
- Recent ETH Zurich theses (systematic documentation)
- MIT fluxonium theses

## Why Theses are Valuable

- Fabrication chapters: 10-30 pages vs 1-2 paragraphs in papers
- Complete process flows with diagrams
- Detailed chemical lists (vendors, purities, concentrations)
- Equipment specifications (models, settings)
- "What didn't work" - negative results
- Student perspective - more pedagogical
- Less space constraints than journal articles

## Access Notes

- Focus on publicly available theses from lab websites
- Check institutional repositories first
- ProQuest requires institutional access
- Some groups link theses directly on their websites
- Contact authors if thesis is not publicly available
- Collection workers must follow `semiconductor_processing_dataset/COLLECTION_SUBAGENT_PROTOCOL.md`

## Collection Status

Track collection progress in a repository-local manifest first, then mirror to a spreadsheet if needed:
- Institution
- Author
- Year
- Title
- Download status
- Quality assessment
- Extraction status

Recommended manifest file:
- `semiconductor_processing_dataset/processed_documents/metadata/manifest_documents.jsonl`
