# DevLog-000: Data Sources and Organization Strategy

## Project Overview

Extracting academic semiconductor processing information using LangExtract, focusing on:
- 2D materials
- Superconducting qubits and quantum computing
- Silicon / TFLN photonics and optomechanics
- MEMS

## Data Sources

### 1. Primary Academic Sources

**Journal Articles:**
- Nature/Science family: Nature Nanotechnology, Nature Physics, Nature Photonics
- APS journals: Physical Review Applied, Physical Review X, PRB
- ACS journals: Nano Letters, ACS Photonics, ACS Applied Materials & Interfaces
- IEEE journals: IEEE Electron Device Letters, IEEE Photonics Technology Letters
- Optica: Optica, Optics Express, Optics Letters
- APL: Applied Physics Letters, APL Photonics

**Preprint servers:**
- arXiv (cond-mat, quant-ph, physics.optics)
- ChemRxiv (for materials chemistry)

### 2. Supplementary Information
- Often contains detailed fabrication recipes
- Process flow diagrams
- More complete chemical formulations
- Annealing/deposition parameters

### 3. Theses and Dissertations
- PhD theses often have comprehensive process chapters
- More detailed than papers due to space constraints
- ProQuest, university repositories

### 4. Patents
- Detailed process specifications
- Google Patents, USPTO, EPO
- Often more complete chemical compositions

## Processing Challenges

### 1. Chemical Formula Extraction

**Problems with standard tools:**
```
Common issues:
- "SiO₂" → "SiO2" or "SiO 2" (subscript handling)
- "Al₂O₃" → "Al2O3" or fragmented
- Chemical names: "tetramethylammonium hydroxide" vs "TMAH"
- Ratios: "HF:H₂O (1:100)" → parsing errors
- Concentrations: "49% HF" vs "HF (49 wt%)"
- Complex etchants: "BOE (7:1)" or "Piranha (3:1 H₂SO₄:H₂O₂)"
```

**Specific challenges:**
- 2D materials: "MoS₂", "WSe₂", "hBN" - subscripts and chemical notation
- Precursors: "SiH₄", "WF₆", "TMGa" - gas phase chemistry
- Organic chemicals: Long IUPAC names, trade names (e.g., "PMMA 950K A4")
- Mixed solutions: Multi-component etchants with ratios

### 2. Process Parameter Extraction

**Temperature variations:**
```
- "500°C" vs "500 °C" vs "500C"
- "RT" (room temperature)
- Ramp rates: "10°C/min"
- Ranges: "400-500°C"
```

**Pressure/vacuum:**
```
- Units: Torr, mTorr, Pa, mbar
- Scientific notation: "1×10⁻⁶ Torr"
- "UHV", "HV" (qualitative)
```

**Time:**
```
- "5 min" vs "5 minutes" vs "5min"
- "overnight" (ambiguous)
- Multi-step: "2 min + 30 s"
```

### 3. Equipment and Tool Names

```
- Vendor-specific: "Oxford Plasmalab 100", "Heidelberg µPG 101"
- Generic: "PECVD", "ICP-RIE", "e-beam evaporator"
- Abbreviations: "EBL" vs "electron beam lithography"
- Model numbers with special characters
```

### 4. PDF Extraction Issues

**Multi-column layouts:**
- Process flows spanning columns
- Tables breaking across pages

**Figures and captions:**
- Process flow diagrams (need OCR + diagram understanding)
- SEM images with scale bars
- Recipes in figure captions

**Tables:**
- Complex process tables with merged cells
- Parameter matrices
- Comparison tables across different recipes

**Mathematical notation:**
- Thickness: "100 nm ± 5 nm"
- Rates: "5 Å/s"
- Formulas in process descriptions

## Data Organization Strategy

### Proposed Hierarchical Structure

```
semiconductor_processing_dataset/
├── raw_documents/
│   ├── papers/
│   │   ├── 2d_materials/
│   │   │   ├── mos2_fabrication/
│   │   │   ├── graphene_transfer/
│   │   │   └── heterostructures/
│   │   ├── superconducting_qubits/
│   │   │   ├── transmon/
│   │   │   ├── fluxonium/
│   │   │   └── materials_loss/
│   │   ├── photonics/
│   │   │   ├── silicon_photonics/
│   │   │   ├── tfln/
│   │   │   └── optomechanics/
│   │   └── mems/
│   ├── supplementary/
│   │   └── [same structure]
│   ├── theses/
│   └── patents/
│
├── processed_documents/
│   ├── text_extracted/
│   │   ├── docling/          # Docling output
│   │   ├── marker/            # Marker output
│   │   ├── grobid/            # GROBID output (good for academic papers)
│   │   └── manual_corrections/
│   └── metadata/
│       └── [paper_id].json    # DOI, authors, journal, year, etc.
│
├── annotations/
│   ├── process_flows/         # Extracted process sequences
│   ├── chemicals/             # Chemical inventory
│   ├── equipment/             # Tool/equipment database
│   └── parameters/            # Process parameters
│
├── knowledge_base/
│   ├── chemical_database.json     # Normalized chemical names
│   ├── equipment_database.json    # Tool specifications
│   ├── abbreviations.json         # Domain-specific abbreviations
│   └── process_ontology.json      # Process step taxonomy
│
└── extraction_configs/
    ├── prompts/
    │   ├── chemical_extraction.py
    │   ├── process_flow_extraction.py
    │   ├── parameter_extraction.py
    │   └── equipment_extraction.py
    └── examples/
        └── few_shot_examples/
```

### Metadata Schema

```json
{
  "document_id": "doi_10.1038_s41586_2023_xxxxx",
  "source_type": "journal_article",
  "domain": ["2d_materials", "photonics"],
  "title": "...",
  "authors": [...],
  "journal": "Nature",
  "year": 2023,
  "doi": "10.1038/...",
  "arxiv_id": "2301.xxxxx",
  "keywords": ["MoS2", "photonic_crystal", "PECVD"],
  "has_supplementary": true,
  "extraction_status": {
    "text_extracted": true,
    "chemicals_extracted": true,
    "process_flow_extracted": false
  },
  "quality_flags": {
    "ocr_quality": "high",
    "table_extraction": "manual_review_needed",
    "formula_preservation": "good"
  }
}
```

### Extraction Schema Design

**Process Step Schema:**
```python
{
  "step_number": 1,
  "process_type": "deposition",  # deposition, etching, lithography, annealing, etc.
  "sub_type": "PECVD",
  "materials": [
    {
      "chemical": "SiH4",
      "role": "precursor",
      "flow_rate": {"value": 20, "unit": "sccm"}
    },
    {
      "chemical": "N2O",
      "role": "oxidizer",
      "flow_rate": {"value": 100, "unit": "sccm"}
    }
  ],
  "parameters": {
    "temperature": {"value": 300, "unit": "°C"},
    "pressure": {"value": 1, "unit": "Torr"},
    "power": {"value": 20, "unit": "W"},
    "time": {"value": 10, "unit": "min"}
  },
  "equipment": {
    "type": "PECVD",
    "model": "Oxford Plasmalab 100",
    "chamber": "SiO2"
  },
  "result": {
    "material": "SiO2",
    "thickness": {"value": 100, "unit": "nm"},
    "quality_metrics": {...}
  },
  "source_text": "original text from paper",
  "source_location": "page 3, methods section"
}
```

## Recommended Approach

### Phase 1: Document Collection & Preprocessing
1. Start small: 20-30 papers from each domain
2. Test extraction tools:
   - Docling (good for structure)
   - Marker (good for formulas)
   - GROBID (academic paper structure)
   - PyMuPDF/pdfplumber (fallback)
3. Compare outputs for chemical formula preservation
4. Manual curation of 5-10 "gold standard" papers

### Phase 2: LangExtract Configuration
1. Create domain-specific prompts:
   - Chemical extraction with formula preservation
   - Process flow extraction
   - Parameter extraction with units
2. Build few-shot examples from gold standard papers
3. Iterative refinement based on extraction quality

### Phase 3: Knowledge Base Building
1. Chemical normalization: Map variants to canonical forms
2. Abbreviation expansion: Build domain dictionary
3. Equipment database: Standardize tool names
4. Process taxonomy: Categorize process types

### Phase 4: Validation & Quality Control
1. Cross-validation: Compare extractions across similar papers
2. Expert review: Sample-based validation
3. Consistency checks: Parameter ranges, chemical compatibility
4. Completeness metrics: Coverage of process steps

## Specific Recommendations for Your Domains

### 2D Materials
- Focus on transfer processes (PMMA-assisted, dry transfer)
- Exfoliation conditions (scotch tape, gold-assisted)
- CVD growth (precursors, temperatures, substrates)
- Encapsulation (hBN stacking, polymer protection)

### SC Qubits
- Substrate preparation (cleaning, surface treatment)
- Josephson junction fabrication (Dolan bridge, Manhattan)
- Metal deposition (Al, Nb, Ta - purity critical)
- Etch chemistry (avoiding contamination)

### Silicon/TFLN Photonics
- Waveguide etching (ICP-RIE recipes, sidewall roughness)
- Cladding deposition (oxide quality)
- Poling conditions (for TFLN)
- Coupling structures (grating couplers, edge couplers)

### MEMS/Optomechanics
- Release processes (HF vapor, XeF2)
- Stress control (annealing, deposition conditions)
- Sacrificial layers (removal chemistry)
- Mechanical property optimization

## Next Steps

1. Create a starter script to organize document collection
2. Set up example LangExtract prompts for chemical/process extraction
3. Build a document metadata tracking system
4. Create a comparison script to test different PDF extraction tools on sample papers

