# Quick Start Guide

## Directory Structure Overview

This dataset is organized for extracting semiconductor fabrication processes from academic literature.

## Where to Put Files

### Downloaded PDFs

**Journal papers:**
```
raw_documents/papers/superconducting_qubits/[category]/
```
Categories: transmon, fluxonium, flux_qubits, materials_studies, reviews

**Supplementary information:**
```
raw_documents/supplementary/superconducting_qubits/[category]/
```

**PhD theses:**
```
raw_documents/theses/[institution]/
```
Institutions: yale, ucsb, eth_zurich, mit, berkeley, delft, others

**Patents:**
```
raw_documents/patents/
```

### Naming Convention

**Papers:** `firstauthor_year_journal_keyword.pdf`
Example: `barends_2013_nature_coherence.pdf`

**Theses:** `lastname_firstname_institution_year.pdf`
Example: `geerlings_kyle_yale_2013.pdf`

**Supplementary:** `[paper_name]_SI.pdf`
Example: `barends_2013_nature_coherence_SI.pdf`

## Workflow

### 1. Download Document
Place PDF in appropriate raw_documents/ subdirectory

### 2. Create Metadata
Copy `processed_documents/metadata/TEMPLATE.json` and fill in:
```bash
cp processed_documents/metadata/TEMPLATE.json \
   processed_documents/metadata/geerlings_kyle_yale_2013.json
```

Edit the JSON file with document details.

### 3. Log Collection Progress
Append one event to:
`processed_documents/metadata/collection_attempts.jsonl`

Keep one latest-state row per document in:
`processed_documents/metadata/manifest_documents.jsonl`

Follow:
`COLLECTION_SUBAGENT_PROTOCOL.md`

### 4. Extract Text
Run text extraction tools:
```bash
# Docling
docling input.pdf -o processed_documents/text_extracted/docling/

# Marker
marker input.pdf -o processed_documents/text_extracted/marker/

# GROBID
# (command depends on GROBID setup)
```

### 5. Run LangExtract
Use extraction configs in `extraction_configs/prompts/`

Output goes to `annotations/[category]/`

### 6. Build Knowledge Base
Add extracted entities to:
- `knowledge_base/chemical_database.json`
- `knowledge_base/equipment_database.json`
- `knowledge_base/abbreviations.json`

## Current Focus: SC Qubits Phase 1A

### Priority Documents (Week 1-2)

**Review papers (3-5):**
1. Kjaergaard et al. (2020) - Annual Review
2. Place et al. (2021) - Nature Materials
3. Devoret & Schoelkopf (2013) - Science

**Theses (20 total, 8 from Yale):**
1. Geerlings (Yale, 2013)
2. Reagor (Yale, 2015)
3. Place (MIT, 2020)
4. Barends (UCSB, 2009)
5. Eichler (ETH, 2013)

**Targeted papers (20-30):**
- Coherence improvement papers
- Junction fabrication methods
- Materials studies

### Where to Find Theses

**Yale:**
- Search: Yale Graduate School + Applied Physics
- ProQuest: institution:"Yale University" AND advisor:"Schoelkopf"

**ETH Zurich:**
- ETH Research Collection: https://www.research-collection.ethz.ch/

**MIT:**
- DSpace: https://dspace.mit.edu/

**UCSB:**
- ProQuest or UCSB library (Physics, 2008-2016)

## Quick Commands

### Check structure:
```bash
find semiconductor_processing_dataset -type d | sort
```

### Count documents by type:
```bash
find raw_documents/papers -name "*.pdf" | wc -l
find raw_documents/theses -name "*.pdf" | wc -l
```

### List all metadata files:
```bash
ls processed_documents/metadata/*.json
```

### Check collection progress:
```bash
wc -l processed_documents/metadata/collection_attempts.jsonl
wc -l processed_documents/metadata/manifest_documents.jsonl
```

## File Locations Reference

| Content | Location |
|---------|----------|
| Main README | `README.md` |
| Thesis guidelines | `raw_documents/theses/README.md` |
| Knowledge base docs | `knowledge_base/README.md` |
| Metadata template | `processed_documents/metadata/TEMPLATE.json` |
| Collection protocol | `COLLECTION_SUBAGENT_PROTOCOL.md` |
| Attempt log | `processed_documents/metadata/collection_attempts.jsonl` |
| Document manifest | `processed_documents/metadata/manifest_documents.jsonl` |
| Directory listing | `STRUCTURE.txt` |
| This guide | `QUICK_START.md` |

## Next Steps

1. Download first batch of documents (3-5 reviews + 5 theses)
2. Create metadata files for each
3. Test text extraction tools
4. Compare extraction quality
5. Begin LangExtract configuration

## Questions?

Refer to:
- DevLog-000: Overall strategy
- DevLog-000-010: SC qubit collection strategy
- DevLog-000-020: Directory structure details
