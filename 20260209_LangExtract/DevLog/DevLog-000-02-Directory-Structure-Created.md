# DevLog-000-020: Directory Structure Created

## Date
2026-02-10

## Summary

Created complete directory structure for semiconductor processing dataset as proposed in DevLog-000 and DevLog-000-010.

## Directory Structure Created

### Top Level
```
semiconductor_processing_dataset/
├── raw_documents/
├── processed_documents/
├── annotations/
├── knowledge_base/
├── extraction_configs/
├── README.md
└── COLLECTION_SUBAGENT_PROTOCOL.md
```

### Raw Documents (raw_documents/)

**Papers by domain:**
- papers/superconducting_qubits/
  - transmon/
  - fluxonium/
  - flux_qubits/
  - materials_studies/
  - reviews/
- papers/2d_materials/
  - mos2_fabrication/
  - graphene_transfer/
  - heterostructures/
- papers/photonics/
  - silicon_photonics/
  - tfln/
  - optomechanics/
- papers/mems/

**Supplementary information:**
- supplementary/ (mirrors papers/ structure)

**Theses by institution:**
- theses/
  - yale/
  - ucsb/
  - eth_zurich/
  - mit/
  - berkeley/
  - delft/
  - others/

**Patents:**
- patents/

### Processed Documents (processed_documents/)

**Text extraction outputs:**
- text_extracted/
  - docling/
  - marker/
  - grobid/
  - manual_corrections/

**Metadata:**
- metadata/
  - TEMPLATE.json (template for document metadata)

### Annotations (annotations/)

Structured extraction outputs:
- process_flows/
- chemicals/
- equipment/
- parameters/

### Knowledge Base (knowledge_base/)

Normalized databases:
- chemical_database.json (initialized)
- equipment_database.json (initialized)
- abbreviations.json (initialized)
- process_ontology.json (initialized)
- README.md (documentation)

### Extraction Configs (extraction_configs/)

LangExtract configuration:
- prompts/
- examples/
  - few_shot_examples/

## Files Created

### Documentation
1. semiconductor_processing_dataset/README.md
   - Complete workflow documentation
   - Directory structure explanation
   - Data quality guidelines

2. raw_documents/theses/README.md
   - Thesis collection guidelines
   - Naming conventions
   - Priority thesis list
   - Access information

3. knowledge_base/README.md
   - Knowledge base structure
   - Building strategy
   - Usage guidelines

### Templates
1. processed_documents/metadata/TEMPLATE.json
   - Complete metadata schema
   - All required and optional fields
   - Example values

### Tracking
1. processed_documents/metadata/manifest_documents.jsonl
   - Canonical latest-state manifest (one row per document_id)
2. processed_documents/metadata/collection_attempts.jsonl
   - Append-only attempt log (attempted, failed, succeeded, blocked, skipped)
3. processed_documents/metadata/README.md
   - Logging schema and update rules
4. COLLECTION_SUBAGENT_PROTOCOL.md
   - Required subagent logging and metadata practices

### Knowledge Base Files
1. chemical_database.json (empty, initialized)
2. equipment_database.json (empty, initialized)
3. abbreviations.json (empty, initialized)
4. process_ontology.json (empty, initialized)

## Total Directories Created

52 directories total:
- 19 for papers (main + supplementary)
- 7 for theses
- 1 for patents
- 4 for text extraction
- 1 for metadata
- 4 for annotations
- 1 for knowledge base
- 2 for extraction configs
- Plus subdirectories

## Next Steps

1. Begin Phase 1A: Seed Collection
   - Download 3-5 key review papers
   - Identify 20 publicly accessible theses
   - Download targeted papers from key groups

2. Create metadata files for each document
   - Use TEMPLATE.json as starting point
   - Track latest status in manifest_documents.jsonl
   - Append all attempts to collection_attempts.jsonl

3. Test text extraction tools
   - Compare docling, marker, grobid outputs
   - Evaluate chemical formula preservation
   - Select best tool for each document type

4. Begin building knowledge base
   - Extract common chemicals from first batch
   - Identify equipment from theses
   - Build initial abbreviation list

## Storage Notes

- All files currently local in: 20260209_LangExtract/semiconductor_processing_dataset/
- After processing and ingestion, raw PDFs will be moved to NAS
- Processed documents and annotations remain local for analysis
- 20 TB NAS available for long-term storage

## Focus

Phase 1 focus: Superconducting qubits
- All qubit types (transmon, fluxonium, flux qubits)
- All fabrication aspects
- Historical coverage (2000+) for process evolution
- Publicly available theses from lab websites
